from __future__ import annotations

import logging
import multiprocessing as mp
import re
from queue import Empty
from urllib.parse import urljoin, urlsplit, urlunsplit

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.linkextractors import LinkExtractor
from scrapy.utils.project import get_project_settings
from scrapy_playwright.page import PageMethod


LOGGER = logging.getLogger(__name__)

EMAIL_REGEX = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
REJECTED_LOCAL_PART_PREFIXES: tuple[str, ...] = (
    "example", "test", "noreply", "no-reply"
)
PRIORITY_PATHS: tuple[str, ...] = (
    "/contact", "/about", "/contact-us", "/about-us",
    "/privacy-policy", "/terms-of-service",
)
_PLAYWRIGHT_PAGE_METHODS: list[PageMethod] = [
    PageMethod("wait_for_load_state", "domcontentloaded"),
    PageMethod("wait_for_timeout", 500),
]

# Reject URLs whose path contains URL-encoded HTML tags — sign of a broken
# server-side error leaking into page content (e.g. WordPress PHP warnings).
_GARBAGE_URL_RE = re.compile(r"%3[Cc]|%3[Ee]|<|>|\bphp\b.*deprecated", re.IGNORECASE)


def _normalize_url(url: str) -> str:
    parts = urlsplit(url)
    path = (parts.path or "/").rstrip("/") or "/"
    return urlunsplit((parts.scheme, parts.netloc.lower(), path, "", ""))


def _is_garbage_url(url: str) -> bool:
    """Return True for URLs that contain HTML-encoded tags or PHP error output."""
    return bool(_GARBAGE_URL_RE.search(url))


class CompanyEmailSpider(scrapy.Spider):
    name = "company_email_spider"

    custom_settings: dict[str, object] = {
        "DOWNLOAD_HANDLERS": {
            "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
        },
        "TWISTED_REACTOR": "twisted.internet.asyncioreactor.AsyncioSelectorReactor",
        "PLAYWRIGHT_BROWSER_TYPE": "chromium",
        "PLAYWRIGHT_LAUNCH_OPTIONS": {"headless": True, "timeout": 30_000},
        "PLAYWRIGHT_DEFAULT_NAVIGATION_TIMEOUT": 30_000,
        "PLAYWRIGHT_MAX_CONTEXTS": 1,
        "PLAYWRIGHT_MAX_PAGES_PER_CONTEXT": 4,
        "CONCURRENT_REQUESTS": 4,
        "CONCURRENT_REQUESTS_PER_DOMAIN": 2,
        "RETRY_TIMES": 0,
        "DOWNLOAD_TIMEOUT": 30,
        "AUTOTHROTTLE_ENABLED": True,
        "AUTOTHROTTLE_START_DELAY": 0.5,
        "AUTOTHROTTLE_MAX_DELAY": 8.0,
        "AUTOTHROTTLE_TARGET_CONCURRENCY": 2.0,
        "COOKIES_ENABLED": False,
        "HTTPPROXY_ENABLED": False,
        "TELNETCONSOLE_ENABLED": False,
        # FIX: handle 404s gracefully — don't treat them as fatal errors,
        # just skip the page. Without this, HttpErrorMiddleware drops 4xx
        # responses before our callback ever sees them.
        "HTTPERROR_ALLOWED_CODES": [404],
        "USER_AGENT": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
    }

    def __init__(
        self,
        company_name: str,
        domain: str,
        official_url: str,
        collected_emails: set[str] | None = None,
        max_pages: int = 50,
        depth_limit: int = 3,       # FIX: raised from 2 → 3 so crawl goes deeper
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.company_name = company_name
        self.domain = domain.lower()
        self.official_url = _normalize_url(official_url)
        self.allowed_domains = [self.domain, f"www.{self.domain}"]
        self.collected_emails: set[str] = (
            collected_emails if collected_emails is not None else set()
        )
        self.max_pages = max_pages
        self.depth_limit = depth_limit
        self.link_extractor = LinkExtractor(allow_domains=self.allowed_domains, unique=True)
        self.visited_urls: set[str] = set()
        self._homepage_done: bool = False

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def _seed_requests(self) -> list[scrapy.Request]:
        split = urlsplit(self.official_url)
        http_url = _normalize_url(
            urlunsplit(("http", split.netloc, split.path, "", ""))
        )
        return [
            self._build_request(
                http_url,
                priority=100,
                use_playwright=False,
                callback=self._homepage_parsed,
                dont_filter=True,
            )
        ]

    def start_requests(self):
        yield from self._seed_requests()

    async def start(self):
        for request in self._seed_requests():
            yield request

    # ------------------------------------------------------------------
    # Homepage callback
    # ------------------------------------------------------------------

    def _homepage_parsed(self, response: scrapy.http.Response, **kwargs):
        canonical = _normalize_url(response.url)

        if self._homepage_done:
            LOGGER.debug("Homepage already processed, skipping: %s", canonical)
            return
        self._homepage_done = True

        self.visited_urls.add(canonical)
        LOGGER.info("Homepage canonical URL: %s", canonical)

        self._extract_emails(response)

        for path in PRIORITY_PATHS:
            url = _normalize_url(urljoin(canonical, path))
            if url not in self.visited_urls:
                yield self._build_request(
                    url,
                    priority=90,
                    use_playwright=False,
                    dont_filter=True,
                )

        yield from self._follow_links(response)

    # ------------------------------------------------------------------
    # Main parse callback
    # ------------------------------------------------------------------

    def parse(self, response: scrapy.http.Response, **kwargs):
        # FIX: silently skip 404s from priority path probing — no retry needed.
        if response.status == 404:
            LOGGER.debug("404 skipped: %s", response.url)
            return

        normalized = _normalize_url(response.url)
        if normalized in self.visited_urls:
            return
        self.visited_urls.add(normalized)
        LOGGER.info("Parsed [%s]: %s", response.status, normalized)

        self._extract_emails(response)
        if len(self.collected_emails) >= 5:
            self.logger.info("Enough emails found, stopping crawl")
            return

        if response.meta.get("depth", 0) >= self.depth_limit:
            return

        if len(self.visited_urls) >= self.max_pages:
            return

        yield from self._follow_links(response)

    def _follow_links(self, response: scrapy.http.Response):
        candidate_links = sorted(
            {
                normalized
                for link in self.link_extractor.extract_links(response)
                # FIX: filter out garbage URLs before they enter the queue.
                if not _is_garbage_url(link.url)
                if (normalized := _normalize_url(link.url)) not in self.visited_urls
                and self._is_internal_url(normalized)
            },
            key=self._link_priority,
            reverse=True,
        )
        for url in candidate_links:
            yield self._build_request(
                url,
                priority=self._link_priority(url),
                use_playwright=False,
                dont_filter=False,
            )

    # ------------------------------------------------------------------
    # Request building
    # ------------------------------------------------------------------

    def _build_request(
        self,
        url: str,
        priority: int,
        use_playwright: bool = False,
        fallback_playwright: bool = False,
        callback=None,
        dont_filter: bool = False,
    ) -> scrapy.Request:
        # FIX: reject garbage URLs before they ever enter the Scrapy queue.
        if _is_garbage_url(url):
            LOGGER.debug("Rejected garbage URL: %s", url[:120])
            # Return a no-op request that will be filtered — we can't return
            # None here, so we redirect to a known-good URL and mark visited.
            # Better: just raise so the caller skips this gracefully.
            raise ValueError(f"Garbage URL rejected: {url[:120]}")

        meta: dict[str, object] = {
            "dont_proxy": True,
            "download_timeout": 30,
            "fallback_playwright": fallback_playwright,
        }
        if use_playwright:
            meta |= {
                "playwright": True,
                "playwright_page_methods": _PLAYWRIGHT_PAGE_METHODS,
                "playwright_context": "default",
                "playwright_context_kwargs": {"ignore_https_errors": True},
            }

        return scrapy.Request(
            url=url,
            callback=callback or self.parse,
            priority=priority,
            dont_filter=dont_filter,
            meta=meta,
            errback=self._handle_error,
        )

    def _handle_error(self, failure):
        request = failure.request
        error_name = failure.value.__class__.__name__

        # FIX: don't retry HttpError (4xx/5xx) — those pages don't exist,
        # retrying with Playwright won't help.
        if error_name == "HttpError":
            LOGGER.debug("HTTP error skipped (no retry): %s", request.url[:80])
            return

        LOGGER.warning("Request failed [%s]: %s", error_name, request.url[:80])

        if not request.meta.get("fallback_playwright"):
            LOGGER.info("Upgrading to Playwright: %s", request.url[:80])
            yield self._build_request(
                request.url,
                priority=request.priority,
                use_playwright=True,
                fallback_playwright=True,
                callback=request.callback,
                dont_filter=True,
            )
        else:
            LOGGER.warning("Both plain and Playwright failed: %s", request.url[:80])

    # ------------------------------------------------------------------
    # URL helpers
    # ------------------------------------------------------------------

    def _is_internal_url(self, url: str) -> bool:
        hostname = (urlsplit(url).hostname or "").lower()
        return hostname == self.domain or hostname.endswith(f".{self.domain}")

    def _link_priority(self, url: str) -> int:
        path = (urlsplit(url).path or "/").lower()
        score = 0
        if "contact" in path:
            score += 100
        elif "about" in path:
            score += 80
        elif "privacy" in path or "terms" in path:
            score += 70
        elif "legal" in path:
            score += 60
        return score

    # ------------------------------------------------------------------
    # Email extraction
    # ------------------------------------------------------------------

    def _extract_emails(self, response: scrapy.http.Response) -> None:
        html = response.text or ""

        for raw_email in EMAIL_REGEX.findall(html):
            self._add_email(raw_email)

        for href in response.css("a::attr(href)").getall():
            if href.lower().startswith("mailto:"):
                self._add_email(href.split("mailto:", 1)[1].split("?")[0])

        deobfuscated = (
            html.lower()
            .replace("[at]", "@")
            .replace("(at)", "@")
            .replace("[dot]", ".")
            .replace("(dot)", ".")
        )
        for raw_email in EMAIL_REGEX.findall(deobfuscated):
            self._add_email(raw_email)

    def _add_email(self, email: str) -> None:
        normalized = email.strip().strip(".,;:()[]{}<>\"'").lower()
        if not normalized or normalized.count("@") != 1:
            return

        local_part, domain_part = normalized.split("@", 1)

        if not domain_part.endswith(self.domain):
            return

        if local_part.startswith(REJECTED_LOCAL_PART_PREFIXES):
            return

        self.collected_emails.add(normalized)


# ---------------------------------------------------------------------------
# Public crawl API
# ---------------------------------------------------------------------------

def crawl_company_site(
    company_name: str,
    domain: str,
    official_url: str,
    max_pages: int = 50,
    depth_limit: int = 3,
) -> list[str]:
    """Run the Scrapy spider and return filtered email addresses."""
    collected_emails: set[str] = set()

    settings = get_project_settings()
    settings.setdict(
        {
            "CLOSESPIDER_PAGECOUNT": max_pages,
            "DEPTH_LIMIT": depth_limit,
            "LOG_ENABLED": True,
            "LOG_LEVEL": "INFO",
        },
        priority="cmdline",
    )

    LOGGER.info("Crawling %s (depth=%s, max_pages=%s)", domain, depth_limit, max_pages)

    process = CrawlerProcess(settings=settings)
    process.crawl(
        CompanyEmailSpider,
        company_name=company_name,
        domain=domain,
        official_url=official_url,
        collected_emails=collected_emails,
        max_pages=max_pages,
        depth_limit=depth_limit,
    )
    process.start(stop_after_crawl=True)

    emails = sorted(collected_emails)
    LOGGER.info("Found %s company-domain email(s)", len(emails))
    return emails


def _crawl_worker(
    company_name: str,
    domain: str,
    official_url: str,
    max_pages: int,
    depth_limit: int,
    result_queue,
) -> None:
    try:
        emails = crawl_company_site(
            company_name=company_name,
            domain=domain,
            official_url=official_url,
            max_pages=max_pages,
            depth_limit=depth_limit,
        )
        result_queue.put({"emails": emails})
    except Exception as exc:  # pragma: no cover – child-process guard
        result_queue.put({"error": str(exc)})


def crawl_company_site_isolated(
    company_name: str,
    domain: str,
    official_url: str,
    max_pages: int = 50,
    depth_limit: int = 3,
    timeout_seconds: int = 180,
) -> list[str]:
    """
    Run crawl in a separate process to avoid Twisted reactor restart issues
    across repeated API requests.
    """
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(
        target=_crawl_worker,
        args=(company_name, domain, official_url, max_pages, depth_limit, result_queue),
        daemon=True,
    )
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError(
            f"Crawl timed out after {timeout_seconds} seconds for domain '{domain}'."
        )

    try:
        payload = result_queue.get_nowait()
    except Empty as exc:
        raise RuntimeError(
            f"Crawl finished without results for domain '{domain}'."
        ) from exc

    if isinstance(payload, dict) and (error := payload.get("error")):
        raise RuntimeError(error)

    return sorted(set(payload.get("emails", []) if isinstance(payload, dict) else []))