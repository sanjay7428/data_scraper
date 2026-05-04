from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from urllib.parse import unquote, urlparse

import requests
from bs4 import BeautifulSoup

from domain_finder import BLOCKED_ROOT_DOMAINS, SERPAPI_ENDPOINT, clean_root_domain


LOGGER = logging.getLogger(__name__)

SEARCH_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
COMPANY_CLEANUP_SPACE_PATTERN = re.compile(r"\s+")
LEADING_INDEX_PATTERN = re.compile(r"^\s*(?:#?\d+[\).:-]?\s+)+")
SPLIT_ON_TRAILING_DETAILS_PATTERN = re.compile(r"\s(?:\||:|-)\s")
VALID_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9&'().,+/ \-]{1,80}$")
LINKEDIN_COMPANY_PATH_PATTERN = re.compile(r"^/company/([^/?#]+)/?", re.IGNORECASE)

STOPWORDS = {
    "and",
    "or",
    "the",
    "a",
    "an",
    "using",
    "with",
    "company",
    "companies",
}
LIST_PAGE_HINTS = {
    "remote companies",
    "fully remote",
    "remote startups",
    "best remote",
    "top remote",
    "list",
    "companies hiring",
}
DISALLOWED_ROOT_DOMAINS = (set(BLOCKED_ROOT_DOMAINS) - {"linkedin.com"}) | {
    "crunchbase.com",
    "angel.co",
    "wellfound.com",
    "glassdoor.com",
    "flexjobs.com",
    "remote.co",
    "weworkremotely.com",
    "workingnomads.com",
    "authenticjobs.com",
    "nodesk.co",
}
SKIP_COMPANY_TERMS = {
    "about",
    "all articles",
    "app integrations",
    "blog",
    "company news",
    "documentation",
    "fulfillment policy",
    "functions beta",
    "guides",
    "help center",
    "home",
    "integration partner program",
    "open blog",
    "platform tips",
    "powered by zapier",
    "pricing",
    "product news",
    "productivity tips",
    "read more",
    "remote work academy",
    "software",
    "startups",
    "tables",
    "web scraping",
    "webhooks and zapier",
    "webinars",
    "learn more",
    "view more",
    "see more",
    "source",
    "jobs",
    "job",
    "here",
    "get directions",
    "commits",
    "remote work",
    "remote only",
    "menu",
    "login",
    "sign in",
    "sign up",
}
NOISE_PHRASES = {
    "interview",
    "salary calculator",
    "salary formula",
    "staff locations",
    "work from anywhere",
    "productivity tools",
    "business intelligence",
    "presentation software",
    "creator tools",
    "meeting assistant",
    "time tracking",
    "travel accessories",
    "video communication",
    "consumer internet",
    "freelance marketplace",
}
NOISE_TOKENS = {
    "across",
    "answer",
    "answers",
    "app",
    "apps",
    "assistants",
    "automation",
    "automations",
    "better",
    "campaign",
    "capture",
    "cases",
    "chatbots",
    "close",
    "customer",
    "decision-making",
    "drive",
    "effectiveness",
    "eliminate",
    "elevate",
    "enterprise",
    "explore",
    "flows",
    "forms",
    "grade",
    "hiring",
    "integrations",
    "join",
    "leaders",
    "manage",
    "marketing",
    "menu",
    "multiply",
    "no-code",
    "questions",
    "repetitive",
    "revenue",
    "roi",
    "sales",
    "security",
    "streamline",
    "support",
    "systems",
    "templates",
    "trigger",
    "use",
    "view",
    "workflow",
    "workflows",
}
ALLOWED_LOWER_CONNECTORS = {"and", "&", "of", "the", "for", "at", "to", "io", "ai"}
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


class CompanySearchError(RuntimeError):
    """Raised when remote company extraction through SerpAPI/scraping fails."""


@dataclass(frozen=True, slots=True)
class SearchResult:
    title: str
    link: str
    snippet: str
    score: int


def _normalize_query_keywords(query: str) -> list[str]:
    tokens = SEARCH_TOKEN_PATTERN.findall(query.lower())
    return [token for token in tokens if len(token) >= 3 and token not in STOPWORDS]


def _is_linkedin_company_query(query: str) -> bool:
    lowered = query.lower()
    return "site:linkedin.com/company" in lowered


def _build_query_variants(query: str) -> list[str]:
    lowered = query.lower().strip()
    variants: list[str] = [query.strip()]

    if _is_linkedin_company_query(query):
        base = lowered.replace("site:linkedin.com/company", "").strip()
        core = base if base else "saas company"
        variants.extend(
            [
                f"{core} site:linkedin.com/company",
                f"b2b {core} site:linkedin.com/company",
                f"cloud software {core} site:linkedin.com/company",
                f"remote software company site:linkedin.com/company",
                f"saas startup site:linkedin.com/company",
            ]
        )
    else:
        variants.extend(
            [
                f"{query} top list",
                f"{query} best companies",
                f"{query} fully remote",
            ]
        )

    deduped: list[str] = []
    seen: set[str] = set()
    for item in variants:
        normalized = item.strip()
        key = normalized.lower()
        if not normalized or key in seen:
            continue
        seen.add(key)
        deduped.append(normalized)
    return deduped


def _root_domain_from_url(url: str) -> str | None:
    parsed = urlparse(url)
    if not parsed.hostname:
        return None
    return clean_root_domain(parsed.hostname)


def _is_disallowed_source(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return True
    root = clean_root_domain(parsed.hostname)
    if root in DISALLOWED_ROOT_DOMAINS:
        return True

    lowered_path = (parsed.path or "").lower()
    if any(token in lowered_path for token in ("/login", "/signin", "/auth")):
        return True
    if lowered_path.endswith((".pdf", ".doc", ".docx")):
        return True
    return False


def _source_score(title: str, snippet: str, link: str) -> int:
    text = f"{title} {snippet} {link}".lower()
    score = 0
    for hint in LIST_PAGE_HINTS:
        if hint in text:
            score += 2
    if "/list" in text or "/top" in text or "/companies" in text:
        score += 2
    if "remote" in text:
        score += 1
    return score


def _keyword_score(title: str, snippet: str, keywords: list[str]) -> int:
    searchable = f"{title} {snippet}".lower()
    return sum(1 for keyword in keywords if keyword in searchable)


def _fetch_serp_results(
    query: str,
    api_key: str,
    max_search_results: int,
    keywords: list[str],
    session: requests.Session,
    min_score: int,
) -> list[SearchResult]:
    results: list[SearchResult] = []
    seen_links: set[str] = set()
    normalized_max_search = max(10, min(max_search_results, 200))
    page_size = 20

    for start in range(0, normalized_max_search, page_size):
        num = min(page_size, normalized_max_search - start)
        params: dict[str, object] = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "num": num,
            "start": start,
            "hl": "en",
            "gl": "us",
        }
        try:
            response = session.get(SERPAPI_ENDPOINT, params=params, timeout=20)
            response.raise_for_status()
            payload: dict[str, object] = response.json()
        except requests.RequestException as exc:
            raise CompanySearchError("SerpAPI request failed.") from exc
        except ValueError as exc:
            raise CompanySearchError("Invalid JSON from SerpAPI.") from exc

        if error := payload.get("error"):
            raise CompanySearchError(str(error))

        organic = payload.get("organic_results", [])
        if not isinstance(organic, list) or not organic:
            break

        accepted_this_page = 0
        for item in organic:
            if not isinstance(item, dict):
                continue
            link = item.get("link")
            title = item.get("title")
            snippet = item.get("snippet")
            if not isinstance(link, str) or not link:
                continue
            if link in seen_links:
                continue
            if _is_disallowed_source(link):
                continue

            safe_title = title if isinstance(title, str) else ""
            safe_snippet = snippet if isinstance(snippet, str) else ""
            score = _source_score(safe_title, safe_snippet, link)
            score += _keyword_score(safe_title, safe_snippet, keywords)
            if score < min_score:
                continue

            seen_links.add(link)
            accepted_this_page += 1
            results.append(
                SearchResult(
                    title=safe_title,
                    link=link,
                    snippet=safe_snippet,
                    score=score,
                )
            )

        # Stop early when pagination returns no usable records.
        if accepted_this_page == 0 and start > 0:
            break

    results.sort(key=lambda item: item.score, reverse=True)
    return results


def _fetch_html(url: str, session: requests.Session) -> str | None:
    headers = {"User-Agent": USER_AGENT}
    try:
        response = session.get(url, timeout=20, headers=headers, allow_redirects=True)
    except requests.RequestException:
        return None

    if response.status_code >= 400:
        return None
    content_type = (response.headers.get("Content-Type") or "").lower()
    if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
        return None
    if not response.text:
        return None
    return response.text


def _normalized_company_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def _looks_like_company_name(name: str) -> bool:
    if not name:
        return False
    lowered = name.lower().strip()
    if lowered in SKIP_COMPANY_TERMS:
        return False
    if any(phrase in lowered for phrase in NOISE_PHRASES):
        return False
    if lowered.endswith((" policy", " center", " academy", " documentation")):
        return False
    if "http" in lowered or "@" in lowered:
        return False
    words = [w for w in lowered.split() if w]
    if len(words) > 5:
        return False
    if not VALID_NAME_PATTERN.match(name):
        return False
    if not any(ch.isalpha() for ch in name):
        return False
    if len(name) > 50:
        return False
    if len(name) <= 2:
        return False

    normalized_tokens = [re.sub(r"[^a-z0-9&+-]", "", token) for token in words]
    if any(token in NOISE_TOKENS for token in normalized_tokens if token):
        return False

    # Keep brand-like strings; reject sentence-like lowercase phrasing.
    original_tokens = [token.strip() for token in name.split() if token.strip()]
    lowercase_non_connectors = 0
    for token in original_tokens:
        pure = re.sub(r"[^A-Za-z0-9&+-]", "", token)
        if not pure:
            continue
        if pure.islower() and pure.lower() not in ALLOWED_LOWER_CONNECTORS:
            lowercase_non_connectors += 1
    if lowercase_non_connectors >= 2:
        return False

    # Most valid company names arrive title-cased in curated lists.
    if not any(ch.isupper() for ch in name):
        return False

    return True


def _is_in_excluded_section(tag) -> bool:
    for parent in getattr(tag, "parents", []):
        if not getattr(parent, "name", None):
            continue
        parent_name = str(parent.name).lower()
        if parent_name in {"nav", "header", "footer", "aside", "form"}:
            return True

        role = parent.attrs.get("role") if hasattr(parent, "attrs") else None
        if isinstance(role, str) and role.lower() in {"navigation", "menu", "contentinfo"}:
            return True

        class_names = " ".join(parent.attrs.get("class", [])) if hasattr(parent, "attrs") else ""
        parent_id = parent.attrs.get("id", "") if hasattr(parent, "attrs") else ""
        container_text = f"{class_names} {parent_id}".lower()
        if any(
            marker in container_text
            for marker in ("nav", "menu", "header", "footer", "sidebar", "breadcrumb")
        ):
            return True
    return False


def _clean_company_candidate(text: str) -> str:
    cleaned = COMPANY_CLEANUP_SPACE_PATTERN.sub(" ", text or "").strip()
    cleaned = LEADING_INDEX_PATTERN.sub("", cleaned)
    cleaned = SPLIT_ON_TRAILING_DETAILS_PATTERN.split(cleaned, maxsplit=1)[0]
    cleaned = cleaned.strip(" \t\r\n\"'`()[]{}|:;,.")
    return cleaned


def _slug_to_company_name(slug: str) -> str:
    words = [w for w in re.split(r"[-_]+", slug) if w]
    if not words:
        return ""
    normalized_words: list[str] = []
    for word in words:
        if word.isupper():
            normalized_words.append(word)
        elif len(word) <= 3:
            normalized_words.append(word.upper())
        else:
            normalized_words.append(word.capitalize())
    return " ".join(normalized_words)


def _extract_linkedin_company_name(link: str, title: str) -> str | None:
    parsed = urlparse(link)
    if clean_root_domain(parsed.hostname or "") != "linkedin.com":
        return None

    # Prefer explicit company slug from URL.
    match = LINKEDIN_COMPANY_PATH_PATTERN.match(parsed.path or "")
    if match:
        slug = unquote(match.group(1)).strip().strip("/")
        if slug and slug not in {"jobs", "posts", "about"}:
            candidate = _clean_company_candidate(_slug_to_company_name(slug))
            if _looks_like_company_name(candidate):
                return candidate

    # Fallback to result title like "Acme | LinkedIn".
    if title:
        candidate = _clean_company_candidate(title.split("|", 1)[0])
        if _looks_like_company_name(candidate):
            return candidate
    return None


def _extract_companies_from_page(html: str, source_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()

    extracted: list[str] = []
    seen_keys: set[str] = set()
    source_domain = _root_domain_from_url(source_url)

    # Pass 1: external links from content/list sections only.
    for anchor in soup.select("main a[href], article a[href], li a[href], td a[href]"):
        if _is_in_excluded_section(anchor):
            continue
        href = anchor.get("href")
        if not isinstance(href, str) or not href:
            continue
        if href.startswith(("/", "#", "mailto:", "tel:")):
            continue
        if _is_disallowed_source(href):
            continue

        target_domain = _root_domain_from_url(href)
        if not target_domain or target_domain == source_domain:
            continue

        anchor_text = _clean_company_candidate(anchor.get_text(" ", strip=True))
        if _looks_like_company_name(anchor_text):
            key = _normalized_company_key(anchor_text)
            if key and key not in seen_keys:
                seen_keys.add(key)
                extracted.append(anchor_text)

    return extracted


def search_companies_from_query(
    query: str,
    max_results: int = 100,
    max_search_results: int = 40,
) -> list[str]:
    """Extract remote company names from real webpage content discovered via SerpAPI."""
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        raise CompanySearchError("SERPAPI_API_KEY is not set.")

    normalized_query = query.strip()
    if not normalized_query:
        return []

    normalized_max = max(50, min(max_results, 200))
    keywords = _normalize_query_keywords(normalized_query)
    if not keywords:
        return []
    query_variants = _build_query_variants(normalized_query)
    linkedin_mode = _is_linkedin_company_query(normalized_query)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    LOGGER.info(
        "Searching source pages for remote companies: query='%s', max_results=%s, max_search_results=%s, variants=%s",
        normalized_query,
        normalized_max,
        max_search_results,
        len(query_variants),
    )

    companies: list[str] = []
    seen_company_keys: set[str] = set()
    scraped_pages = 0
    remaining_budget = max(10, min(max_search_results, 200))

    for index, variant in enumerate(query_variants):
        if len(companies) >= normalized_max:
            break
        if remaining_budget <= 0:
            break

        variants_left = len(query_variants) - index
        budget_for_variant = max(10, remaining_budget // max(1, variants_left))
        min_score = 1 if linkedin_mode else 3

        search_results = _fetch_serp_results(
            query=variant,
            api_key=api_key,
            max_search_results=budget_for_variant,
            keywords=keywords,
            session=session,
            min_score=min_score,
        )
        remaining_budget -= budget_for_variant

        for result in search_results:
            if scraped_pages >= max_search_results:
                break
            if result.score <= 0:
                continue

            linkedin_name = _extract_linkedin_company_name(result.link, result.title)
            if linkedin_name:
                key = _normalized_company_key(linkedin_name)
                if key and key not in seen_company_keys:
                    seen_company_keys.add(key)
                    companies.append(linkedin_name)
                    if len(companies) >= normalized_max:
                        return companies
                continue

            html = _fetch_html(result.link, session=session)
            if not html:
                continue
            scraped_pages += 1

            page_companies = _extract_companies_from_page(html, result.link)
            for company_name in page_companies:
                key = _normalized_company_key(company_name)
                if not key or key in seen_company_keys:
                    continue
                seen_company_keys.add(key)
                companies.append(company_name)
                if len(companies) >= normalized_max:
                    return companies

    return companies
