from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from domain_finder import BLOCKED_ROOT_DOMAINS, SERPAPI_ENDPOINT, clean_root_domain


LOGGER = logging.getLogger(__name__)

SEARCH_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
COMPANY_CLEANUP_SPACE_PATTERN = re.compile(r"\s+")
LEADING_INDEX_PATTERN = re.compile(r"^\s*(?:#?\d+[\).:-]?\s+)+")
SPLIT_ON_TRAILING_DETAILS_PATTERN = re.compile(r"\s(?:\||:|-)\s")
VALID_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9&'().,+/ \-]{1,80}$")

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
DISALLOWED_ROOT_DOMAINS = set(BLOCKED_ROOT_DOMAINS) | {
    "linkedin.com",
    "crunchbase.com",
    "angel.co",
    "wellfound.com",
}
SKIP_COMPANY_TERMS = {
    "read more",
    "learn more",
    "view more",
    "see more",
    "source",
    "menu",
    "login",
    "sign in",
    "sign up",
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
) -> list[SearchResult]:
    params: dict[str, object] = {
        "engine": "google",
        "q": query,
        "api_key": api_key,
        "num": max(10, min(max_search_results, 20)),
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

    results: list[SearchResult] = []
    for item in payload.get("organic_results", []):
        if not isinstance(item, dict):
            continue
        link = item.get("link")
        title = item.get("title")
        snippet = item.get("snippet")
        if not isinstance(link, str) or not link:
            continue
        if _is_disallowed_source(link):
            continue

        safe_title = title if isinstance(title, str) else ""
        safe_snippet = snippet if isinstance(snippet, str) else ""
        score = _source_score(safe_title, safe_snippet, link)
        score += _keyword_score(safe_title, safe_snippet, keywords)
        if score <= 0:
            continue
        results.append(
            SearchResult(
                title=safe_title,
                link=link,
                snippet=safe_snippet,
                score=score,
            )
        )

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

    return True


def _clean_company_candidate(text: str) -> str:
    cleaned = COMPANY_CLEANUP_SPACE_PATTERN.sub(" ", text or "").strip()
    cleaned = LEADING_INDEX_PATTERN.sub("", cleaned)
    cleaned = SPLIT_ON_TRAILING_DETAILS_PATTERN.split(cleaned, maxsplit=1)[0]
    cleaned = cleaned.strip(" \t\r\n\"'`()[]{}|:;,.")
    return cleaned


def _extract_companies_from_page(html: str, source_url: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()

    extracted: list[str] = []
    seen_keys: set[str] = set()
    source_domain = _root_domain_from_url(source_url)

    # Pass 1: external links from list pages are usually the highest-quality company signals.
    for anchor in soup.select("a[href]"):
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

    # Pass 2: structured list/table items for pages that don't link every company externally.
    selector = "li, td"
    for node in soup.select(selector):
        text = node.get_text(" ", strip=True)
        candidate = _clean_company_candidate(text)
        if _looks_like_company_name(candidate):
            key = _normalized_company_key(candidate)
            if key and key not in seen_keys:
                seen_keys.add(key)
                extracted.append(candidate)

        for anchor in node.find_all("a", href=True):
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
    max_search_results: int = 20,
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

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    LOGGER.info(
        "Searching source pages for remote companies: query='%s', max_results=%s",
        normalized_query,
        normalized_max,
    )
    search_results = _fetch_serp_results(
        query=normalized_query,
        api_key=api_key,
        max_search_results=max_search_results,
        keywords=keywords,
        session=session,
    )

    companies: list[str] = []
    seen_company_keys: set[str] = set()
    scraped_pages = 0

    for result in search_results:
        if scraped_pages >= max_search_results:
            break
        if result.score <= 0:
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
