from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from urllib.parse import urlparse

import requests


LOGGER = logging.getLogger(__name__)

SERPAPI_ENDPOINT = "https://serpapi.com/search.json"
BLOCKED_ROOT_DOMAINS: frozenset[str] = frozenset({
    "google.com",
    "linkedin.com",
    "wikipedia.org",
    "facebook.com",
    "twitter.com",
    "x.com",
    "instagram.com",
    "youtube.com",
    "crunchbase.com",
    "bloomberg.com",
})
COMMON_SECOND_LEVEL_TLDS: frozenset[str] = frozenset({
    "ac.uk",
    "co.in",
    "co.jp",
    "co.kr",
    "co.nz",
    "co.uk",
    "com.au",
    "com.br",
    "com.cn",
    "com.mx",
    "com.tr",
    "gov.uk",
    "net.au",
    "org.au",
})


class DomainFinderError(RuntimeError):
    """Raised when the official domain cannot be fetched from SerpAPI."""


@dataclass(frozen=True, slots=True)
class DomainMatch:
    domain: str
    official_url: str


def _normalize_hostname(hostname: str) -> str:
    return hostname.strip().lower().rstrip(".")


def clean_root_domain(hostname: str) -> str:
    hostname = _normalize_hostname(hostname)
    hostname = hostname.removeprefix("www.")

    parts = hostname.split(".")
    if len(parts) <= 2:
        return hostname

    suffix = ".".join(parts[-2:])
    if suffix in COMMON_SECOND_LEVEL_TLDS and len(parts) >= 3:
        return ".".join(parts[-3:])

    return ".".join(parts[-2:])


def _extract_candidate_url(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    return f"{parsed.scheme}://{clean_root_domain(parsed.hostname)}/"


def _is_blocked_domain(hostname: str) -> bool:
    return clean_root_domain(hostname) in BLOCKED_ROOT_DOMAINS


def _iter_result_links(payload: dict[str, object]) -> list[str]:
    return [
        link
        for result in payload.get("organic_results", [])
        if isinstance(result, dict)
        if isinstance(link := result.get("link"), str) and link
    ]


def _score_candidate(domain: str, company_keyword: str) -> int:
    score = 0
    if company_keyword:
        if company_keyword in domain:
            score += 2
        if domain.startswith(company_keyword):
            score += 1
    if len(domain) < 25:
        score += 1
    return score


def find_official_domain(
    company_name: str,
    api_key: str | None = None,
    session: requests.Session | None = None,
    timeout_seconds: int = 15,
) -> DomainMatch | None:
    """Find the best non-blocked official domain from a SerpAPI Google query."""
    resolved_api_key = api_key or os.getenv("SERPAPI_API_KEY")
    if not resolved_api_key:
        raise DomainFinderError("SERPAPI_API_KEY is not set.")

    query = f"{company_name} official website"
    params: dict[str, object] = {
        "engine": "google",
        "q": query,
        "api_key": resolved_api_key,
        "num": 10,
        "hl": "en",
        "gl": "us",
    }

    http = session or requests.Session()
    LOGGER.info("Searching SerpAPI for official website: %s", query)

    try:
        response = http.get(SERPAPI_ENDPOINT, params=params, timeout=timeout_seconds)
        response.raise_for_status()
        payload: dict[str, object] = response.json()
    except requests.RequestException as exc:
        raise DomainFinderError(f"SerpAPI request failed: {exc}") from exc
    except ValueError as exc:
        raise DomainFinderError("Invalid JSON from SerpAPI.") from exc

    if error := payload.get("error"):
        raise DomainFinderError(str(error))

    company_keyword = company_name.lower().split()[0] if company_name.strip() else ""
    candidates: list[tuple[int, str, str]] = []

    for link in _iter_result_links(payload):
        parsed = urlparse(link)
        if not (hostname := parsed.hostname):
            continue

        normalized = _normalize_hostname(hostname)
        if _is_blocked_domain(normalized):
            LOGGER.debug("Skipping blocked domain result: %s", normalized)
            continue

        if not (candidate_url := _extract_candidate_url(link)):
            continue

        domain = clean_root_domain(normalized)
        candidates.append((_score_candidate(domain, company_keyword), domain, candidate_url))

    if not candidates:
        LOGGER.warning("No valid official domain found for company: %s", company_name)
        return None

    best_score, best_domain, best_url = max(candidates, key=lambda item: item[0])
    LOGGER.info(
        "Selected official domain: %s (score=%s)",
        best_domain,
        best_score,
    )
    return DomainMatch(domain=best_domain, official_url=best_url)