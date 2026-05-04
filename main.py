from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool

from company_search import CompanySearchError, search_companies_from_query
from domain_finder import DomainFinderError, find_official_domain
from spider import crawl_company_site_isolated


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def build_result(company_name: str, max_pages: int, depth_limit: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "company": company_name,
        "domain": "",
        "emails": [],
    }

    try:
        domain_match = find_official_domain(company_name)
    except DomainFinderError as exc:
        LOGGER.error("Domain lookup failed: %s", exc)
        return result

    if not domain_match:
        return result

    result["domain"] = domain_match.domain

    try:
        result["emails"] = crawl_company_site_isolated(
            company_name=company_name,
            domain=domain_match.domain,
            official_url=domain_match.official_url,
            max_pages=max_pages,
            depth_limit=depth_limit,
        )
    except TimeoutError as exc:
        LOGGER.warning("Crawler timeout for %s: %s", company_name, exc)
        result["emails"] = []
    except Exception as exc:  # pragma: no cover - runtime safety net
        LOGGER.exception("Crawler failed for %s: %s", company_name, exc)
        result["emails"] = []

    return result


class EmailLookupRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    company: str = Field(..., min_length=1, description="Company name, e.g., Stripe")
    max_pages: int = Field(50, ge=1, le=200, description="Maximum pages to crawl")
    depth: int = Field(2, ge=0, le=5, description="Maximum internal crawl depth")


class EmailLookupResponse(BaseModel):
    company: str
    domain: str
    emails: list[str]


class CompanyQueryRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(..., description="Search query, e.g. remote SaaS companies")
    max_results: int = Field(100, ge=50, le=200, description="Maximum companies in the response")
    max_search_results: int = Field(
        40,
        ge=10,
        le=200,
        description="Number of SerpAPI organic results to inspect (supports pagination)",
    )


class CompanyQueryResponse(BaseModel):
    query: str
    companies: list[str]


PROJECT_ROOT = Path(__file__).resolve().parent
DOTENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(dotenv_path=DOTENV_PATH)
configure_logging()

app = FastAPI(
    title="Company Official Website Email Extractor",
    description=(
        "Finds emails published on the official company website only. "
        "No guessing and no third-party source scraping."
    ),
    version="1.0.0",
)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "serpapi_configured": bool(os.getenv("SERPAPI_API_KEY")),
        "dotenv_path": str(DOTENV_PATH),
    }


@app.post("/extract-emails", response_model=EmailLookupResponse)
async def extract_emails(payload: EmailLookupRequest) -> EmailLookupResponse:
    company_name = payload.company.strip()
    if not company_name:
        raise HTTPException(status_code=400, detail="Company name cannot be empty.")

    LOGGER.info(
        "Email extraction requested: company='%s', max_pages=%s, depth=%s",
        company_name,
        payload.max_pages,
        payload.depth,
    )

    result = await run_in_threadpool(
        build_result,
        company_name,
        payload.max_pages,
        payload.depth,
    )
    return EmailLookupResponse(**result)


@app.post("/search-companies", response_model=CompanyQueryResponse)
async def search_companies(payload: CompanyQueryRequest) -> CompanyQueryResponse:
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    LOGGER.info(
        "Company search requested: query='%s', max_results=%s, max_search_results=%s",
        query,
        payload.max_results,
        payload.max_search_results,
    )

    try:
        companies = await run_in_threadpool(
            search_companies_from_query,
            query,
            payload.max_results,
            payload.max_search_results,
        )
    except CompanySearchError as exc:
        LOGGER.error("Company search failed: %s", exc)
        companies = []

    return CompanyQueryResponse(query=query, companies=companies)
