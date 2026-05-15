"""Web search tool integrated with weakagent `BaseTool` / `ToolExecutionResult`."""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, ConfigDict, Field, model_validator
from tenacity import retry, stop_after_attempt, wait_exponential

from weakagent.config.settings import config
from weakagent.tools.base import BaseTool, ToolExecutionResult
from weakagent.tools.search.baidu_search import BaiduSearchEngine
from weakagent.tools.search.base import SearchItem, WebSearchEngine
from weakagent.tools.search.bing_search import BingSearchEngine
from weakagent.tools.search.duckduckgo_engine import DuckDuckGoSearchEngine
from weakagent.tools.search.google_search import GoogleSearchEngine
from weakagent.utils.logger import get_logger

logger = get_logger(__name__)


class SearchResult(BaseModel):
    """Single normalized search hit (after engine-specific parsing)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    position: int = Field(description="Position in search results")
    url: str = Field(description="URL of the search result")
    title: str = Field(default="", description="Title of the search result")
    description: str = Field(
        default="", description="Description or snippet of the search result"
    )
    source: str = Field(description="The search engine that provided this result")
    raw_content: Optional[str] = Field(
        default=None, description="Fetched page text when fetch_content is enabled"
    )

    def __str__(self) -> str:
        return f"{self.title} ({self.url})"


class SearchMetadata(BaseModel):
    """Metadata for one successful search invocation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    total_results: int = Field(description="Total number of results returned")
    language: str = Field(description="Language code used for the search")
    country: str = Field(description="Country code used for the search")


class SearchResponse(BaseModel):
    """Structured payload; rendered to tool output text via `apply_output`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    query: str = Field(description="The search query that was executed")
    results: List[SearchResult] = Field(default_factory=list)
    metadata: Optional[SearchMetadata] = None
    error: Optional[str] = None

    @model_validator(mode="after")
    def apply_output(self) -> SearchResponse:
        """Attach human-readable `output` for `ToolExecutionResult`."""
        if self.error:
            return self

        lines: List[str] = [f"Search results for '{self.query}':"]

        for i, result in enumerate(self.results, 1):
            title = result.title.strip() or "No title"
            lines.append(f"\n{i}. {title}")
            lines.append(f"   URL: {result.url}")
            if result.description.strip():
                lines.append(f"   Description: {result.description}")
            if result.raw_content:
                preview = result.raw_content[:1000].replace("\n", " ").strip()
                if len(result.raw_content) > 1000:
                    preview += "..."
                lines.append(f"   Content: {preview}")

        if self.metadata:
            lines.extend(
                [
                    "\nMetadata:",
                    f"- Total results: {self.metadata.total_results}",
                    f"- Language: {self.metadata.language}",
                    f"- Country: {self.metadata.country}",
                ]
            )

        object.__setattr__(self, "_output_text", "\n".join(lines))
        return self

    @property
    def output_text(self) -> str:
        if self.error:
            return ""
        return getattr(self, "_output_text", "")


class WebContentFetcher:
    """Fetch and lightly clean HTML body text (optional snippet enrichment)."""

    @staticmethod
    async def fetch_content(url: str, timeout: int = 10) -> Optional[str]:
        """
        Args:
            url: Page URL.
            timeout: HTTP timeout seconds.

        Returns:
            Plain text excerpt or None if the fetch/parse fails.
        """
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.0.0 Safari/537.36"
            )
        }

        try:
            response = await asyncio.to_thread(
                lambda: requests.get(url, headers=headers, timeout=timeout)
            )
            if response.status_code != 200:
                logger.warning(
                    "Failed to fetch content from %s: HTTP %s", url, response.status_code
                )
                return None

            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "header", "footer", "nav"]):
                tag.extract()

            text = soup.get_text(separator="\n", strip=True)
            text = " ".join(text.split())
            return text[:10000] if text else None
        except Exception as e:
            logger.warning("Error fetching content from %s: %s", url, e)
            return None


class WebSearch(BaseTool):
    """Search the web using configured engines and fallbacks."""

    name: str = "web_search"
    description: str = """Search the web for real-time information about any topic.
    Returns titles, URLs, and snippets. If the primary engine fails, other engines are tried."""
    parameters: dict = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "(required) The search query to submit to the search engine.",
            },
            "num_results": {
                "type": "integer",
                "description": "(optional) The number of search results to return. Default is 5.",
                "default": 5,
            },
            "lang": {
                "type": "string",
                "description": "(optional) Language code for search results (default from config).",
            },
            "country": {
                "type": "string",
                "description": "(optional) Country code for search results (default from config).",
            },
            "fetch_content": {
                "type": "boolean",
                "description": "(optional) Whether to fetch full content from result pages. Default is false.",
                "default": False,
            },
        },
        "required": ["query"],
    }

    _search_engine: dict[str, WebSearchEngine] = {
        "google": GoogleSearchEngine(),
        "baidu": BaiduSearchEngine(),
        "duckduckgo": DuckDuckGoSearchEngine(),
        "bing": BingSearchEngine(),
    }
    content_fetcher: WebContentFetcher = WebContentFetcher()

    async def execute(
        self,
        query: str,
        num_results: int = 5,
        lang: Optional[str] = None,
        country: Optional[str] = None,
        fetch_content: bool = False,
    ) -> ToolExecutionResult:
        """
        Run web search and return `ToolExecutionResult`.

        Args:
            query: Search query.
            num_results: Max hits to return.
            lang: BCP-47 / engine-specific language hint; overrides config when set.
            country: Region hint; overrides config when set.
            fetch_content: Whether to pull main text from each result URL.

        Returns:
            ToolExecutionResult with formatted `output` and optional structured `data`.
        """
        sc = config.search_config
        retry_delay = sc.retry_delay
        max_retries = sc.max_retries

        if lang is None:
            lang = sc.lang
        if country is None:
            country = sc.country

        search_params: Dict[str, Any] = {"lang": lang, "country": country}

        for retry_count in range(max_retries + 1):
            results = await self._try_all_engines(query, num_results, search_params)

            if results:
                if fetch_content:
                    results = await self._fetch_content_for_results(results)

                payload = SearchResponse(
                    query=query,
                    results=results,
                    metadata=SearchMetadata(
                        total_results=len(results),
                        language=lang,
                        country=country,
                    ),
                )
                return ToolExecutionResult.ok(
                    output=payload.output_text,
                    data={
                        "query": query,
                        "results": [r.model_dump() for r in results],
                        "metadata": payload.metadata.model_dump()
                        if payload.metadata
                        else None,
                    },
                )

            if retry_count < max_retries:
                logger.warning(
                    "All search engines failed. Waiting %s s before retry %s/%s...",
                    retry_delay,
                    retry_count + 1,
                    max_retries,
                )
                await asyncio.sleep(retry_delay)
            else:
                logger.error(
                    "All search engines failed after %s retries.", max_retries
                )

        return self.fail_response(
            "All search engines failed to return results after multiple retries."
        )

    async def _try_all_engines(
        self, query: str, num_results: int, search_params: Dict[str, Any]
    ) -> List[SearchResult]:
        """Try engines in config order until one returns hits."""
        engine_order = self._get_engine_order()
        failed: List[str] = []

        for engine_name in engine_order:
            engine = self._search_engine[engine_name]
            logger.info("Attempting search with %s...", engine_name)
            search_items = await self._perform_search_with_engine(
                engine, query, num_results, search_params
            )

            if not search_items:
                failed.append(engine_name)
                continue

            if failed:
                logger.info(
                    "Search succeeded with %s after trying: %s",
                    engine_name,
                    ", ".join(failed),
                )

            return [
                SearchResult(
                    position=i + 1,
                    url=item.url,
                    title=item.title or f"Result {i + 1}",
                    description=item.description or "",
                    source=engine_name,
                )
                for i, item in enumerate(search_items)
            ]

        if failed:
            logger.error("All search engines failed: %s", ", ".join(failed))
        return []

    async def _fetch_content_for_results(
        self, results: List[SearchResult]
    ) -> List[SearchResult]:
        """Fetch main text for each result in parallel."""
        if not results:
            return []

        fetched = await asyncio.gather(
            *[self._fetch_single_result_content(r) for r in results]
        )
        return list(fetched)

    async def _fetch_single_result_content(self, result: SearchResult) -> SearchResult:
        if result.url:
            content = await self.content_fetcher.fetch_content(result.url)
            if content:
                result.raw_content = content
        return result

    def _get_engine_order(self) -> List[str]:
        sc = config.search_config
        preferred = sc.engine.lower()
        fallbacks = [e.lower() for e in sc.fallback_engines]

        order: List[str] = []
        if preferred in self._search_engine:
            order.append(preferred)
        for fb in fallbacks:
            if fb in self._search_engine and fb not in order:
                order.append(fb)
        for name in self._search_engine:
            if name not in order:
                order.append(name)
        return order

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def _perform_search_with_engine(
        self,
        engine: WebSearchEngine,
        query: str,
        num_results: int,
        search_params: Dict[str, Any],
    ) -> List[SearchItem]:
        """Run blocking `perform_search` in a worker thread."""

        def _run() -> List[SearchItem]:
            try:
                items = list(
                    engine.perform_search(
                        query,
                        num_results=num_results,
                        lang=search_params.get("lang"),
                        country=search_params.get("country"),
                    )
                )
            except Exception:
                logger.exception(
                    "Search engine %s raised",
                    type(engine).__name__,
                )
                raise
            if not items:
                logger.warning(
                    "Search engine %s returned no results (blocked network, captcha, or empty SERP?)",
                    type(engine).__name__,
                )
            return items

        return await asyncio.to_thread(_run)


if __name__ == "__main__":
    async def _demo() -> None:
        res = await WebSearch().execute(
            query="Python asyncio", fetch_content=False, num_results=2
        )
        print(res.model_dump())

    asyncio.run(_demo())
