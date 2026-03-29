"""Base scraper with rate limiting, retry, and structured logging."""

from __future__ import annotations

import httpx
import structlog
from aiolimiter import AsyncLimiter
from tenacity import (
    retry,
    retry_if_exception,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from providence.config import Settings

logger = structlog.get_logger()


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code >= 500:
        return True
    if isinstance(exc, (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError)):
        return True
    return False


class BaseScraper:
    """Shared HTTP client infrastructure for all scrapers.

    Features:
    - Async httpx client with configurable timeout
    - Per-instance rate limiting (aiolimiter)
    - Automatic retry on timeout and 5xx errors (tenacity)
    - 4xx errors (403 CSRF, 404 not found) are NOT retried
    - Structured logging via structlog
    - Context manager for proper resource cleanup
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(settings.scrape_timeout_sec),
            follow_redirects=True,
            headers={"User-Agent": "Providence/0.1 (+autorace-research)"},
        )
        self._rate_limiter = AsyncLimiter(max_rate=1, time_period=settings.scrape_interval_sec)
        self._log = logger.bind(scraper=self.__class__.__name__)

    async def __aenter__(self) -> BaseScraper:
        return self

    async def __aexit__(self, *args: object) -> None:
        await self.client.aclose()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=(retry_if_exception_type(httpx.TimeoutException) | retry_if_exception(_is_retryable)),
        reraise=True,
    )
    async def _request(self, method: str, url: str, **kwargs: object) -> httpx.Response:
        """Rate-limited HTTP request with retry on timeout/5xx only."""
        async with self._rate_limiter:
            self._log.debug("http_request", method=method, url=url)
            resp = await self.client.request(method, url, **kwargs)
            resp.raise_for_status()
            return resp
