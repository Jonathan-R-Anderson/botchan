from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from urllib.parse import quote_plus, urljoin

import httpx
from bs4 import BeautifulSoup


@dataclass
class MemeResult:
    title: str
    page_url: str
    image_url: str | None


# Status-filter navigation that also lives under /memes/ and would
# otherwise look like search hits.
CHROME_PATHS = frozenset(
    {
        "/memes/submissions",
        "/memes/confirmed",
        "/memes/newsworthy",
        "/memes/deadpool",
        "/memes/popular",
        "/memes/researching",
        "/memes/all",
        "/memes/submit",
    }
)


class KnowYourMemeSearch:
    """
    HTML search fallback — Know Your Meme has no official search API.

    Confirm that automated access is permitted before enabling this in
    production. Cache results and keep request frequency low.
    """

    def __init__(
        self,
        *,
        user_agent: str,
        minimum_interval_seconds: int = 60,
        cache_ttl_seconds: int = 86400,
    ) -> None:
        self.user_agent = user_agent
        self.minimum_interval_seconds = minimum_interval_seconds
        self.cache_ttl_seconds = cache_ttl_seconds
        self._last_request = 0.0
        self._lock = asyncio.Lock()
        self._cache: dict[str, tuple[float, list[MemeResult]]] = {}

    async def search(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[MemeResult]:
        normalized = " ".join(query.lower().split())

        if not normalized:
            return []

        cached = self._cache.get(normalized)
        if cached and time.monotonic() - cached[0] < self.cache_ttl_seconds:
            return cached[1][:limit]

        async with self._lock:
            elapsed = time.monotonic() - self._last_request
            delay = self.minimum_interval_seconds - elapsed

            if delay > 0:
                await asyncio.sleep(delay)

            search_url = (
                "https://knowyourmeme.com/search"
                f"?q={quote_plus(normalized)}"
            )

            async with httpx.AsyncClient(
                headers={"User-Agent": self.user_agent},
                timeout=30.0,
                follow_redirects=True,
            ) as client:
                response = await client.get(search_url)
                response.raise_for_status()

            self._last_request = time.monotonic()

        soup = BeautifulSoup(response.text, "html.parser")
        results: list[MemeResult] = []
        seen: set[str] = set()

        # KYM markup may change. Keep selectors isolated here.
        # Query matches are entry cards inside .groups; .search-results
        # is a generic newest-entries block that doubles as a fallback.
        # Anchors outside both containers are navigation/site chrome.
        containers = soup.select(".groups") + soup.select(".search-results")

        for anchor in (a for c in containers for a in c.find_all("a")):
            href = anchor.get("href")
            if not href or "/memes/" not in href:
                continue
            if href in CHROME_PATHS or href.startswith("/sensitive/"):
                continue

            page_url = urljoin("https://knowyourmeme.com", href)

            if page_url in seen:
                continue

            title = (
                anchor.get("title")
                or anchor.get_text(" ", strip=True)
                or normalized
            )

            image = anchor.find("img")
            image_url = None

            if image:
                source = (
                    image.get("src")
                    or image.get("data-src")
                    or image.get("data-original")
                )
                # Entry covers come from i.kym-cdn.com; anything under
                # /assets/ is a site icon, not a meme image.
                if (
                    source
                    and "/assets/" not in source
                    and not source.endswith(".svg")
                ):
                    image_url = urljoin(
                        "https://knowyourmeme.com",
                        source,
                    )

            seen.add(page_url)
            results.append(
                MemeResult(
                    title=title[:200],
                    page_url=page_url,
                    image_url=image_url,
                )
            )

            if len(results) >= limit:
                break

        self._cache[normalized] = (time.monotonic(), results)
        return results

    async def random_image(self) -> MemeResult | None:
        """Fetch a random KYM entry and return its cover image.

        Uses the site's own /random redirect; results are deliberately
        not cached. Returns None if the entry has no usable cover or
        landed on a sensitive-flagged page.
        """
        async with self._lock:
            elapsed = time.monotonic() - self._last_request
            delay = self.minimum_interval_seconds - elapsed

            if delay > 0:
                await asyncio.sleep(delay)

            async with httpx.AsyncClient(
                headers={"User-Agent": self.user_agent},
                timeout=30.0,
                follow_redirects=True,
            ) as client:
                response = await client.get(
                    "https://knowyourmeme.com/random"
                )
                response.raise_for_status()

            self._last_request = time.monotonic()

        page_url = str(response.url)
        if "/sensitive/" in page_url:
            return None

        soup = BeautifulSoup(response.text, "html.parser")

        image_meta = soup.find("meta", property="og:image")
        image_url = image_meta.get("content") if image_meta else None

        if not image_url or "/assets/" in image_url:
            return None

        title_meta = soup.find("meta", property="og:title")
        title = title_meta.get("content") if title_meta else page_url

        return MemeResult(
            title=str(title)[:200],
            page_url=page_url,
            image_url=str(image_url),
        )
