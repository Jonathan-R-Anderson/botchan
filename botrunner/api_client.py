from __future__ import annotations

from typing import Any

import httpx

from .models import Board, ThreadListing, ThreadSummary


class ForumAPIError(RuntimeError):
    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class ForumClient:
    def __init__(
        self,
        base_url: str,
        token: str,
        boards_endpoint: str,
        post_endpoint: str,
        catalog_endpoint: str,
        thread_endpoint: str,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.boards_endpoint = boards_endpoint
        self.post_endpoint = post_endpoint
        self.catalog_endpoint = catalog_endpoint
        self.thread_endpoint = thread_endpoint

        # Reading is public; only /api/v1/bot/post requires the token,
        # so the bearer header is attached per-request when posting.
        self._auth_headers = {"Authorization": f"Bearer {token}"}
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Accept": "application/json",
                "User-Agent": "DesignatedForumBot/1.0",
            },
            timeout=httpx.Timeout(30.0),
            follow_redirects=True,
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def discover_boards(self) -> list[Board]:
        response = await self.client.get(self.boards_endpoint)
        self._raise(response)
        payload = response.json()

        # The discovery response advertises the posting endpoint; prefer
        # it over the configured default so the two can't drift apart.
        advertised = payload.get("post_endpoint")
        if advertised:
            self.post_endpoint = advertised

        return [Board.model_validate(item) for item in payload.get("boards", [])]

    async def list_threads(self, board: str) -> list[ThreadListing]:
        endpoint = self.catalog_endpoint.format(board=board)
        response = await self.client.get(endpoint)
        self._raise(response)

        payload = response.json()
        return [
            ThreadListing.model_validate(item)
            for item in payload.get("threads", [])
        ]

    async def read_thread(
        self,
        board: str,
        thread_id: int | str,
    ) -> ThreadSummary:
        endpoint = self.thread_endpoint.format(
            board=board,
            thread_id=thread_id,
        )
        response = await self.client.get(endpoint)
        self._raise(response)

        payload = response.json()
        payload.setdefault("board", board)
        payload.setdefault("thread", thread_id)
        return ThreadSummary.model_validate(payload)

    async def create_thread(
        self,
        board: str,
        body: str,
        *,
        subject: str | None = None,
        name: str | None = None,
        spoiler: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "board": board,
            "body": body,
        }

        if subject:
            payload["subject"] = subject
        if name:
            payload["name"] = name
        if spoiler:
            payload["spoiler"] = True

        return await self._post(payload)

    async def reply(
        self,
        board: str,
        thread_id: int | str,
        body: str,
        *,
        name: str | None = None,
        spoiler: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "board": board,
            "thread": thread_id,
            "body": body,
        }

        if name:
            payload["name"] = name
        if spoiler:
            payload["spoiler"] = True

        return await self._post(payload)

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = await self.client.post(
            self.post_endpoint,
            json=payload,
            headers=self._auth_headers,
        )
        self._raise(response)
        return response.json()

    @staticmethod
    def _raise(response: httpx.Response) -> None:
        if response.is_success:
            return

        retry_after_header = response.headers.get("Retry-After")
        retry_after: float | None = None

        if retry_after_header:
            try:
                retry_after = float(retry_after_header)
            except ValueError:
                retry_after = None

        content_type = response.headers.get("Content-Type", "unknown")

        # An HTML body on a JSON endpoint almost always means the path is
        # wrong and a web server served its own error page, so say that
        # instead of dumping 1000 characters of markup into the dashboard.
        if "html" in content_type.lower():
            detail = "<HTML error page, not JSON — check the endpoint path>"
        else:
            # API errors are JSON objects with an "error" message
            # (401 bad token, 403 wrong IP or non-bot board, 404 unknown
            # board/thread).
            try:
                detail = response.json()["error"]
            except (ValueError, KeyError, TypeError):
                detail = response.text[:400]

        raise ForumAPIError(
            f"{response.request.method} {response.request.url} "
            f"-> HTTP {response.status_code} ({content_type}); "
            f"retry_after={retry_after_header!r}; response={detail!r}",
            retry_after=retry_after,
        )
