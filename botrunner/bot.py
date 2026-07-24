from __future__ import annotations

import random
from collections.abc import Awaitable, Callable
from typing import Any

import yaml

from .api_client import ForumClient
from .llm import OpenAICompatibleLLM
from .memory import BotMemory
from .meme_search import KnowYourMemeSearch
from .models import Board, ProgressEvent, ThreadSummary
from .moderation import validate_post
from .seeborg import SeeborgLinein


ProgressCallback = Callable[[ProgressEvent], Awaitable[None]]


class ForumBot:
    def __init__(
        self,
        *,
        bot_id: str,
        name: str | None,
        personality_file: str,
        preferred_boards: list[str],
        meme_probability: float,
        forum: ForumClient,
        memory: BotMemory,
        seeborg: SeeborgLinein,
        llm: OpenAICompatibleLLM,
        meme_search: KnowYourMemeSearch | None,
        blocklist: list[str],
        dry_run: bool,
        progress_callback: ProgressCallback,
    ) -> None:
        self.bot_id = bot_id
        self.name = name
        self.personality_file = personality_file
        self.preferred_boards = preferred_boards
        self.meme_probability = max(0.0, min(1.0, meme_probability))
        self.forum = forum
        self.memory = memory
        self.seeborg = seeborg
        self.llm = llm
        self.meme_search = meme_search
        self.blocklist = blocklist
        self.dry_run = dry_run
        self.progress_callback = progress_callback

        with open(personality_file, "r", encoding="utf-8") as handle:
            self.personality = yaml.safe_load(handle)

    async def progress(
        self,
        stage: str,
        percent: int,
        message: str,
        **details: Any,
    ) -> None:
        await self.progress_callback(
            ProgressEvent(
                bot_id=self.bot_id,
                stage=stage,
                percent=max(0, min(100, percent)),
                message=message,
                details=details,
            )
        )

    async def run_once(self) -> None:
        await self.progress("discover", 5, "Discovering bot boards")
        boards = await self.forum.discover_boards()

        if self.preferred_boards:
            boards = [
                board
                for board in boards
                if board.board in self.preferred_boards
            ]

        if not boards:
            await self.progress(
                "idle",
                100,
                "No designated boards are available",
            )
            return

        board = self.choose_board(boards)
        await self.progress(
            "select-board",
            15,
            f"Selected /{board.board}/",
        )

        thread = await self.choose_thread(board)
        await self.progress(
            "read",
            25,
            (
                f"Read thread {thread.thread_id}"
                if thread
                else "Preparing a new thread"
            ),
        )

        discussion = self.render_discussion(thread, board)
        memories = await self.memory.retrieve(discussion, limit=12)

        await self.progress(
            "memory",
            38,
            f"Retrieved {len(memories)} relevant memories",
        )

        seed = await self.seeborg.generate_seed(discussion)

        await self.progress(
            "seeborg",
            50,
            f"Collected {len(seed)} Seeborg style fragments",
        )

        generated = await self.llm.generate(
            personality=self.personality,
            board_description=board.description,
            discussion=discussion,
            memories=memories,
            seeborg_seed=seed,
        )

        await self.progress(
            "generate",
            68,
            "Generated candidate response",
        )

        body = validate_post(generated.body, blocklist=self.blocklist)

        if await self.memory.is_duplicate(body):
            raise RuntimeError(
                f"{self.bot_id} generated a near-duplicate post"
            )

        if (
            self.meme_search
            and generated.meme_query
            and random.random() < self.meme_probability
        ):
            await self.progress(
                "meme-search",
                76,
                f"Searching Know Your Meme for {generated.meme_query!r}",
            )

            results = await self.meme_search.search(
                generated.meme_query,
                limit=3,
            )

            if results:
                selected = random.choice(results)

                # Prefer linking to the KYM page rather than hotlinking
                # an image whose hosting or reuse policy is unclear.
                body += (
                    "\n\n"
                    f"[Relevant meme: {selected.title}]"
                    f"({selected.page_url})"
                )

        await self.progress(
            "validation",
            85,
            "Candidate passed local validation",
        )

        if self.dry_run:
            await self.memory.record_post(
                board.board,
                thread.thread_id if thread else None,
                body,
            )

            await self.progress(
                "complete",
                100,
                "Dry run complete; no post was sent",
                preview=body,
            )
            return

        if thread:
            response = await self.forum.reply(
                board=board.board,
                thread_id=thread.thread_id,
                body=body,
                name=self.name,
            )
        else:
            response = await self.forum.create_thread(
                board=board.board,
                body=body,
                subject=generated.subject,
                name=self.name,
            )

        returned_thread_id = response.get("thread") or (
            thread.thread_id if thread else None
        )

        await self.memory.record_post(
            board.board,
            returned_thread_id,
            body,
        )

        await self.memory.remember(
            "own_post",
            body,
            metadata={
                "board": board.board,
                "thread_id": returned_thread_id,
                "response": response,
            },
            importance=0.65,
        )

        await self.progress(
            "complete",
            100,
            "Post submitted successfully",
            response=response,
        )

    @staticmethod
    def choose_board(boards: list[Board]) -> Board:
        return random.choice(boards)

    async def choose_thread(
        self,
        board: Board,
    ) -> ThreadSummary | None:
        threads = await self.forum.list_threads(board.board)

        if not threads or random.random() < 0.20:
            return None

        candidates = threads[:20]
        selected = random.choice(candidates)

        return await self.forum.read_thread(
            board.board,
            selected.id,
        )

    @staticmethod
    def render_discussion(
        thread: ThreadSummary | None,
        board: Board,
    ) -> str:
        if thread is None:
            return (
                f"Board title: {board.title}\n"
                f"Board description: {board.description}\n"
                "Task: Start a new thread appropriate for this board."
            )

        lines = [
            f"Board: /{board.board}/",
            f"Board description: {board.description}",
            f"Thread subject: {thread.subject or '(none)'}",
            "",
        ]

        for post in thread.posts[-30:]:
            poster = post.poster or "Anonymous"
            lines.append(f"{poster}: {post.body}")

        return "\n".join(lines)
