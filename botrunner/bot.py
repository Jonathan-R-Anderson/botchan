from __future__ import annotations

import logging
import random
import re
from collections.abc import Awaitable, Callable
from typing import Any

import yaml

from .api_client import ForumClient
from .llm import OpenAICompatibleLLM
from .memory import BotMemory
from .meme_search import KnowYourMemeSearch, MemeResult
from .models import Board, ProgressEvent, ThreadSummary
from .moderation import validate_post
from .seeborg import SeeborgLinein


ProgressCallback = Callable[[ProgressEvent], Awaitable[None]]

log = logging.getLogger("botchan.bot")

STOPWORDS = frozenset(
    """
    a about after all also am an and any are as at be because been before
    but by can could did do does dont for from get got had has have he her
    him his how i if im in into is it its just like me more most my no not
    now of off on one only or other our out over said she so some than
    that the their them then there they this to too up us very was we were
    what when which who will with would you your
    """.split()
)


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

        # New threads must carry a Know Your Meme image; replies attach
        # one only occasionally. The comment itself drives the search,
        # with progressively broader fallbacks.
        image_queries = [
            self.comment_query(body),
            generated.meme_query or "",
            generated.subject or "",
            board.title or board.board,
        ]

        image: MemeResult | None = None

        if thread is None:
            if not self.meme_search:
                raise RuntimeError(
                    "meme_search is disabled but new threads require an image"
                )

            await self.progress(
                "meme-search",
                76,
                "Finding the required OP image on Know Your Meme",
            )
            image = await self.find_meme_image(image_queries)

            if image is None:
                raise RuntimeError(
                    "no Know Your Meme image found for the OP; "
                    "skipping this cycle"
                )
        elif self.meme_search and random.random() < self.meme_probability:
            await self.progress(
                "meme-search",
                76,
                "Looking for a meme image to attach",
            )
            image = await self.find_meme_image(image_queries)

        media: tuple[str, bytes, str] | None = None

        if image:
            try:
                media = await self.meme_search.download_image(
                    image.image_url
                )
            except Exception as exc:
                log.warning(
                    "%s: image download failed for %s: %s",
                    self.bot_id,
                    image.image_url,
                    exc,
                )

            if media is None:
                if thread is None:
                    raise RuntimeError(
                        "could not download the required OP image; "
                        "skipping this cycle"
                    )
                log.warning(
                    "%s: posting reply without the image",
                    self.bot_id,
                )
            else:
                log.info(
                    "%s: image ready to attach: %s (%d bytes, %s)",
                    self.bot_id,
                    media[0],
                    len(media[1]),
                    media[2],
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

            log.info(
                "%s: dry run — would have posted to /%s/ (thread %s)%s",
                self.bot_id,
                board.board,
                thread.thread_id if thread else "new",
                f" with image {media[0]}" if media else "",
            )
            await self.progress(
                "complete",
                100,
                "Dry run complete; no post was sent",
                preview=body,
            )
            return

        if thread:
            log.info(
                "%s: replying to /%s/ thread %s",
                self.bot_id,
                board.board,
                thread.thread_id,
            )
            response = await self.forum.reply(
                board=board.board,
                thread_id=thread.thread_id,
                body=body,
                name=self.name,
                media=media,
            )
        else:
            log.info(
                "%s: opening a thread on /%s/ (subject %r)",
                self.bot_id,
                board.board,
                generated.subject,
            )
            response = await self.forum.create_thread(
                board=board.board,
                body=body,
                subject=generated.subject,
                name=self.name,
                media=media,
            )

        log.info(
            "%s: posted post %s in thread %s (%s)",
            self.bot_id,
            response.get("post"),
            response.get("thread"),
            response.get("url"),
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
    def comment_query(text: str, word_count: int = 4) -> str:
        """Distill a comment into a few content words for a KYM search."""
        words = re.findall(r"[a-z][a-z']{2,}", text.lower())

        unique: list[str] = []
        for word in words:
            if word in STOPWORDS or word in unique:
                continue
            unique.append(word)
            if len(unique) >= word_count:
                break

        return " ".join(unique)

    async def find_meme_image(
        self,
        queries: list[str],
    ) -> MemeResult | None:
        """Try each query in order; return the first result with an image."""
        tried: set[str] = set()

        for query in queries:
            query = (query or "").strip()
            if not query or query.lower() in tried:
                continue
            tried.add(query.lower())

            try:
                results = await self.meme_search.search(query, limit=5)
            except Exception as exc:
                log.warning(
                    "%s: Know Your Meme search failed for %r: %s",
                    self.bot_id,
                    query,
                    exc,
                )
                continue

            with_images = [r for r in results if r.image_url]
            if with_images:
                choice = random.choice(with_images)
                log.info(
                    "%s: KYM image found for %r: %s",
                    self.bot_id,
                    query,
                    choice.image_url,
                )
                return choice

        # Nothing matched any query — fall back to a completely random
        # KYM entry so an image is still available.
        try:
            fallback = await self.meme_search.random_image()
        except Exception as exc:
            log.warning(
                "%s: random KYM image fetch failed: %s",
                self.bot_id,
                exc,
            )
            return None

        if fallback:
            log.info(
                "%s: using random KYM image: %s",
                self.bot_id,
                fallback.image_url,
            )
        return fallback

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
