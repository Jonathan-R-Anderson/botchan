from __future__ import annotations

import logging
import random
import re
from collections.abc import Awaitable, Callable
from typing import Any

import yaml
from pydantic import ValidationError

from .api_client import ForumClient
from .llm import LLMError, OpenAICompatibleLLM
from .memory import BotMemory
from .meme_search import KnowYourMemeSearch, MemeResult
from .models import Board, GeneratedPost, ProgressEvent, ThreadSummary
from .moderation import ContentRejected, validate_post
from .seeborg import SeeborgLinein


ProgressCallback = Callable[[ProgressEvent], Awaitable[None]]

log = logging.getLogger("botchan.bot")

GENERATION_ATTEMPTS = 3

# One angle + one length gets rolled per generation attempt, so even the
# same thread read twice produces differently shaped posts.
ANGLES = [
    "mock the most recent post's take in a sarcastic way, acting like it completely misses the obvious point while sounding overly confident about your own interpretation",
    "derail the thread by introducing a completely unrelated tangent that you somehow insist is more interesting than the original discussion",
    "tell a short but detailed personal anecdote that is obviously fabricated, complete with ridiculous specifics that make it impossible to believe",
    "ask one genuinely sincere-sounding but incredibly dumb question that would make everyone wonder if you actually read the thread",
    "make an absurd comparison between the topic and something completely unrelated, then defend the comparison as though it makes perfect sense",
    "pick one exact phrase from the thread and obsess over it, ignoring the rest of the conversation entirely while treating that phrase like it's the most important part",
    "confidently state an obviously incorrect fact about the topic, presenting it with absolute certainty and refusing to acknowledge that it might be wrong",
    "accuse nearly everyone participating in the thread of secretly being bots, shills, or automated accounts without providing any convincing evidence",
    "drop a deliberately controversial hot take related to the board's topic and present it like it's a universally accepted truth",
    "reply as though you completely misunderstood what the original post was about, confidently arguing against points that nobody actually made",
    "act like you're the only person participating in the thread who has any taste, standards, or understanding of the subject being discussed",
    "start a surprisingly passionate argument over an extremely small, trivial, or irrelevant detail that nobody else seemed to care about",
    "reminisce about how the board, community, or internet used to be so much better years ago, insisting that everything has gone downhill ever since",
    "review the thread like a disappointed professional critic, assigning it an imaginary rating and explaining all the ways it failed to meet your expectations",
    "declare that you've completely won an argument despite the fact that nobody was actually debating you in the first place",
    "pretend to be an expert on the subject while making it obvious you only skimmed the thread",
    "focus on one tiny typo or grammatical mistake instead of responding to the actual discussion",
    "invent an imaginary conspiracy that supposedly explains why everyone in the thread agrees with each other",
    "respond as though you're personally offended by something completely harmless in the post",
    "treat an obvious joke as if it were a serious policy proposal",
    "act like you've seen this exact discussion a thousand times and you're exhausted by everyone repeating themselves",
    "insist that the thread proves society is collapsing for reasons that barely relate to the topic",
    "reply with dramatic overconfidence despite offering almost no reasoning",
    "pretend you have inside information that changes everything but refuse to elaborate",
    "argue from an extremely niche perspective that almost nobody else would have considered",
    "make everything about yourself by repeatedly steering the conversation back to your own experiences",
    "reply with fake nostalgia for an event that almost certainly never happened",
    "respond as though the discussion is life-or-death when it's actually trivial",
    "play devil's advocate so aggressively that people can't tell whether you're serious",
    "nitpick the wording of the original post instead of engaging with its actual meaning",
    "write as if you're reluctantly educating everyone else because nobody seems to understand the obvious",
    "take the least charitable interpretation of every comment you respond to",
    "pretend you're calmly explaining things while sounding increasingly irritated",
    "randomly bring up an unrelated historical event and insist the situations are basically identical",
    "reply as though the thread is secretly about a completely different topic than everyone else thinks",
    "act like everyone has forgotten one incredibly obvious fact that you repeatedly remind them about",
    "turn the discussion into an imaginary competition and announce arbitrary winners and losers",
    "pretend to misunderstand a common expression literally and base your entire reply around that misunderstanding",
    "respond in a tone that sounds unnecessarily formal for the discussion, as if writing an academic critique",
    "invent fake statistics that sound believable enough to make people hesitate for a moment",
    "behave like you're trying to mediate the conversation while quietly making it even worse",
    "frame your opinion as an unpopular truth that nobody else is brave enough to admit",
    "pretend you're correcting misinformation while introducing even more misinformation",
    "reply as though everyone else has missed an incredibly obvious hidden meaning in the thread",
    "act disappointed that the conversation never addressed a bizarre edge case that only you care about",
    "write as if you're ending the discussion forever with the ultimate final word",
    "reply like a dramatic movie narrator describing an otherwise ordinary internet argument",
    "pretend every comment is part of an elaborate social experiment",
    "respond as though the thread belongs in a museum because it's such a perfect example of internet behavior",
]

LENGTHS = [
    "a single punchy one-liner that lands immediately",
    "one short sentence with a memorable punchline",
    "two concise sentences that quickly build to a joke",
    "two or three short sentences that sound natural and conversational",
    "three or four sentences with a steady buildup and a funny ending",
    "a medium-length reply of four to six sentences that becomes increasingly ridiculous",
    "a detailed rant of seven to ten sentences that starts reasonable before slowly becoming completely unhinged",
    "a long wall of text consisting of ten to fifteen sentences that rambles, contradicts itself, and somehow circles back to its original point",
    "a surprisingly detailed response of one full paragraph that reads like someone took the discussion far too seriously",
    "two long paragraphs that begin thoughtfully but spiral into complete nonsense by the end",
    "a rambling essay of fifteen to twenty sentences with multiple unnecessary digressions",
    "an overly detailed explanation that could have been answered in a single sentence but refuses to stop talking",
    "a stream-of-consciousness rant that jumps unpredictably between ideas while still vaguely staying on topic",
    "a dramatic monologue that sounds like you're delivering a speech to a crowd instead of replying to a forum thread",
    "a fake analytical breakdown with multiple points, examples, and conclusions despite the topic not warranting that much effort",
]

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

        recent_posts = await self.memory.recent_own_posts(limit=8)

        generated: GeneratedPost | None = None
        body: str | None = None
        feedback: str | None = None
        last_reason = "generation failed"

        for attempt in range(1, GENERATION_ATTEMPTS + 1):
            directive = (
                f"Angle: {random.choice(ANGLES)}. "
                f"Length: {random.choice(LENGTHS)}."
            )

            # A shuffled subset of the style seeds, so the same markov
            # material doesn't steer every attempt the same way.
            seed_sample = random.sample(
                seed,
                k=min(len(seed), max(3, len(seed) - attempt + 1)),
            ) if seed else []

            await self.progress(
                "generate",
                68,
                f"Generating candidate response (attempt {attempt})",
            )

            try:
                candidate = await self.llm.generate(
                    personality=self.personality,
                    board_description=board.description,
                    discussion=discussion,
                    memories=memories,
                    seeborg_seed=seed_sample,
                    recent_own_posts=recent_posts,
                    retry_feedback=feedback,
                    style_directive=directive,
                )
                candidate_body = validate_post(
                    candidate.body,
                    blocklist=self.blocklist,
                )
            except (LLMError, ValidationError, ContentRejected) as exc:
                last_reason = str(exc)
                feedback = (
                    "Your previous attempt was rejected: "
                    f"{last_reason}. Produce a valid post this time."
                )
                log.info(
                    "%s: generation attempt %d rejected: %s",
                    self.bot_id,
                    attempt,
                    last_reason,
                )
                continue

            if await self.memory.is_duplicate(candidate_body):
                last_reason = "near-duplicate of a recent post"
                feedback = (
                    "Your previous attempt was rejected as a "
                    "near-duplicate of something you already posted. "
                    "Say something NEW — different topic, different "
                    "angle, different phrasing."
                )
                log.info(
                    "%s: generation attempt %d was a near-duplicate",
                    self.bot_id,
                    attempt,
                )
                continue

            generated, body = candidate, candidate_body
            break

        if generated is None or body is None:
            raise RuntimeError(
                f"{self.bot_id}: no usable post after "
                f"{GENERATION_ATTEMPTS} attempts (last: {last_reason})"
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
