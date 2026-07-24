from __future__ import annotations

import asyncio
import random
import re
import shutil
from pathlib import Path

from .moderation import contains_blocked_term


# mIRC color codes (\x03 + optional fg[,bg] digits) must be matched before
# the generic control-byte class, otherwise the bare \x03 is consumed and
# the color digits leak into the text ("^C4rainbow" -> "4rainbow").
IRC_FORMAT_RE = re.compile(
    r"\x03(?:\d{1,2}(?:,\d{1,2})?)?"
    r"|[\x00-\x08\x0b\x0c\x0e-\x1f]"
)

URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)

# Lines like ".kb nick" / "!seen nick" are IRC service commands, not speech.
IRC_COMMAND_RE = re.compile(r"^[.!]\w")

REPLY_MARKER = "<Seeborg> "


def clean_dictionary_line(line: str, *, strip_urls: bool = True) -> str:
    line = IRC_FORMAT_RE.sub("", line)
    line = line.replace("�", "")
    if strip_urls:
        line = URL_RE.sub("", line)
    return " ".join(line.split()).strip()


class DictionaryCorpus:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.lines: list[str] = []

    def load(self) -> None:
        raw = self.path.read_text(encoding="utf-8", errors="replace")

        seen: set[str] = set()

        for source_line in raw.splitlines():
            line = clean_dictionary_line(source_line)

            if len(line) < 2 or len(line) > 300:
                continue
            if IRC_COMMAND_RE.match(line):
                continue

            normalized = line.casefold()
            if normalized in seen:
                continue

            seen.add(normalized)
            self.lines.append(line)

    def sample(self, count: int = 20) -> list[str]:
        if not self.lines:
            return []

        return random.sample(self.lines, k=min(count, len(self.lines)))


class SeeborgUnavailable(RuntimeError):
    pass


class SeeborgLinein:
    """
    Adapter for the real seeborg-linein binary (SeeBorg 0.51 beta).

    seeborg-linein is an interactive offline chat: it loads a hardcoded
    "lines.txt" from its working directory, prints a "> " prompt, and
    answers each stdin line with "<Seeborg> <reply>\\n" (flushed per reply).
    On EOF it rewrites lines.txt via SaveSettings, so every bot gets its
    own working directory to keep dictionaries isolated.
    """

    def __init__(
        self,
        *,
        binary_path: str | Path,
        dictionary_file: str | Path,
        work_dir: str | Path,
        blocklist: list[str] | None = None,
        reply_timeout_seconds: float = 20.0,
        startup_timeout_seconds: float = 180.0,
        seed_count: int = 8,
    ) -> None:
        self.binary_path = Path(binary_path).resolve()
        self.dictionary_file = Path(dictionary_file)
        self.work_dir = Path(work_dir)
        self.blocklist = blocklist or []
        self.reply_timeout_seconds = reply_timeout_seconds
        self.startup_timeout_seconds = startup_timeout_seconds
        self.seed_count = seed_count

        self.corpus = DictionaryCorpus(dictionary_file)
        self.corpus.load()

        self._process: asyncio.subprocess.Process | None = None
        self._warmed_up = False
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        if self._process and self._process.returncode is None:
            return

        if not self.binary_path.exists():
            raise SeeborgUnavailable(
                f"seeborg-linein binary not found at {self.binary_path}; "
                "run `make` in the seeborg/ directory"
            )

        self.work_dir.mkdir(parents=True, exist_ok=True)
        lines_txt = self.work_dir / "lines.txt"

        # Refresh the working copy when the source dictionary changed.
        if (
            not lines_txt.exists()
            or lines_txt.stat().st_mtime < self.dictionary_file.stat().st_mtime
        ):
            shutil.copyfile(self.dictionary_file, lines_txt)

        self._process = await asyncio.create_subprocess_exec(
            str(self.binary_path),
            cwd=str(self.work_dir),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        self._warmed_up = False

    async def close(self) -> None:
        process = self._process
        self._process = None

        if process is None or process.returncode is not None:
            return

        try:
            if process.stdin:
                process.stdin.close()
            await asyncio.wait_for(process.wait(), timeout=10.0)
        except (asyncio.TimeoutError, ProcessLookupError, OSError):
            try:
                process.kill()
                await process.wait()
            except ProcessLookupError:
                pass

    async def generate_seed(self, discussion: str) -> list[str]:
        """
        Feed recent discussion lines to Seeborg and collect its markov
        replies as stylistic seed material. Falls back to raw corpus
        samples if the subprocess is unavailable.
        """
        prompts = self._extract_prompts(discussion)
        seeds: list[str] = []
        seen: set[str] = set()

        try:
            async with self._lock:
                await self.start()

                for prompt in prompts:
                    reply = await self._ask(prompt)
                    cleaned = clean_dictionary_line(reply)

                    if len(cleaned) < 2 or len(cleaned) > 300:
                        continue
                    if contains_blocked_term(cleaned, self.blocklist):
                        continue

                    normalized = cleaned.casefold()
                    if normalized in seen:
                        continue

                    seen.add(normalized)
                    seeds.append(cleaned)

                    if len(seeds) >= self.seed_count:
                        break
        except (SeeborgUnavailable, asyncio.TimeoutError, OSError):
            await self.close()

        # Pad with corpus flavor so the LLM always gets material.
        for line in self.corpus.sample(self.seed_count * 2):
            if len(seeds) >= self.seed_count:
                break
            if contains_blocked_term(line, self.blocklist):
                continue
            normalized = line.casefold()
            if normalized not in seen:
                seen.add(normalized)
                seeds.append(line)

        return seeds

    def _extract_prompts(self, discussion: str) -> list[str]:
        prompts: list[str] = []

        for raw_line in discussion.splitlines():
            line = raw_line.strip()

            if not line or line.startswith(("Board", "Thread subject", "Task:")):
                continue

            # Discussion lines are rendered as "author: body".
            _, separator, body = line.partition(": ")
            text = body if separator else line
            text = clean_dictionary_line(text)

            if len(text) >= 3:
                prompts.append(text[:400])

        return prompts[-self.seed_count * 2:] or ["hello"]

    async def _ask(self, prompt: str) -> str:
        process = self._process

        if process is None or process.returncode is not None:
            raise SeeborgUnavailable("seeborg-linein process is not running")

        assert process.stdin and process.stdout

        # A leading '!' would trigger ParseCommands instead of Reply.
        line = prompt.replace("\n", " ").replace("\r", " ").lstrip("!").strip()
        if not line:
            line = "hm"

        timeout = (
            self.startup_timeout_seconds
            if not self._warmed_up
            else self.reply_timeout_seconds
        )

        try:
            process.stdin.write(line.encode("utf-8", errors="replace") + b"\n")
            await process.stdin.drain()

            while True:
                raw = await asyncio.wait_for(
                    process.stdout.readline(),
                    timeout=timeout,
                )

                if not raw:
                    raise SeeborgUnavailable("seeborg-linein closed stdout")

                decoded = raw.decode("utf-8", errors="replace")

                # Replies arrive as "> <Seeborg> text"; startup banner and
                # dictionary statistics lines are skipped here.
                marker_index = decoded.find(REPLY_MARKER)
                if marker_index >= 0:
                    self._warmed_up = True
                    return decoded[marker_index + len(REPLY_MARKER):].strip()
        except (BrokenPipeError, ConnectionResetError) as exc:
            raise SeeborgUnavailable("seeborg-linein pipe broke") from exc
