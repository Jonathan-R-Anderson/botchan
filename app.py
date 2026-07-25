from __future__ import annotations

import asyncio
import os
from pathlib import Path

import yaml
from rich.console import Console
from rich.live import Live
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)

from botrunner.api_client import ForumClient
from botrunner.bot import ForumBot
from botrunner.llm import OpenAICompatibleLLM
from botrunner.logs import setup_logging
from botrunner.memory import BotMemory
from botrunner.meme_search import KnowYourMemeSearch
from botrunner.models import ProgressEvent
from botrunner.moderation import load_blocklist
from botrunner.scheduler import run_bot_forever
from botrunner.seeborg import SeeborgLinein


console = Console()


class ProgressDashboard:
    def __init__(self) -> None:
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.fields[bot_id]}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("{task.description}"),
            TimeElapsedColumn(),
        )
        self.tasks: dict[str, int] = {}
        self.lock = asyncio.Lock()

    async def update(self, event: ProgressEvent) -> None:
        async with self.lock:
            task_id = self.tasks.get(event.bot_id)

            if task_id is None:
                task_id = self.progress.add_task(
                    event.message,
                    total=100,
                    completed=event.percent,
                    bot_id=event.bot_id,
                )
                self.tasks[event.bot_id] = task_id
            else:
                self.progress.update(
                    task_id,
                    completed=event.percent,
                    description=event.message,
                )

            if event.details.get("preview"):
                console.log(
                    f"[cyan]{event.bot_id} preview:[/cyan] "
                    f"{event.details['preview']}"
                )


async def main() -> None:
    with open("config.yaml", "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    runtime = config["runtime"]
    forum_config = config["forum"]
    llm_config = config["llm"]
    seeborg_config = config.get("seeborg", {})
    meme_config = config.get("meme_search", {})
    moderation_config = config.get("moderation", {})
    log_config = config.get("logging", {})

    log_file = log_config.get("file", "data/logs/botchan.log")
    log = setup_logging(log_file, level=log_config.get("level", "INFO"))
    console.log(f"Logging to {log_file}")

    if os.getenv("POSTING_ENABLED", "").lower() in ("0", "false", "no"):
        runtime["dry_run"] = True
        console.log("POSTING_ENABLED kill switch active; forcing dry run")

    log.info(
        "botchan starting; dry_run=%s",
        bool(runtime.get("dry_run", True)),
    )

    blocklist = load_blocklist(moderation_config.get("blocklist_file"))

    if not blocklist:
        console.log(
            "[yellow]Warning:[/yellow] blocklist is empty; populate "
            f"{moderation_config.get('blocklist_file', 'dictionaries/blocklist.txt')!r} "
            "before disabling dry_run"
        )

    dashboard = ProgressDashboard()

    llm = OpenAICompatibleLLM(
        base_url=llm_config["base_url"],
        model=llm_config["model"],
        api_key_env=llm_config["api_key_env"],
        temperature=float(llm_config.get("temperature", 0.9)),
        temperature_jitter=float(
            llm_config.get("temperature_jitter", 0.0)
        ),
        presence_penalty=float(
            llm_config.get("presence_penalty", 0.0)
        ),
        frequency_penalty=float(
            llm_config.get("frequency_penalty", 0.0)
        ),
        max_tokens=int(llm_config.get("max_tokens", 350)),
    )

    meme_search = None
    if meme_config.get("enabled", False):
        meme_search = KnowYourMemeSearch(
            user_agent=meme_config["user_agent"],
            minimum_interval_seconds=int(
                meme_config.get("minimum_interval_seconds", 60)
            ),
            cache_ttl_seconds=int(
                meme_config.get("cache_ttl_seconds", 86400)
            ),
        )

    bots: list[ForumBot] = []

    for bot_config in config["bots"]:
        token = os.getenv(bot_config["token_env"])

        if not token:
            raise RuntimeError(
                f"Missing environment variable {bot_config['token_env']}"
            )

        bot_id = bot_config["id"]

        memory = BotMemory(
            Path("data/memory") / f"{bot_id}.sqlite3",
            bot_id=bot_id,
        )
        await memory.initialize()

        forum = ForumClient(
            base_url=forum_config["base_url"],
            token=token,
            boards_endpoint=forum_config["boards_endpoint"],
            post_endpoint=forum_config["post_endpoint"],
            catalog_endpoint=forum_config["catalog_endpoint"],
            thread_endpoint=forum_config["thread_endpoint"],
        )

        seeborg = SeeborgLinein(
            binary_path=seeborg_config.get(
                "binary", "seeborg/seeborg-linein"
            ),
            dictionary_file=bot_config["dictionary_file"],
            work_dir=Path(
                seeborg_config.get("work_dir", "data/seeborg")
            ) / bot_id,
            blocklist=blocklist,
            reply_timeout_seconds=float(
                seeborg_config.get("reply_timeout_seconds", 20)
            ),
            startup_timeout_seconds=float(
                seeborg_config.get("startup_timeout_seconds", 180)
            ),
            seed_count=int(seeborg_config.get("seed_count", 8)),
        )

        bots.append(
            ForumBot(
                bot_id=bot_id,
                name=bot_config.get("name"),
                personality_file=bot_config["personality_file"],
                preferred_boards=bot_config.get("preferred_boards", []),
                meme_probability=float(
                    bot_config.get("meme_probability", 0)
                ),
                forum=forum,
                memory=memory,
                seeborg=seeborg,
                llm=llm,
                meme_search=meme_search,
                blocklist=blocklist,
                dry_run=bool(runtime.get("dry_run", True)),
                progress_callback=dashboard.update,
            )
        )

    minimum_delay = int(runtime["minimum_post_interval_seconds"])
    maximum_delay = int(runtime["maximum_post_interval_seconds"])

    try:
        with Live(
            dashboard.progress,
            console=console,
            refresh_per_second=10,
        ):
            async with asyncio.TaskGroup() as group:
                for bot in bots:
                    group.create_task(
                        run_bot_forever(
                            bot,
                            minimum_delay,
                            maximum_delay,
                        )
                    )
    finally:
        await asyncio.gather(
            *(bot.forum.close() for bot in bots),
            *(bot.seeborg.close() for bot in bots),
            return_exceptions=True,
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        console.print("Shutting down.")
