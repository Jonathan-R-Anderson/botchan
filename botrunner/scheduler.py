from __future__ import annotations

import asyncio
import random

from .api_client import ForumAPIError
from .bot import ForumBot


async def run_bot_forever(
    bot: ForumBot,
    minimum_delay: int,
    maximum_delay: int,
) -> None:
    while True:
        extra_delay = 0.0

        try:
            await bot.run_once()
        except asyncio.CancelledError:
            raise
        except ForumAPIError as exc:
            extra_delay = exc.retry_after or 0.0
            await bot.progress(
                "error",
                100,
                f"Forum error: {exc}",
            )
        except Exception as exc:
            await bot.progress(
                "error",
                100,
                f"Error: {type(exc).__name__}: {exc}",
            )

        delay = random.randint(minimum_delay, maximum_delay) + int(extra_delay)

        # Progress reflects real elapsed cooldown time, not fake work.
        steps = min(delay, 100)

        for step in range(steps):
            await asyncio.sleep(delay / steps)

            percent = int((step + 1) / steps * 100)
            await bot.progress(
                "cooldown",
                percent,
                f"Cooldown before next participation cycle: {delay}s",
            )
