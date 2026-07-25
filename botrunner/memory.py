from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

import aiosqlite


WORD_RE = re.compile(r"[a-zA-Z0-9_']+")


def tokenize(text: str) -> Counter[str]:
    return Counter(word.lower() for word in WORD_RE.findall(text))


def lexical_similarity(left: str, right: str) -> float:
    a = tokenize(left)
    b = tokenize(right)

    if not a or not b:
        return 0.0

    numerator = sum(a[word] * b[word] for word in a.keys() & b.keys())
    denominator = math.sqrt(
        sum(value * value for value in a.values())
        * sum(value * value for value in b.values())
    )

    return numerator / denominator if denominator else 0.0


class BotMemory:
    def __init__(self, database_path: str | Path, bot_id: str) -> None:
        self.database_path = str(database_path)
        self.bot_id = bot_id

    async def initialize(self) -> None:
        Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)

        async with aiosqlite.connect(self.database_path) as db:
            await db.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bot_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    importance REAL NOT NULL DEFAULT 0.5,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_memories_bot
                    ON memories(bot_id, created_at DESC);

                CREATE TABLE IF NOT EXISTS posted_content (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    bot_id TEXT NOT NULL,
                    board TEXT NOT NULL,
                    thread_id TEXT,
                    body TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE INDEX IF NOT EXISTS idx_posted_bot
                    ON posted_content(bot_id, created_at DESC);
                """
            )
            await db.commit()

    async def remember(
        self,
        kind: str,
        content: str,
        *,
        metadata: dict | None = None,
        importance: float = 0.5,
    ) -> None:
        async with aiosqlite.connect(self.database_path) as db:
            await db.execute(
                """
                INSERT INTO memories (
                    bot_id, kind, content, metadata, importance
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    self.bot_id,
                    kind,
                    content,
                    json.dumps(metadata or {}, default=str),
                    max(0.0, min(1.0, importance)),
                ),
            )
            await db.commit()

    async def retrieve(self, query: str, limit: int = 12) -> list[dict]:
        async with aiosqlite.connect(self.database_path) as db:
            cursor = await db.execute(
                """
                SELECT kind, content, metadata, importance, created_at
                FROM memories
                WHERE bot_id = ?
                ORDER BY created_at DESC
                LIMIT 300
                """,
                (self.bot_id,),
            )
            rows = await cursor.fetchall()

        ranked = []

        for kind, content, metadata, importance, created_at in rows:
            similarity = lexical_similarity(query, content)
            score = similarity * 0.75 + float(importance) * 0.25

            ranked.append(
                {
                    "kind": kind,
                    "content": content,
                    "metadata": json.loads(metadata),
                    "importance": importance,
                    "created_at": created_at,
                    "score": score,
                }
            )

        ranked.sort(key=lambda item: item["score"], reverse=True)
        return ranked[:limit]

    async def record_post(
        self,
        board: str,
        thread_id: int | str | None,
        body: str,
    ) -> None:
        async with aiosqlite.connect(self.database_path) as db:
            await db.execute(
                """
                INSERT INTO posted_content (
                    bot_id, board, thread_id, body
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    self.bot_id,
                    board,
                    str(thread_id) if thread_id is not None else None,
                    body,
                ),
            )
            await db.commit()

    async def recent_own_posts(self, limit: int = 8) -> list[str]:
        async with aiosqlite.connect(self.database_path) as db:
            cursor = await db.execute(
                """
                SELECT body
                FROM posted_content
                WHERE bot_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (self.bot_id, limit),
            )
            rows = await cursor.fetchall()

        return [row[0] for row in rows]

    async def is_duplicate(
        self,
        candidate: str,
        *,
        threshold: float = 0.88,
    ) -> bool:
        async with aiosqlite.connect(self.database_path) as db:
            cursor = await db.execute(
                """
                SELECT body
                FROM posted_content
                WHERE bot_id = ?
                ORDER BY created_at DESC
                LIMIT 100
                """,
                (self.bot_id,),
            )
            rows = await cursor.fetchall()

        return any(
            lexical_similarity(candidate, row[0]) >= threshold
            for row in rows
        )
