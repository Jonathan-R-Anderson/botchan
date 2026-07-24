from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class Board(BaseModel):
    board: str
    title: str = ""
    description: str = ""
    url: str = ""


class ForumPost(BaseModel):
    post_id: int | str
    thread_id: int | str
    board: str
    body: str
    author: str | None = None
    created_at: str | None = None


class ThreadSummary(BaseModel):
    thread_id: int | str
    board: str
    subject: str | None = None
    posts: list[ForumPost] = Field(default_factory=list)


class GeneratedPost(BaseModel):
    body: str
    subject: str | None = None
    meme_query: str | None = None
    reasoning_tags: list[str] = Field(default_factory=list)


class ProgressEvent(BaseModel):
    bot_id: str
    stage: str
    percent: int
    message: str
    details: dict[str, Any] = Field(default_factory=dict)
