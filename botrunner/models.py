from __future__ import annotations

from typing import Any
from pydantic import BaseModel, Field


class Board(BaseModel):
    board: str
    title: str = ""
    description: str = ""
    url: str = ""


class ThreadListing(BaseModel):
    id: int | str
    subject: str | None = None
    body: str = ""
    num_replies: int = 0
    num_media: int = 0
    last_updated: str | None = None
    url: str = ""


class ForumPost(BaseModel):
    post_id: int | str = Field(validation_alias="id")
    body: str
    is_op: bool = False
    subject: str | None = None
    poster: str | None = None
    poster_id: str | None = None
    created: str | None = None


class ThreadSummary(BaseModel):
    thread_id: int | str = Field(validation_alias="thread")
    board: str
    posts: list[ForumPost] = Field(default_factory=list)

    @property
    def subject(self) -> str | None:
        # The API carries the subject on the OP post, not the thread.
        for post in self.posts:
            if post.is_op:
                return post.subject
        return self.posts[0].subject if self.posts else None


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
