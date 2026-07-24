from __future__ import annotations

import re
from pathlib import Path


CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
REPEATED_CHAR_RE = re.compile(r"(.)\1{20,}", re.DOTALL)
REPEATED_WORD_RE = re.compile(
    r"\b(\w+)(?:\s+\1){10,}\b",
    re.IGNORECASE,
)


class ContentRejected(ValueError):
    pass


def load_blocklist(path: str | Path | None) -> list[str]:
    """Load one blocked term per line; blank lines and # comments ignored.

    The source dictionary contains slurs and other toxic language. Populate
    dictionaries/blocklist.txt with the terms you never want a bot to emit;
    the same list is applied when building dictionaries and when validating
    generated posts.
    """
    if path is None:
        return []

    file = Path(path)
    if not file.exists():
        return []

    terms: list[str] = []
    for line in file.read_text(encoding="utf-8").splitlines():
        term = line.strip().casefold()
        if term and not term.startswith("#"):
            terms.append(term)
    return terms


def contains_blocked_term(text: str, blocklist: list[str]) -> str | None:
    if not blocklist:
        return None

    folded = text.casefold()
    for term in blocklist:
        # Substring match by default; a term wrapped in slashes (/word/)
        # is matched on word boundaries instead, for short terms that
        # would otherwise hit inside innocent words.
        if term.startswith("/") and term.endswith("/") and len(term) > 2:
            if re.search(rf"\b{re.escape(term[1:-1])}\b", folded):
                return term
        elif term in folded:
            return term
    return None


def validate_post(
    body: str,
    *,
    maximum_length: int = 4000,
    blocklist: list[str] | None = None,
) -> str:
    body = CONTROL_RE.sub("", body).strip()

    if not body:
        raise ContentRejected("Generated body is empty")

    if len(body) > maximum_length:
        raise ContentRejected(
            f"Generated body exceeds {maximum_length} characters"
        )

    if REPEATED_CHAR_RE.search(body):
        raise ContentRejected("Excessive repeated characters")

    if REPEATED_WORD_RE.search(body):
        raise ContentRejected("Excessive repeated words")

    blocked = contains_blocked_term(body, blocklist or [])
    if blocked:
        raise ContentRejected("Generated body contains a blocklisted term")

    return body
