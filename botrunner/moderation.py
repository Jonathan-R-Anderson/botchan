from __future__ import annotations

import re
from pathlib import Path


CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

# Deliberately loose: "AAAAAAAAAAAAAAAAAAAAAAAAAAAA" and drawn-out vowels are
# native to the register. These thresholds only exist to catch degenerate
# markov spam, not to police shouting.
REPEATED_CHAR_RE = re.compile(r"(.)\1{60,}", re.DOTALL)
REPEATED_WORD_RE = re.compile(
    r"\b(\w+)(?:\s+\1){15,}\b",
    re.IGNORECASE,
)

# Evasion normalization for blocklist matching. A markov chain trained on IRC
# will happily emit spaced-out, leetspeak, and letter-stretched spellings of
# terms that the literal blocklist would miss.
LEET_MAP = str.maketrans(
    {
        "0": "o", "1": "i", "3": "e", "4": "a", "5": "s",
        "6": "g", "7": "t", "8": "b", "9": "g",
        "@": "a", "$": "s", "!": "i", "|": "i", "+": "t",
    }
)
NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
RUN_RE = re.compile(r"(.)\1+")
LONG_RUN_RE = re.compile(r"(.)\1{2,}")

# Three or more single characters in a row separated by whitespace is a
# spelling-out evasion ("j a p s") and effectively never ordinary prose.
SPACED_LETTERS_RE = re.compile(r"\b(?:[a-z0-9][^\S\n]+){2,}[a-z0-9]\b")
WHITESPACE_RE = re.compile(r"\s+")

# Below this length a space-stripped term is too promiscuous to match across
# word joins: "heeb" collapses to "heb", which appears inside "...t-he b-ar...".
MINIMUM_NORMALIZED_LENGTH = 5


def join_spaced_letters(text: str) -> str:
    return SPACED_LETTERS_RE.sub(
        lambda match: WHITESPACE_RE.sub("", match.group(0)),
        text,
    )


def normalize_for_blocklist(text: str) -> str:
    """Collapse spacing, punctuation, leetspeak, and stretched letters.

    "N-i-g-g-3-r" and "niiiiggggerrr" both fold onto the same normalized
    form as the plain spelling, so one blocklist entry covers the variants.
    Word boundaries do not survive this, so it is only used for substring
    terms; see normalize_words for the boundary-anchored path.
    """
    folded = text.casefold().translate(LEET_MAP)
    folded = NON_ALNUM_RE.sub("", folded)
    return RUN_RE.sub(r"\1", folded)


def normalize_words(text: str) -> str:
    """Fold leetspeak and stretched letters while keeping word boundaries.

    Runs collapse to two characters rather than one, so "c00oooon" folds
    onto "coon" without "coon" itself folding onto "con" and firing on
    every innocent use of that word.
    """
    folded = text.casefold().translate(LEET_MAP)
    folded = NON_ALNUM_RE.sub(" ", folded)
    return LONG_RUN_RE.sub(r"\1\1", folded)


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

    folded = join_spaced_letters(text.casefold())
    normalized = normalize_for_blocklist(folded)
    worded = normalize_words(folded)

    for term in blocklist:
        # A term wrapped in slashes (/word/) matches on word boundaries only,
        # for short terms that would otherwise hit inside innocent words.
        # It still gets leetspeak folding, via the boundary-preserving
        # normalizer, so "sp1c" does not walk straight through.
        if term.startswith("/") and term.endswith("/") and len(term) > 2:
            bare = term[1:-1]
            if re.search(rf"\b{re.escape(bare)}\b", folded):
                return term
            normalized_bare = normalize_words(bare).strip()
            if normalized_bare and re.search(
                rf"\b{re.escape(normalized_bare)}\b", worded
            ):
                return term
            continue

        if term in folded:
            return term

        normalized_term = normalize_for_blocklist(term)
        if (
            len(normalized_term) >= MINIMUM_NORMALIZED_LENGTH
            and normalized_term in normalized
        ):
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
