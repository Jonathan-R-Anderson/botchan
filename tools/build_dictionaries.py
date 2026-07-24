#!/usr/bin/env python3
"""Build a cleaned per-bot Seeborg dictionary from a raw IRC log dump.

The raw db.txt keeps IRC formatting control bytes, service commands, and
toxic language. This tool strips formatting, drops command lines and
blocklisted lines, deduplicates, and (optionally) samples the result down
to a size seeborg-linein can load quickly.

Usage:
    python tools/build_dictionaries.py db.txt dictionaries/scruffy.txt \
        --sample 60000 --seed 1 --blocklist dictionaries/blocklist.txt
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from botrunner.moderation import contains_blocked_term, load_blocklist
from botrunner.seeborg import IRC_COMMAND_RE, clean_dictionary_line


def build(
    source: Path,
    output: Path,
    *,
    sample: int | None,
    seed: int,
    blocklist_path: Path | None,
    keep_urls: bool,
) -> None:
    blocklist = load_blocklist(blocklist_path)

    kept: list[str] = []
    seen: set[str] = set()
    total = dropped_blocked = dropped_command = 0

    with source.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            total += 1
            line = clean_dictionary_line(raw_line, strip_urls=not keep_urls)

            if len(line) < 2 or len(line) > 300:
                continue

            if IRC_COMMAND_RE.match(line):
                dropped_command += 1
                continue

            if contains_blocked_term(line, blocklist):
                dropped_blocked += 1
                continue

            normalized = line.casefold()
            if normalized in seen:
                continue

            seen.add(normalized)
            kept.append(line)

    if sample is not None and sample < len(kept):
        rng = random.Random(seed)
        kept = rng.sample(kept, k=sample)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(kept) + "\n", encoding="utf-8")

    print(
        f"{source} -> {output}: read {total} lines, "
        f"wrote {len(kept)} "
        f"(dropped {dropped_command} command lines, "
        f"{dropped_blocked} blocklisted lines)"
    )

    if not blocklist:
        print(
            "warning: blocklist is empty; the output still contains "
            "whatever toxic language the source had"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="randomly sample the cleaned corpus down to N lines",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed so each bot gets a different but reproducible subset",
    )
    parser.add_argument(
        "--blocklist",
        type=Path,
        default=Path("dictionaries/blocklist.txt"),
    )
    parser.add_argument(
        "--keep-urls",
        action="store_true",
        help="keep URLs in lines instead of stripping them",
    )

    arguments = parser.parse_args()

    build(
        arguments.source,
        arguments.output,
        sample=arguments.sample,
        seed=arguments.seed,
        blocklist_path=arguments.blocklist,
        keep_urls=arguments.keep_urls,
    )


if __name__ == "__main__":
    main()
