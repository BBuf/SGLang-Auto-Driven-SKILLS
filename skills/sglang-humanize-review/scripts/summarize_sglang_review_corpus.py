#!/usr/bin/env python3
"""Exhaustive, segmented sweep + synthesis over the SGLang human review corpus.

Unlike query_sglang_review_corpus.py (which prints the first --limit matches in
file order and stops), this scans the WHOLE corpus in memory-bounded segments,
folds EVERY thread relevant to the PR under review into an aggregate, and keeps
only the requested top-ranked representative review opinions. Use it before
writing a review so the findings are grounded in all relevant historical review
behavior, not just the first few hits.

Example:
    python3 summarize_sglang_review_corpus.py \
        --path python/sglang/srt/speculative --path python/sglang/srt/managers \
        --query eagle --query "cuda graph" --query logprob \
        --top 30
"""

from __future__ import annotations

import argparse
import gzip
import heapq
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

# Reuse the matching/text helpers from the first-N query tool so the two stay
# consistent in how a thread is read and rendered.
from query_sglang_review_corpus import (  # noqa: E402
    DEFAULT_CORPUS,
    clip,
    row_type_for_thread,
    thread_text,
)

TopMatch = tuple[float, int, list[str], dict[str, Any]]


@dataclass
class SweepResult:
    total_scanned: int = 0
    matched_count: int = 0
    top_matches: list[TopMatch] = field(default_factory=list)
    cat_counts: Counter[str] = field(default_factory=Counter)
    path_counts: Counter[str] = field(default_factory=Counter)
    reviewer_counts: Counter[str] = field(default_factory=Counter)
    type_counts: Counter[str] = field(default_factory=Counter)
    year_counts: Counter[str] = field(default_factory=Counter)
    matched_comments: int = 0
    matched_prs: set[Any] = field(default_factory=set)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exhaustively sweep and summarize the SGLang review corpus."
    )
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        help="Path substring (repeatable, OR). A thread matches if its path "
        "contains any of these.",
    )
    parser.add_argument(
        "--query",
        action="append",
        default=[],
        help="Keyword/phrase (repeatable, OR). Case-insensitive substring over "
        "the whole thread text. Distinct terms hit drive the relevance score.",
    )
    parser.add_argument(
        "--category",
        action="append",
        default=[],
        help="Category label (repeatable, OR).",
    )
    parser.add_argument("--kind", default="", help="Thread/comment kind filter.")
    parser.add_argument("--reviewer", default="", help="Reviewer login filter.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2000,
        help="Rows loaded into memory per segment before they are folded into "
        "the running aggregate. Bounds peak memory on the 128MB corpus.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="How many highest-relevance threads to render in full.",
    )
    parser.add_argument(
        "--include-agent-reviewers",
        action="store_true",
        help="Include threads whose only comments are bot/coding-agent comments.",
    )
    parser.add_argument(
        "--excerpt-chars",
        type=int,
        default=320,
        help="Per-comment body clip width in the rendered excerpts.",
    )
    parser.add_argument("--format", choices=("digest", "jsonl"), default="digest")
    return parser.parse_args()


def iter_segments(path: Path, batch_size: int) -> Iterable[list[dict[str, Any]]]:
    """Yield the corpus in memory-bounded segments of parsed rows."""
    opener = gzip.open if path.suffix == ".gz" else open
    batch: list[dict[str, Any]] = []
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            batch.append(json.loads(line))
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


def score_thread(
    thread: dict[str, Any],
    args: argparse.Namespace,
    text_lower: str,
) -> tuple[bool, float, list[str]]:
    """Return (matched, relevance_score, hit_terms) for one thread."""
    # Hard filters (AND): a thread that fails any of these is not a match.
    if (
        not args.include_agent_reviewers
        and thread.get("human_reviewer_comment_count", 0) == 0
    ):
        return False, 0.0, []
    if args.kind:
        kind = args.kind.lower()
        comment_kinds = {
            str(c.get("kind", "")).lower() for c in thread.get("comments", [])
        }
        if kind != row_type_for_thread(thread).lower() and kind not in comment_kinds:
            return False, 0.0, []
    if args.reviewer:
        reviewer = args.reviewer.lower()
        if not any(
            reviewer == str(c.get("author", {}).get("login", "")).lower()
            for c in thread.get("comments", [])
        ):
            return False, 0.0, []
    if args.category:
        cats = set(thread.get("categories", []))
        if not cats.intersection(args.category):
            return False, 0.0, []

    path = str(thread.get("path", "")).lower()
    path_hit = any(p.lower() in path for p in args.path) if args.path else False
    if args.path and not path_hit:
        return False, 0.0, []

    hit_terms: list[str] = []
    occurrences = 0
    if args.query:
        for q in args.query:
            c = text_lower.count(q.lower())
            if c:
                hit_terms.append(q)
                occurrences += c
        if not hit_terms:
            return False, 0.0, []

    # Relevance score: distinct query terms dominate, then path/subsystem match,
    # then volume of discussion and recency.
    score = 0.0
    score += 3.0 * len(hit_terms)
    score += min(occurrences, 8) * 0.5
    score += 2.0 if path_hit else 0.0
    score += 1.0 if thread.get("categories") else 0.0
    score += min(thread.get("human_reviewer_comment_count", 0), 5) * 0.4
    try:
        year = int(str(thread["pull_request"].get("created_at", ""))[:4])
        score += (year - 2024) * 0.3  # gentle recency tilt
    except (KeyError, ValueError):
        pass
    return True, score, hit_terms


def render_thread(thread: dict[str, Any], hit_terms: list[str], width: int) -> str:
    pr = thread.get("pull_request", {})
    comments = [
        c
        for c in thread.get("comments", [])
        if not c.get("author", {}).get("is_agent", False)
    ]
    url = comments[0].get("html_url") if comments else ""
    lines = [
        f"### PR #{pr.get('number')}: {pr.get('title', '')}",
        f"- Type: `{row_type_for_thread(thread)}` | Path: "
        f"`{thread.get('path') or '<conversation>'}`",
        f"- Categories: `{', '.join(thread.get('categories', []))}` | "
        f"Hit terms: `{', '.join(hit_terms) or '-'}`",
        f"- Link: {url}",
    ]
    diff_hunk = thread.get("diff_hunk", "")
    if diff_hunk:
        lines += ["```diff", clip(diff_hunk, 400), "```"]
    for c in comments[:4]:
        who = c.get("author", {}).get("login", "unknown")
        lines.append(f"- **{who}**: {clip(c.get('body', ''), width)}")
    return "\n".join(lines)


def sweep_corpus(
    args: argparse.Namespace,
    emit_jsonl: Callable[[str], None] = print,
) -> SweepResult:
    """Scan all rows while retaining at most ``args.top`` full payloads."""
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")
    if args.top < 0:
        raise ValueError("--top must be non-negative")

    result = SweepResult()
    serial = 0
    for segment in iter_segments(args.corpus, args.batch_size):
        for thread in segment:
            result.total_scanned += 1
            ok, score, hits = score_thread(thread, args, thread_text(thread).lower())
            if not ok:
                continue
            result.matched_count += 1
            if args.format == "jsonl":
                emit_jsonl(json.dumps(thread, ensure_ascii=False, sort_keys=True))
            elif args.top:
                candidate: TopMatch = (score, serial, hits, thread)
                if len(result.top_matches) < args.top:
                    heapq.heappush(result.top_matches, candidate)
                elif candidate[:2] > result.top_matches[0][:2]:
                    heapq.heapreplace(result.top_matches, candidate)
            serial += 1
            for cat in thread.get("categories", []):
                result.cat_counts[cat] += 1
            if thread.get("path"):
                result.path_counts[thread["path"]] += 1
            result.type_counts[row_type_for_thread(thread)] += 1
            pr = thread.get("pull_request", {})
            result.matched_prs.add(pr.get("number"))
            result.year_counts[str(pr.get("created_at", ""))[:4] or "?"] += 1
            for c in thread.get("comments", []):
                if not c.get("author", {}).get("is_agent", False):
                    result.matched_comments += 1
                    login = c.get("author", {}).get("login")
                    if login:
                        result.reviewer_counts[login] += 1
    return result


def main() -> int:
    args = parse_args()
    try:
        result = sweep_corpus(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.format == "jsonl":
        return 0

    matched = sorted(result.top_matches, key=lambda item: item[:2], reverse=True)
    print("# SGLang review corpus sweep")
    print()
    print(
        f"- Filters: paths={args.path or '-'} queries={args.query or '-'} "
        f"category={args.category or '-'} kind={args.kind or '-'}"
    )
    print(
        f"- Scanned **{result.total_scanned}** threads in segments of "
        f"{args.batch_size}; matched **{result.matched_count}** threads across "
        f"**{len([p for p in result.matched_prs if p is not None])}** PRs, "
        f"**{result.matched_comments}** human comments."
    )
    print()
    if not result.matched_count:
        print("No relevant historical review threads found. Widen --query/--path.")
        return 0

    def top(counter: Counter[str], n: int = 12) -> str:
        return ", ".join(f"{k} ({v})" for k, v in counter.most_common(n)) or "-"

    print("## Aggregate over ALL matches")
    print(f"- By type: {top(result.type_counts)}")
    print(f"- By category: {top(result.cat_counts)}")
    print(f"- Top paths: {top(result.path_counts)}")
    print(f"- Top reviewers: {top(result.reviewer_counts)}")
    print(f"- By PR year: {top(result.year_counts)}")
    print()
    shown = len(matched)
    print(f"## Top {shown} most relevant review threads (read and summarize these)")
    print()
    for score, _, hits, thread in matched:
        print(render_thread(thread, hits, args.excerpt_chars))
        print(f"- _relevance: {score:.1f}_")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
