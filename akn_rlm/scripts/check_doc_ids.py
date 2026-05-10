"""
Check every doc_id referenced in AlgerianLegalBench v3.0 against the registry.

Reports:
  RESOLVED   — alias maps to a canonical id that exists in the corpus
  COLLISION  — maps to None (anti-corruption / known collision)
  MISSING    — maps to None and is in MISSING_FROM_CORPUS
  UNKNOWN    — no alias entry at all + not a direct canonical id

Usage:
    python scripts/check_doc_ids.py
"""
from __future__ import annotations

import io
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from akn_rlm.config import get_benchmark_path
from akn_rlm.corpus.akn_parser import parse_all
from akn_rlm.corpus.article_registry import (
    ArticleRegistry,
    MISSING_FROM_CORPUS,
    FILENAME_COLLISIONS,
)


def main() -> None:
    benchmark_path = get_benchmark_path()
    print(f"Benchmark : {benchmark_path}")

    with benchmark_path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    questions = data.get("questions", data) if isinstance(data, dict) else data
    print(f"Questions : {len(questions)}\n")

    # ── Collect every raw doc_id from benchmark ──────────────────────────────
    # Maps raw_id → list of question IDs that reference it
    raw_doc_ids: dict[str, list[str]] = defaultdict(list)

    for q in questions:
        qid = str(q.get("id", "?"))
        for d in q.get("expected_documents", []):
            raw_doc_ids[d].append(qid)
        for art in q.get("expected_articles", []):
            d = art.get("document_id", "")
            if d:
                raw_doc_ids[d].append(qid)

    print(f"Unique raw doc_ids in benchmark : {len(raw_doc_ids)}\n")

    # ── Build registry ───────────────────────────────────────────────────────
    print("Parsing corpus …")
    registry = ArticleRegistry()
    registry.build(parse_all())
    canonical_ids = registry.canonical_ids
    print(f"Canonical docs in corpus        : {len(canonical_ids)}\n")

    _missing_set  = {m.lower() for m in MISSING_FROM_CORPUS}
    _collision_set = {k.lower() for k in FILENAME_COLLISIONS}

    # ── Classify each raw_id ─────────────────────────────────────────────────
    resolved:  dict[str, str]       = {}   # raw → canonical
    collision: dict[str, None]      = {}   # raw → None (known collision)
    missing:   dict[str, None]      = {}   # raw → None (missing from corpus)
    unknown:   dict[str, list[str]] = {}   # raw → question ids (no mapping at all)

    for raw_id, qids in sorted(raw_doc_ids.items()):
        key = raw_id.lower().strip()

        canonical = registry.resolve_alias(raw_id)

        if canonical is not None:
            resolved[raw_id] = canonical
        elif key in _collision_set or key in {"06-01_2006-02-20", "06-01"}:
            collision[raw_id] = None
        elif key in _missing_set or any(raw_id.lower() == m.lower() for m in MISSING_FROM_CORPUS):
            missing[raw_id] = None
        else:
            # resolve_alias returned None but it's not in the known sets
            # — could be in _STATIC_ALIASES mapped to None (also collision/missing)
            from akn_rlm.corpus.article_registry import _STATIC_ALIASES
            sa_key = raw_id.lower().strip()
            if sa_key in _STATIC_ALIASES:
                val = _STATIC_ALIASES[sa_key]
                if val is None:
                    missing[raw_id] = None  # explicitly mapped to None
                else:
                    resolved[raw_id] = val
            else:
                unknown[raw_id] = qids

    # ── Print report ─────────────────────────────────────────────────────────
    W = 38

    print("=" * 72)
    print(f"{'STATUS':<12} {'RAW DOC_ID':<{W}} {'RESOLVES TO'}")
    print("=" * 72)

    print(f"\n── RESOLVED ({len(resolved)}) ──────────────────────────────────────────────")
    for raw, canon in sorted(resolved.items()):
        same = "(identity)" if raw == canon else canon
        q_count = len(raw_doc_ids[raw])
        print(f"  {'OK':<10} {raw:<{W}} → {same}  [{q_count} q]")

    print(f"\n── COLLISION / UNAVAILABLE ({len(collision)}) ──────────────────────────────")
    for raw in sorted(collision):
        q_count = len(raw_doc_ids[raw])
        print(f"  {'COLLISION':<10} {raw:<{W}} → None (file exists but content mismatch)  [{q_count} q]")

    print(f"\n── MISSING FROM CORPUS ({len(missing)}) ──────────────────────────────────")
    for raw in sorted(missing):
        q_count = len(raw_doc_ids[raw])
        print(f"  {'MISSING':<10} {raw:<{W}} → None (no XML in corpus)  [{q_count} q]")

    print(f"\n── UNKNOWN / UNRESOLVED ({len(unknown)}) ────────────────────────────────")
    if unknown:
        for raw, qids in sorted(unknown.items()):
            print(f"  {'UNKNOWN':<10} {raw:<{W}} ← questions: {qids[:5]}")
    else:
        print("  (none — all benchmark doc_ids are accounted for)")

    print("\n" + "=" * 72)
    total = len(questions)
    # Count questions affected by each category
    def _affected(id_set):
        seen = set()
        for raw in id_set:
            for qid in raw_doc_ids[raw]:
                seen.add(qid)
        return len(seen)

    print(f"Summary:")
    print(f"  Resolved   : {len(resolved):>3} unique ids  ({_affected(resolved):>3} questions have at least one)")
    print(f"  Collision  : {len(collision):>3} unique ids  ({_affected(collision):>3} questions affected → must abstain)")
    print(f"  Missing    : {len(missing):>3} unique ids  ({_affected(missing):>3} questions affected → must abstain)")
    print(f"  Unknown    : {len(unknown):>3} unique ids  ({_affected(unknown):>3} questions affected → NEEDS FIX)")
    print("=" * 72)

    # ── Per-question abstain list ─────────────────────────────────────────────
    if collision or missing or unknown:
        bad_ids = set(collision) | set(missing) | set(unknown)
        affected_qs = {qid for raw in bad_ids for qid in raw_doc_ids[raw]}
        print(f"\nQuestions that reference an unresolvable doc_id ({len(affected_qs)} total):")
        for q in questions:
            qid = str(q.get("id", "?"))
            if qid not in affected_qs:
                continue
            bad_refs = [
                d for d in q.get("expected_documents", []) if d in bad_ids
            ] + [
                art.get("document_id", "") for art in q.get("expected_articles", [])
                if art.get("document_id", "") in bad_ids
            ]
            answerable = q.get("answerable", True)
            print(f"  Q{qid:<6} answerable={str(answerable):<5}  bad_refs={list(set(bad_refs))}")


if __name__ == "__main__":
    main()
