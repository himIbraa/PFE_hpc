"""Compare AlgerianLegalBench baselines + the latest RLM run.

Reads ``metrics.json`` from every ``eval_results/baseline_*`` (and the latest
RLM run), and renders three markdown tables:

* a one-row-per-pipeline overview of which runs were included,
* the **headline Cite F1 table** (rows = query type, cols = pipelines —
  this is the table that goes in thesis Chapter 5),
* an **Overall metrics** table (one row per pipeline, columns =
  MRR@10 doc/art, Cite F1, Doc Cite F1, R@10 art, HCR, JIR, Abst F1),
* a **Per query type** block with the same columns for each of the eight
  AlgerianLegalBench v3.0 query types.

The rendered markdown is saved to
``eval_results/comparison_<timestamp>.md`` and (unless ``--no-stdout``)
printed to the terminal.

The script reads from BOTH ``D:\\TRY_AGAIN\\eval_results\\`` and
``D:\\TRY_AGAIN\\akn_rlm\\eval_results\\``. Some baseline runs landed at the
repo root because the runner was invoked from one level above the package
(e.g. ``baseline_kg_hybrid_smoke``); when the same ``run_id`` exists in both
trees, the freshest-mtime wins.

Discovery rules:

* A run is a directory under one of the eval roots that contains a
  ``metrics.json``.
* The ``run_id`` is the directory name. ``baseline_<name>...`` runs map to
  one of the six Phase-1 baselines; everything else is treated as RLM.
* A run is a "smoke" run if its ``run_id`` contains ``_smoke`` /
  ``_strat`` / ``_partial``; otherwise it's a "full" run.

Selection rules:

* By default, the freshest run per baseline (across smoke + full) is
  selected, plus the freshest RLM run.
* ``--include-smoke`` restricts the candidate pool to smoke runs.
* ``--include-full`` restricts the candidate pool to full runs.
* ``--runs <glob1>,<glob2>`` accepts an explicit list of fnmatch-style
  globs; only ``run_id``s matching at least one pattern are included
  (and *all* matching runs are shown — the freshest-per-baseline
  reduction is skipped).
* ``--all`` shows every discovered run.

Usage:

    $py = "C:\\Users\\21355\\.conda\\envs\\pfe_env\\python.exe"

    # Default — freshest run per baseline + RLM, smoke or full
    & $py D:\\TRY_AGAIN\\akn_rlm\\scripts\\compare_baselines.py

    # Only smoke runs
    & $py D:\\TRY_AGAIN\\akn_rlm\\scripts\\compare_baselines.py --include-smoke

    # Only full 244-q runs
    & $py D:\\TRY_AGAIN\\akn_rlm\\scripts\\compare_baselines.py --include-full

    # Pin specific runs by glob
    & $py D:\\TRY_AGAIN\\akn_rlm\\scripts\\compare_baselines.py \\
        --runs "baseline_bm25_smoke,baseline_kg_hybrid_strat5"

    # Custom output filename
    & $py D:\\TRY_AGAIN\\akn_rlm\\scripts\\compare_baselines.py \\
        --out D:\\TRY_AGAIN\\eval_results\\comparison_phase1.md
"""
from __future__ import annotations

import argparse
import fnmatch
import io
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

log = logging.getLogger("compare_baselines")


def _configure_console() -> None:
    """Wrap stdout in UTF-8 and configure logging.

    Done lazily inside ``main`` (rather than at import time) so importing
    this module under pytest doesn't replace the test runner's captured
    ``sys.stdout``.
    """
    if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding.lower() != "utf-8":
        try:
            sys.stdout = io.TextIOWrapper(
                sys.stdout.buffer, encoding="utf-8", errors="replace"
            )
        except (AttributeError, ValueError):
            # No .buffer (already wrapped) or detached — nothing to do.
            pass
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Order matters: longest prefix first so `baseline_hybrid_rerank` is matched
# before `baseline_hybrid`, and `baseline_kg_hybrid` before `baseline_kg`.
BASELINE_PREFIXES: list[tuple[str, str, str, int]] = [
    # (run_id_prefix, baseline_id, display label, sort order)
    ("baseline_hybrid_rerank", "hybrid_rerank", "Hybrid+Rerank", 4),
    ("baseline_kg_hybrid",     "kg_hybrid",     "KG+Hybrid",     6),
    ("baseline_bm25",          "bm25",          "BM25",          1),
    ("baseline_dense",         "dense",         "Dense",         2),
    ("baseline_hybrid",        "hybrid",        "Hybrid (RRF)",  3),
    ("baseline_kg",            "kg",            "KG (SPARQL)",   5),
]
RLM_BASELINE_ID = "rlm"
RLM_LABEL = "RLM"
RLM_SORT_ORDER = 100  # last column

QUERY_TYPES: list[str] = [
    "exact_article",
    "rule_application",
    "multi_hop",
    "temporal_factual",
    "conceptual_definitional",
    "unanswerable",
    "layman",
    "long_context",
]

METRICS: list[tuple[str, str]] = [
    # (metrics.json key, display label)
    ("mrr_doc",          "MRR@10 doc"),
    ("mrr_article",      "MRR@10 art"),
    ("citation_f1",      "Cite F1"),
    ("doc_citation_f1",  "Doc Cite F1"),
    ("recall_article",   "R@10 art"),
    ("hcr",              "HCR↓"),
    ("jir",              "JIR↓"),
    ("abstention_f1",    "Abst F1"),
]

_SMOKE_TOKENS = ("_smoke", "_strat", "_partial")
_SMOKE_PREFIXES = ("smoke_", "phase0_", "strat_")

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_DIR.parent           # D:\TRY_AGAIN\akn_rlm
REPO_ROOT = PACKAGE_ROOT.parent            # D:\TRY_AGAIN
DEFAULT_ROOTS: list[Path] = [
    PACKAGE_ROOT / "eval_results",
    REPO_ROOT / "eval_results",
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class RunInfo:
    run_id: str
    path: Path
    metrics: dict
    baseline_id: str
    label: str
    sort_order: int
    is_smoke: bool
    n_total: int
    mtime: float = field(compare=False)


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

def classify_run_id(run_id: str) -> tuple[str, str, int]:
    """Return ``(baseline_id, display_label, sort_order)`` for a ``run_id``.

    Run-ids that do not start with any of the six known baseline prefixes
    are classified as RLM runs (the user's RLM benchmark uses ``run_*``
    or ``smoke_*`` / ``phase0_*`` ids).
    """
    for prefix, baseline_id, label, order in BASELINE_PREFIXES:
        if run_id == prefix or run_id.startswith(prefix + "_"):
            return baseline_id, label, order
    return RLM_BASELINE_ID, RLM_LABEL, RLM_SORT_ORDER


def is_smoke_run(run_id: str) -> bool:
    """True if the run id is a sub-244-q smoke / stratified / partial run.

    Catches both the new ``baseline_*_smoke`` / ``_strat5`` / ``_partial``
    convention and the legacy RLM run naming (``smoke_05``, ``phase0_smoke2``,
    ``strat_v1``).
    """
    rid = run_id.lower()
    if any(tok in rid for tok in _SMOKE_TOKENS):
        return True
    if any(rid.startswith(p) for p in _SMOKE_PREFIXES):
        return True
    return False


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _load_metrics(metrics_path: Path) -> dict | None:
    try:
        with metrics_path.open(encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Skipping %s: %s", metrics_path, exc)
        return None


def discover_runs(
    roots: Iterable[Path],
    *,
    include_smoke: bool = True,
    include_full: bool = True,
    runs_filter: list[str] | None = None,
) -> list[RunInfo]:
    """Walk every ``eval_results`` root and return one ``RunInfo`` per run.

    When the same ``run_id`` exists in more than one root, the entry with
    the largest ``mtime`` wins.
    """
    seen: dict[str, RunInfo] = {}
    for root in roots:
        root = Path(root)
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            mpath = child / "metrics.json"
            if not mpath.is_file():
                continue
            metrics = _load_metrics(mpath)
            if metrics is None:
                continue

            run_id = child.name
            baseline_id, label, order = classify_run_id(run_id)
            smoke = is_smoke_run(run_id)

            if not include_smoke and smoke:
                continue
            if not include_full and not smoke:
                continue
            if runs_filter:
                if not any(fnmatch.fnmatchcase(run_id, pat) for pat in runs_filter):
                    continue

            counts = metrics.get("counts") or {}
            try:
                n_total = int(counts.get("total", 0) or 0)
            except (TypeError, ValueError):
                n_total = 0
            mtime = mpath.stat().st_mtime

            info = RunInfo(
                run_id=run_id,
                path=child,
                metrics=metrics,
                baseline_id=baseline_id,
                label=label,
                sort_order=order,
                is_smoke=smoke,
                n_total=n_total,
                mtime=mtime,
            )
            existing = seen.get(run_id)
            if existing is None or info.mtime > existing.mtime:
                seen[run_id] = info
    return sorted(seen.values(), key=lambda r: (r.sort_order, r.run_id))


def select_freshest_per_baseline(runs: list[RunInfo]) -> list[RunInfo]:
    """Keep only the largest-``mtime`` run per ``baseline_id``."""
    by_baseline: dict[str, RunInfo] = {}
    for r in runs:
        cur = by_baseline.get(r.baseline_id)
        if cur is None or r.mtime > cur.mtime:
            by_baseline[r.baseline_id] = r
    return sorted(by_baseline.values(), key=lambda r: (r.sort_order, r.run_id))


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _fmt(val: object) -> str:
    """Format a metric value for the table; missing or NaN → em-dash."""
    if val is None:
        return "—"
    try:
        v = float(val)
    except (TypeError, ValueError):
        return "—"
    if v != v:  # NaN
        return "—"
    return f"{v:.3f}"


def _row_metrics(stratum: dict | None) -> list[str]:
    if not stratum:
        return ["—"] * len(METRICS)
    return [_fmt(stratum.get(key)) for key, _ in METRICS]


def _md_table(
    header: list[str],
    rows: list[list[str]],
    align: list[str] | None = None,
) -> str:
    if align is None:
        align = ["left"] + ["right"] * (len(header) - 1)
    sep_cells = []
    for a in align:
        if a == "right":
            sep_cells.append("---:")
        elif a == "center":
            sep_cells.append(":---:")
        else:
            sep_cells.append("---")
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(sep_cells) + " |")
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def render_runs_overview(runs: list[RunInfo]) -> str:
    header = ["Pipeline", "run_id", "kind", "n", "Path"]
    rows: list[list[str]] = []
    for r in runs:
        rows.append([
            r.label,
            f"`{r.run_id}`",
            "smoke" if r.is_smoke else "full",
            str(r.n_total),
            f"`{r.path}`",
        ])
    return _md_table(header, rows, align=["left"] * 5)


def render_overall_table(runs: list[RunInfo]) -> str:
    header = ["Pipeline", "run_id", "n"] + [label for _, label in METRICS]
    rows: list[list[str]] = []
    for r in runs:
        overall = r.metrics.get("overall") or {}
        rows.append([
            r.label,
            f"`{r.run_id}`",
            str(r.n_total),
            *_row_metrics(overall),
        ])
    align = ["left", "left", "right"] + ["right"] * len(METRICS)
    return _md_table(header, rows, align=align)


def render_per_type_block(runs: list[RunInfo], qt: str) -> str:
    header = ["Pipeline", "n"] + [label for _, label in METRICS]
    rows: list[list[str]] = []
    for r in runs:
        by_qt = r.metrics.get("by_query_type") or {}
        stratum = by_qt.get(qt) or {}
        n = ((r.metrics.get("counts") or {})
             .get("by_query_type") or {}).get(qt, 0)
        try:
            n_int = int(n)
        except (TypeError, ValueError):
            n_int = 0
        rows.append([
            r.label,
            str(n_int),
            *_row_metrics(stratum),
        ])
    align = ["left", "right"] + ["right"] * len(METRICS)
    return _md_table(header, rows, align=align)


def render_headline_cite_f1(runs: list[RunInfo]) -> str:
    """Per-query-type x per-baseline Cite F1 table — the headline number."""
    header = ["Query type"] + [r.label for r in runs]
    rows: list[list[str]] = []
    for qt in QUERY_TYPES:
        row = [qt]
        for r in runs:
            stratum = (r.metrics.get("by_query_type") or {}).get(qt) or {}
            row.append(_fmt(stratum.get("citation_f1")))
        rows.append(row)
    overall_row = ["**overall**"]
    for r in runs:
        overall_row.append(_fmt((r.metrics.get("overall") or {}).get("citation_f1")))
    rows.append(overall_row)
    align = ["left"] + ["right"] * len(runs)
    return _md_table(header, rows, align=align)


def render_report(runs: list[RunInfo]) -> str:
    if not runs:
        return "_No runs discovered._"
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = [
        "# AlgerianLegalBench — Phase 1 Comparison",
        "",
        f"Generated: {timestamp}",
        "",
        "## Runs included",
        "",
        render_runs_overview(runs),
        "",
        "## Headline — Cite F1 by query type",
        "",
        "Article-level Citation F1 is the headline thesis metric. The RLM target",
        "is to beat the best baseline on the hard types (`multi_hop`,",
        "`temporal_factual`, `conceptual_definitional`, `unanswerable`).",
        "",
        render_headline_cite_f1(runs),
        "",
        "## Overall metrics",
        "",
        render_overall_table(runs),
        "",
        "## Per query type",
        "",
    ]
    for qt in QUERY_TYPES:
        any_data = any(
            (r.metrics.get("by_query_type") or {}).get(qt) for r in runs
        )
        if not any_data:
            continue
        lines += [f"### {qt}", "", render_per_type_block(runs, qt), ""]

    lines += [
        "",
        "_HCR↓ / JIR↓ — lower is better. Deterministic baselines (B1–B6)",
        "have HCR=JIR=0 by construction (no LLM in the loop)._",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare AlgerianLegalBench baselines + RLM run as a markdown table",
    )
    parser.add_argument(
        "--eval-roots", nargs="*", default=None,
        help="One or more eval_results directories to scan. "
             f"Default: {[str(p) for p in DEFAULT_ROOTS]}",
    )
    parser.add_argument(
        "--include-smoke", action="store_true",
        help="Restrict to smoke runs (run_ids containing _smoke / _strat / _partial).",
    )
    parser.add_argument(
        "--include-full", action="store_true",
        help="Restrict to full 244-q runs (run_ids without smoke/strat/partial suffix).",
    )
    parser.add_argument(
        "--runs", default=None,
        help="Comma-separated fnmatch glob(s) restricting which run_ids to include "
             "(e.g. 'baseline_bm25_smoke,baseline_kg_hybrid_strat5'). When supplied, "
             "every matching run is shown (skips the freshest-per-baseline reduction).",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Show every discovered run instead of selecting one per baseline.",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output markdown file. "
             "Default: <repo>/eval_results/comparison_<timestamp>.md",
    )
    parser.add_argument(
        "--no-stdout", action="store_true",
        help="Skip printing the table to stdout (still writes the .md file).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _configure_console()
    args = _parse_args(argv)

    roots: list[Path]
    if args.eval_roots:
        roots = [Path(p) for p in args.eval_roots]
    else:
        roots = list(DEFAULT_ROOTS)

    if not args.include_smoke and not args.include_full:
        include_smoke = True
        include_full = True
    else:
        include_smoke = args.include_smoke
        include_full = args.include_full

    runs_filter = None
    if args.runs:
        runs_filter = [s.strip() for s in args.runs.split(",") if s.strip()]

    discovered = discover_runs(
        roots,
        include_smoke=include_smoke,
        include_full=include_full,
        runs_filter=runs_filter,
    )

    if not discovered:
        log.error("No runs matched. Roots scanned: %s", [str(r) for r in roots])
        return 1

    if args.all or runs_filter:
        selected = sorted(discovered, key=lambda r: (r.sort_order, r.run_id))
    else:
        selected = select_freshest_per_baseline(discovered)

    log.info("Comparing %d run(s):", len(selected))
    for r in selected:
        kind = "smoke" if r.is_smoke else "full"
        log.info("  %-15s  %-40s  (%s, n=%d)", r.label, r.run_id, kind, r.n_total)

    md = render_report(selected)

    if args.out:
        out_path = Path(args.out)
    else:
        out_dir = REPO_ROOT / "eval_results"
        out_path = out_dir / time.strftime("comparison_%Y%m%d_%H%M%S.md")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    log.info("Comparison written to %s", out_path)

    if not args.no_stdout:
        print(md)

    return 0


if __name__ == "__main__":
    sys.exit(main())
