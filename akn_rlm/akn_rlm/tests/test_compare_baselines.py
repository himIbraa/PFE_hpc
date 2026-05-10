"""Tests for ``scripts/compare_baselines.py``.

These tests build minimal ``metrics.json`` files in a tmp_path-based
``eval_results/`` tree and verify that:

* Run ids are correctly classified into baseline families and
  smoke-vs-full.
* Discovery walks every requested root, dedupes ``run_id``s across
  roots by largest mtime, and respects ``--include-smoke`` /
  ``--include-full`` / ``--runs`` filters.
* The freshest-per-baseline reduction picks the right run.
* The rendered markdown contains every required section (Runs
  included, Headline Cite F1, Overall metrics, Per query type) and a
  row per pipeline.
* Missing strata render as em-dashes, not crashes.
* ``main()`` writes the comparison file, prints to stdout, and exits
  non-zero with a sensible error when no runs match.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

# Make the `scripts` package importable when pytest is invoked from the
# repo root. parents[2] is D:\TRY_AGAIN\akn_rlm (which contains scripts/).
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts import compare_baselines as cb  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures: write a minimal but realistic metrics.json
# ---------------------------------------------------------------------------

def _full_stratum(
    *,
    mrr_doc: float = 0.5,
    mrr_article: float = 0.2,
    citation_f1: float = 0.1,
    doc_citation_f1: float = 0.3,
    recall_article: float = 0.15,
    hcr: float = 0.0,
    jir: float = 0.0,
    abstention_f1: float = 0.0,
) -> dict:
    return {
        "mrr_doc": mrr_doc,
        "mrr_article": mrr_article,
        "citation_f1": citation_f1,
        "doc_citation_f1": doc_citation_f1,
        "recall_article": recall_article,
        "hcr": hcr,
        "jir": jir,
        "abstention_f1": abstention_f1,
    }


def _write_run(
    root: Path,
    run_id: str,
    *,
    overall: dict | None = None,
    by_query_type: dict | None = None,
    counts_total: int = 16,
    counts_qt: dict | None = None,
) -> Path:
    metrics = {
        "overall": overall or _full_stratum(),
        "by_query_type": by_query_type or {
            qt: _full_stratum(citation_f1=0.05) for qt in cb.QUERY_TYPES
        },
        "counts": {
            "total": counts_total,
            "by_query_type": counts_qt or {qt: 2 for qt in cb.QUERY_TYPES},
        },
    }
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False), encoding="utf-8"
    )
    return run_dir


# ---------------------------------------------------------------------------
# classify_run_id / is_smoke_run
# ---------------------------------------------------------------------------

def test_classify_known_baselines():
    assert cb.classify_run_id("baseline_bm25_smoke")[0] == "bm25"
    assert cb.classify_run_id("baseline_dense")[0] == "dense"
    assert cb.classify_run_id("baseline_hybrid_smoke")[0] == "hybrid"
    # hybrid_rerank must be matched BEFORE hybrid (longest-prefix-first)
    assert cb.classify_run_id("baseline_hybrid_rerank_strat5")[0] == "hybrid_rerank"
    assert cb.classify_run_id("baseline_hybrid_rerank")[0] == "hybrid_rerank"
    # kg_hybrid must be matched BEFORE kg
    assert cb.classify_run_id("baseline_kg_hybrid_smoke")[0] == "kg_hybrid"
    assert cb.classify_run_id("baseline_kg_hybrid")[0] == "kg_hybrid"
    assert cb.classify_run_id("baseline_kg")[0] == "kg"
    assert cb.classify_run_id("baseline_kg_smoke")[0] == "kg"


def test_classify_rlm_fallback():
    assert cb.classify_run_id("phase0_smoke2")[0] == cb.RLM_BASELINE_ID
    assert cb.classify_run_id("run_final")[0] == cb.RLM_BASELINE_ID
    assert cb.classify_run_id("smoke_05")[0] == cb.RLM_BASELINE_ID
    assert cb.classify_run_id("anything_else")[0] == cb.RLM_BASELINE_ID


def test_classify_returns_label_and_order():
    bid, label, order = cb.classify_run_id("baseline_kg_hybrid_smoke")
    assert label == "KG+Hybrid"
    assert order == 6
    bid, label, order = cb.classify_run_id("phase0_smoke2")
    assert label == "RLM"
    assert order == cb.RLM_SORT_ORDER


def test_is_smoke_run():
    assert cb.is_smoke_run("baseline_bm25_smoke") is True
    assert cb.is_smoke_run("baseline_hybrid_strat5") is True
    assert cb.is_smoke_run("baseline_kg_hybrid_strat5") is True
    assert cb.is_smoke_run("baseline_dense_partial") is True
    # Plain baseline_<name> is a full 244-q run by convention.
    assert cb.is_smoke_run("baseline_bm25") is False
    assert cb.is_smoke_run("baseline_kg") is False
    assert cb.is_smoke_run("run_final") is False
    assert cb.is_smoke_run("phase0_smoke2") is True
    # Legacy RLM run-id prefixes (the project predates the new convention).
    assert cb.is_smoke_run("smoke_05") is True
    assert cb.is_smoke_run("phase0_smoke") is True
    assert cb.is_smoke_run("strat_v1") is True


# ---------------------------------------------------------------------------
# discover_runs
# ---------------------------------------------------------------------------

def test_discover_finds_metrics_json(tmp_path):
    root = tmp_path / "eval_results"
    root.mkdir()
    _write_run(root, "baseline_bm25_smoke")
    _write_run(root, "baseline_dense_smoke")
    # A directory without metrics.json should be ignored.
    (root / "junk_dir").mkdir()
    # A loose file (not a directory) should be ignored.
    (root / "not_a_run.txt").write_text("hi")

    runs = cb.discover_runs([root])
    ids = sorted(r.run_id for r in runs)
    assert ids == ["baseline_bm25_smoke", "baseline_dense_smoke"]
    bm25 = next(r for r in runs if r.baseline_id == "bm25")
    assert bm25.is_smoke is True
    assert bm25.n_total == 16


def test_discover_skips_corrupt_metrics_json(tmp_path, caplog):
    root = tmp_path / "eval_results"
    root.mkdir()
    bad = root / "baseline_bm25_smoke"
    bad.mkdir()
    (bad / "metrics.json").write_text("{not json")
    _write_run(root, "baseline_dense_smoke")
    runs = cb.discover_runs([root])
    ids = [r.run_id for r in runs]
    assert ids == ["baseline_dense_smoke"]


def test_discover_dedupes_across_roots_keeping_freshest(tmp_path):
    """Same run_id in two roots → freshest mtime wins."""
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir(); root_b.mkdir()

    older = _write_run(root_a, "baseline_kg_hybrid_smoke",
                       overall=_full_stratum(citation_f1=0.10))
    newer = _write_run(root_b, "baseline_kg_hybrid_smoke",
                       overall=_full_stratum(citation_f1=0.30))
    # Force older root_a to have an older mtime than root_b.
    import os
    os.utime(older / "metrics.json", (1, 1))
    os.utime(newer / "metrics.json", (10**9, 10**9))

    runs = cb.discover_runs([root_a, root_b])
    assert len(runs) == 1
    assert runs[0].path == newer
    assert runs[0].metrics["overall"]["citation_f1"] == 0.30


def test_discover_include_smoke_only(tmp_path):
    root = tmp_path / "eval_results"
    root.mkdir()
    _write_run(root, "baseline_bm25_smoke")
    _write_run(root, "baseline_bm25")  # full run
    runs = cb.discover_runs([root], include_smoke=True, include_full=False)
    assert {r.run_id for r in runs} == {"baseline_bm25_smoke"}


def test_discover_include_full_only(tmp_path):
    root = tmp_path / "eval_results"
    root.mkdir()
    _write_run(root, "baseline_bm25_smoke")
    _write_run(root, "baseline_bm25")  # full run
    runs = cb.discover_runs([root], include_smoke=False, include_full=True)
    assert {r.run_id for r in runs} == {"baseline_bm25"}


def test_discover_runs_filter_glob(tmp_path):
    root = tmp_path / "eval_results"
    root.mkdir()
    _write_run(root, "baseline_bm25_smoke")
    _write_run(root, "baseline_dense_smoke")
    _write_run(root, "baseline_kg_hybrid_strat5")
    runs = cb.discover_runs(
        [root], runs_filter=["baseline_bm25_smoke", "baseline_kg_*"]
    )
    assert {r.run_id for r in runs} == {
        "baseline_bm25_smoke", "baseline_kg_hybrid_strat5"
    }


def test_discover_skips_missing_root(tmp_path):
    runs = cb.discover_runs([tmp_path / "does_not_exist"])
    assert runs == []


# ---------------------------------------------------------------------------
# select_freshest_per_baseline
# ---------------------------------------------------------------------------

def test_select_freshest_per_baseline_picks_largest_mtime(tmp_path):
    root = tmp_path / "eval_results"
    root.mkdir()
    older = _write_run(root, "baseline_bm25_smoke")
    newer = _write_run(root, "baseline_bm25")
    import os
    os.utime(older / "metrics.json", (1, 1))
    os.utime(newer / "metrics.json", (10**9, 10**9))

    runs = cb.discover_runs([root])
    selected = cb.select_freshest_per_baseline(runs)
    assert len(selected) == 1
    assert selected[0].run_id == "baseline_bm25"


def test_select_freshest_per_baseline_keeps_one_per_family(tmp_path):
    root = tmp_path / "eval_results"
    root.mkdir()
    _write_run(root, "baseline_bm25_smoke")
    _write_run(root, "baseline_dense_smoke")
    _write_run(root, "baseline_hybrid_smoke")
    _write_run(root, "baseline_hybrid_rerank_smoke")
    _write_run(root, "baseline_kg_smoke")
    _write_run(root, "baseline_kg_hybrid_smoke")
    _write_run(root, "phase0_smoke2")  # RLM

    runs = cb.discover_runs([root])
    selected = cb.select_freshest_per_baseline(runs)
    ids = sorted(r.baseline_id for r in selected)
    assert ids == ["bm25", "dense", "hybrid", "hybrid_rerank", "kg", "kg_hybrid", "rlm"]


def test_select_sort_order_baseline_then_rlm():
    runs = [
        cb.RunInfo("rlm_run", Path("x"), {}, "rlm", "RLM",
                   cb.RLM_SORT_ORDER, False, 0, mtime=0.0),
        cb.RunInfo("baseline_kg_smoke", Path("y"), {}, "kg", "KG (SPARQL)",
                   5, True, 0, mtime=0.0),
        cb.RunInfo("baseline_bm25_smoke", Path("z"), {}, "bm25", "BM25",
                   1, True, 0, mtime=0.0),
    ]
    selected = cb.select_freshest_per_baseline(runs)
    assert [r.baseline_id for r in selected] == ["bm25", "kg", "rlm"]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _runs_for_render():
    overall = _full_stratum(mrr_doc=0.80, mrr_article=0.30, citation_f1=0.15,
                            doc_citation_f1=0.46, recall_article=0.31)
    by_qt = {qt: _full_stratum(citation_f1=0.10) for qt in cb.QUERY_TYPES}
    by_qt["temporal_factual"] = _full_stratum(citation_f1=0.33,
                                              mrr_article=0.5)
    metrics = {
        "overall": overall,
        "by_query_type": by_qt,
        "counts": {
            "total": 16,
            "by_query_type": {qt: 2 for qt in cb.QUERY_TYPES},
        },
    }
    return [
        cb.RunInfo("baseline_bm25_smoke", Path("/tmp/bm25"), metrics,
                   "bm25", "BM25", 1, True, 16, mtime=1.0),
        cb.RunInfo("baseline_kg_hybrid_smoke", Path("/tmp/kgh"), metrics,
                   "kg_hybrid", "KG+Hybrid", 6, True, 16, mtime=1.0),
    ]


def test_render_overall_table_has_one_row_per_run():
    md = cb.render_overall_table(_runs_for_render())
    # header + sep + 2 rows = 4 lines
    assert len(md.strip().splitlines()) == 4
    assert "BM25" in md and "KG+Hybrid" in md
    assert "MRR@10 doc" in md and "Cite F1" in md


def test_render_per_type_block_for_temporal_factual():
    md = cb.render_per_type_block(_runs_for_render(), "temporal_factual")
    assert "0.330" in md  # the inflated Cite F1
    assert "BM25" in md and "KG+Hybrid" in md


def test_render_per_type_block_handles_missing_stratum():
    runs = _runs_for_render()
    # remove temporal_factual from the second run
    runs[1].metrics["by_query_type"].pop("temporal_factual")
    md = cb.render_per_type_block(runs, "temporal_factual")
    assert "—" in md  # at least one em-dash for the missing stratum


def test_render_headline_cite_f1_includes_overall_row():
    md = cb.render_headline_cite_f1(_runs_for_render())
    assert "**overall**" in md
    for qt in cb.QUERY_TYPES:
        assert qt in md


def test_render_report_contains_all_sections():
    md = cb.render_report(_runs_for_render())
    assert "# AlgerianLegalBench" in md
    assert "## Runs included" in md
    assert "## Headline — Cite F1 by query type" in md
    assert "## Overall metrics" in md
    assert "## Per query type" in md
    # At least one query-type sub-section is rendered.
    assert "### temporal_factual" in md


def test_render_report_empty_runs():
    assert "_No runs discovered._" in cb.render_report([])


def test_fmt_handles_nan_and_none():
    assert cb._fmt(None) == "—"
    assert cb._fmt(float("nan")) == "—"
    assert cb._fmt(0.123456) == "0.123"
    assert cb._fmt(0) == "0.000"


def test_md_table_alignment():
    out = cb._md_table(["A", "B"], [["x", "1"]], align=["left", "right"])
    lines = out.splitlines()
    assert lines[0] == "| A | B |"
    assert lines[1] == "| --- | ---: |"
    assert lines[2] == "| x | 1 |"


# ---------------------------------------------------------------------------
# main() — end-to-end
# ---------------------------------------------------------------------------

def test_main_writes_file_and_returns_zero(tmp_path, capsys):
    root = tmp_path / "eval_results"
    root.mkdir()
    _write_run(root, "baseline_bm25_smoke")
    _write_run(root, "baseline_kg_hybrid_smoke")
    out_path = tmp_path / "comparison.md"

    rc = cb.main([
        "--eval-roots", str(root),
        "--out", str(out_path),
    ])
    assert rc == 0
    assert out_path.is_file()
    md = out_path.read_text(encoding="utf-8")
    assert "BM25" in md and "KG+Hybrid" in md
    captured = capsys.readouterr()
    # Stdout receives the rendered markdown unless --no-stdout is set
    assert "BM25" in captured.out


def test_main_no_stdout_writes_file_only(tmp_path, capsys):
    root = tmp_path / "eval_results"
    root.mkdir()
    _write_run(root, "baseline_bm25_smoke")
    out_path = tmp_path / "comparison.md"

    rc = cb.main([
        "--eval-roots", str(root),
        "--out", str(out_path),
        "--no-stdout",
    ])
    assert rc == 0
    captured = capsys.readouterr()
    assert "BM25" not in captured.out
    assert out_path.read_text(encoding="utf-8").startswith("# AlgerianLegalBench")


def test_main_no_runs_returns_one(tmp_path):
    empty_root = tmp_path / "eval_results_empty"
    empty_root.mkdir()
    rc = cb.main(["--eval-roots", str(empty_root),
                  "--out", str(tmp_path / "out.md")])
    assert rc == 1


def test_main_runs_filter_skips_freshest_reduction(tmp_path):
    root = tmp_path / "eval_results"
    root.mkdir()
    _write_run(root, "baseline_bm25_smoke")
    _write_run(root, "baseline_bm25")  # full
    out_path = tmp_path / "compare.md"
    rc = cb.main([
        "--eval-roots", str(root),
        "--runs", "baseline_bm25*",
        "--out", str(out_path),
        "--no-stdout",
    ])
    assert rc == 0
    md = out_path.read_text(encoding="utf-8")
    # When --runs is supplied, the freshest-per-baseline reduction is
    # skipped, so BOTH bm25 runs should be present.
    assert "baseline_bm25_smoke" in md
    assert "baseline_bm25" in md


def test_main_default_picks_one_per_baseline(tmp_path):
    root = tmp_path / "eval_results"
    root.mkdir()
    older = _write_run(root, "baseline_bm25_smoke")
    newer = _write_run(root, "baseline_bm25")
    import os
    os.utime(older / "metrics.json", (1, 1))
    os.utime(newer / "metrics.json", (10**9, 10**9))

    out_path = tmp_path / "compare.md"
    rc = cb.main([
        "--eval-roots", str(root),
        "--out", str(out_path),
        "--no-stdout",
    ])
    assert rc == 0
    md = out_path.read_text(encoding="utf-8")
    # Default picks freshest-per-baseline → only baseline_bm25 (newer mtime),
    # not baseline_bm25_smoke. Path is on a code-fenced line; check the
    # backticked run_id strings to disambiguate.
    assert "`baseline_bm25`" in md
    assert "`baseline_bm25_smoke`" not in md


def test_main_all_flag_includes_every_run(tmp_path):
    root = tmp_path / "eval_results"
    root.mkdir()
    _write_run(root, "baseline_bm25_smoke")
    _write_run(root, "baseline_bm25")
    out_path = tmp_path / "compare.md"
    rc = cb.main([
        "--eval-roots", str(root),
        "--all",
        "--out", str(out_path),
        "--no-stdout",
    ])
    assert rc == 0
    md = out_path.read_text(encoding="utf-8")
    assert "`baseline_bm25_smoke`" in md
    assert "`baseline_bm25`" in md
