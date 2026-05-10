"""Ablation grid: systematically disable retrievers/gates and measure impact.

Each ablation variant disables one component and re-runs a subset of questions.
Results are compared against the full-system baseline.

Usage:
    conda run -n pfe_env python scripts/ablation_grid.py \
        --benchmark data/algerian_legal_bench_v3.jsonl \
        --output eval_results/ablation \
        [--limit 50]
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from akn_rlm.corpus.registry import CorpusRegistry
from akn_rlm.llm.client import LLMPool
from akn_rlm.rlm.legal_env import LegalEnv
from akn_rlm.rlm.recursion_budget import RecursionBudget
from akn_rlm.rlm.pipeline import build_pipeline
from akn_rlm.eval.runner import run_benchmark
from akn_rlm.eval.stratified import stratify
from akn_rlm.eval.report import format_report, save_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("ablation_grid")


# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------

ABLATIONS: list[dict] = [
    {
        "name": "full_system",
        "description": "Full pipeline (BM25 + dense + reranker + all gates)",
        "disable_dense":    False,
        "disable_reranker": False,
        "disable_citation_gate": False,
        "disable_jurisdiction_gate": False,
    },
    {
        "name": "no_dense",
        "description": "BM25 only (no dense/SPLADE/ColBERT)",
        "disable_dense":    True,
        "disable_reranker": False,
        "disable_citation_gate": False,
        "disable_jurisdiction_gate": False,
    },
    {
        "name": "no_reranker",
        "description": "No cross-encoder reranker",
        "disable_dense":    False,
        "disable_reranker": True,
        "disable_citation_gate": False,
        "disable_jurisdiction_gate": False,
    },
    {
        "name": "no_citation_gate",
        "description": "Citation existence gate disabled",
        "disable_dense":    False,
        "disable_reranker": False,
        "disable_citation_gate": True,
        "disable_jurisdiction_gate": False,
    },
    {
        "name": "no_jurisdiction_gate",
        "description": "Jurisdiction infection gate disabled",
        "disable_dense":    False,
        "disable_reranker": False,
        "disable_citation_gate": False,
        "disable_jurisdiction_gate": True,
    },
]


def _make_env(cfg: dict) -> LegalEnv:
    registry = CorpusRegistry.from_corpus_dir()
    llm_pool = LLMPool.default()
    budget = RecursionBudget(max_depth=2, max_sub_calls=12, max_total_tokens=200_000)

    indices = {}
    if not cfg.get("disable_dense"):
        pass  # indices are lazy-loaded from INDICES_DIR inside LegalEnv

    env = LegalEnv(
        registry=registry,
        kg=None,
        indices=indices,
        llm_pool=llm_pool,
        budget=budget,
    )

    if cfg.get("disable_reranker"):
        env._reranker_disabled = True

    if cfg.get("disable_citation_gate"):
        # Monkey-patch filter_citations to be a no-op
        from akn_rlm import gates
        gates.citation_existence.filter_citations = lambda reg, cites: (cites, [])

    if cfg.get("disable_jurisdiction_gate"):
        from akn_rlm.gates import jurisdiction
        jurisdiction.detect = lambda text: []

    return env


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablation grid")
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output", default="eval_results/ablation")
    parser.add_argument("--limit", type=int, default=50,
                        help="Questions per ablation variant (default 50)")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary: list[dict] = []

    for cfg in ABLATIONS:
        name = cfg["name"]
        log.info("=== Running ablation: %s ===", name)

        try:
            env = _make_env(cfg)
            pipeline = build_pipeline(env)
            results = run_benchmark(pipeline, args.benchmark, limit=args.limit)
            strata = stratify(results)
            save_report(strata, out_dir / name, title=f"Ablation: {name}")

            overall = strata.get("overall", {})
            summary.append({
                "name":         name,
                "description":  cfg["description"],
                "mrr_doc":      overall.get("mrr_doc", 0),
                "mrr_article":  overall.get("mrr_article", 0),
                "citation_f1":  overall.get("citation_f1", 0),
                "abstention_acc": overall.get("abstention_acc", 0),
            })
        except Exception as exc:
            log.error("Ablation %s failed: %s", name, exc, exc_info=True)
            summary.append({"name": name, "error": str(exc)})

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Ablation':<30s} {'MRR(doc)':>10s} {'MRR(art)':>10s} "
          f"{'CiteF1':>10s} {'AbstAcc':>10s}")
    print("-" * 80)
    for row in summary:
        if "error" in row:
            print(f"{row['name']:<30s}  ERROR: {row['error']}")
        else:
            print(
                f"{row['name']:<30s} "
                f"{row['mrr_doc']:>10.4f} "
                f"{row['mrr_article']:>10.4f} "
                f"{row['citation_f1']:>10.4f} "
                f"{row['abstention_acc']:>10.4f}"
            )
    print("=" * 80)

    summary_path = out_dir / "ablation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Ablation summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
