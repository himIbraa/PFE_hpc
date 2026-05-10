"""Run AlgerianLegalBench v3.0 against the KG-augmented hybrid baseline (B6).

Pipeline: KG entity / token-coverage SPARQL on the original query →
extract surface labels from matched spans → append them to the query as
expansion → RRF(BM25, Dense) hybrid retrieval on the rewritten query →
small score boost for fused candidates that the KG already surfaced →
top-K (default 5) feed the same Arabic template answer used by B1-B5.
No LLM, no reranker.

Output layout matches ``run_benchmark.py`` so
``scripts/compare_baselines.py`` consumes it directly:

    eval_results/{run_id}/predictions.jsonl
    eval_results/{run_id}/metrics.json
    eval_results/{run_id}/metrics.md
    eval_results/{run_id}/report.txt

Usage:
    $py = "C:\\Users\\21355\\.conda\\envs\\pfe_env\\python.exe"

    # Stratified smoke (16-q sample across all 8 query types)
    & $py scripts\\run_baseline_kg_hybrid.py --stratified 2 --run-id baseline_kg_hybrid_smoke

    # Full 244-question run
    & $py scripts\\run_baseline_kg_hybrid.py --run-id baseline_kg_hybrid

    # Tune the bias and expansion budget
    & $py scripts\\run_baseline_kg_hybrid.py --kg-boost 0.02 --expansion-terms 8 \
        --run-id baseline_kg_hybrid_tuned
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from pathlib import Path

# UTF-8 console on Windows (matches run_benchmark.py)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Ensure package root is importable
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from akn_rlm.baselines.kg_hybrid_pipeline import (  # noqa: E402
    DEFAULT_EXPANSION_TERMS,
    DEFAULT_K_EACH,
    DEFAULT_KG_BOOST,
    DEFAULT_TOP_K,
    build_kg_hybrid_pipeline,
)
from akn_rlm.config import (  # noqa: E402
    BM25_INDEX_PATH,
    DENSE_FAISS_PATH,
    DENSE_META_PATH,
    get_benchmark_path,
    get_kg_path,
)
from akn_rlm.corpus.akn_parser import parse_all  # noqa: E402
from akn_rlm.corpus.article_registry import ArticleRegistry  # noqa: E402
from akn_rlm.corpus.kg_loader import load_kg  # noqa: E402
from akn_rlm.eval.report import format_report, print_report  # noqa: E402
from akn_rlm.eval.runner import _answer_to_result, _format_markdown  # noqa: E402
from akn_rlm.eval.stratified import stratify  # noqa: E402
from akn_rlm.indexers.bm25 import BM25Index  # noqa: E402
from akn_rlm.indexers.dense import DenseIndex  # noqa: E402

# Reuse the benchmark-format converter + stratified sampler from run_benchmark
from scripts.run_benchmark import (  # noqa: E402
    _benchmark_to_records,
    _stratified_sample,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_baseline_kg_hybrid")


def _run(
    pipeline,
    records: list[dict],
    *,
    limit: int | None,
    query_types: list[str] | None,
    difficulty: str | None,
    stratified: int | None,
    output_dir: Path,
    run_id: str,
) -> list[dict]:
    questions = records
    if query_types:
        questions = [q for q in questions if q.get("query_type") in query_types]
    if difficulty:
        questions = [q for q in questions if q.get("difficulty") == difficulty]
    if stratified:
        questions = _stratified_sample(questions, stratified)
        log.info(
            "Stratified sample: %d questions across %d types",
            len(questions),
            len({q.get("query_type") for q in questions}),
        )
    if limit:
        questions = questions[:limit]

    log.info("Running %d questions through KG-hybrid baseline …", len(questions))

    results: list[dict] = []
    for i, q in enumerate(questions):
        t0 = time.time()
        try:
            answer = pipeline.run(q["query"])
        except Exception as exc:
            log.error("Q%s failed: %s", q.get("id", i), exc)
            answer = {
                "answer_text": "",
                "abstention": True,
                "abstention_reason": "pipeline_error",
                "citations": [],
            }
        answer["_latency_s"] = time.time() - t0
        results.append(_answer_to_result(q, answer))

        if (i + 1) % 50 == 0 or (i + 1) == len(questions):
            avg = sum(r.get("latency_s", 0) for r in results) / len(results)
            log.info("  Progress: %d/%d  (%.3fs/q avg)", i + 1, len(questions), avg)

    out_dir = output_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = out_dir / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    log.info("Predictions saved → %s", pred_path)

    strata = stratify(results)

    def _default(obj):
        if isinstance(obj, set):
            return sorted(obj)
        raise TypeError

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(strata, fh, ensure_ascii=False, indent=2, default=_default)

    md_path = out_dir / "metrics.md"
    md_path.write_text(_format_markdown(strata), encoding="utf-8")

    report_path = out_dir / "report.txt"
    report_path.write_text(
        format_report(strata, title=f"KG-hybrid baseline  {run_id}"),
        encoding="utf-8",
    )

    log.info("All results saved to %s/", out_dir)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AlgerianLegalBench v3.0 against the KG-augmented hybrid baseline"
    )
    parser.add_argument("--benchmark", default=None,
                        help="Path to AlgerianLegalBench JSON file (default: auto-detect)")
    parser.add_argument("--output-dir", default="eval_results",
                        help="Parent directory for run outputs (default: eval_results/)")
    parser.add_argument("--run-id", default=None,
                        help="Sub-directory name for this run (default: timestamp)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after N questions (for quick testing)")
    parser.add_argument("--stratified", type=int, default=None,
                        help="Pick N questions per query_type (covers all 8 types)")
    parser.add_argument(
        "--query-types", nargs="*",
        choices=[
            "rule_application", "exact_article", "multi_hop", "unanswerable",
            "layman", "long_context", "conceptual_definitional", "temporal_factual",
        ],
        default=None,
        help="Only evaluate these query types",
    )
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default=None,
                        help="Only evaluate this difficulty level")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                        help=f"Top-K fused hits to cite (default: {DEFAULT_TOP_K})")
    parser.add_argument("--k-each", type=int, default=DEFAULT_K_EACH,
                        help=f"Per-retriever depth before RRF fusion (default: {DEFAULT_K_EACH})")
    parser.add_argument("--expansion-terms", type=int, default=DEFAULT_EXPANSION_TERMS,
                        help=f"Max KG-derived terms appended to the query (default: {DEFAULT_EXPANSION_TERMS})")
    parser.add_argument("--kg-boost", type=float, default=DEFAULT_KG_BOOST,
                        help=f"RRF score boost for KG-surfaced (doc_id, ref) (default: {DEFAULT_KG_BOOST})")
    args = parser.parse_args()

    run_id = args.run_id or time.strftime("baseline_kg_hybrid_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    benchmark_path = Path(args.benchmark) if args.benchmark else get_benchmark_path()

    log.info("=" * 64)
    log.info("KG-hybrid baseline  run_id=%s", run_id)
    log.info("Benchmark : %s", benchmark_path)
    log.info("Output    : %s/%s/", output_dir, run_id)
    log.info("Top-K     : %d   K-each: %d", args.top_k, args.k_each)
    log.info("Expansion : %d terms   KG boost: %.4f", args.expansion_terms, args.kg_boost)
    log.info("=" * 64)

    log.info("Parsing corpus to build registry …")
    registry = ArticleRegistry()
    registry.build(parse_all())

    if not BM25_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"BM25 index missing at {BM25_INDEX_PATH} — run scripts/build_indices.py first"
        )
    if not DENSE_FAISS_PATH.exists() or not DENSE_META_PATH.exists():
        raise FileNotFoundError(
            f"Dense index missing at {DENSE_FAISS_PATH} / {DENSE_META_PATH} — "
            "run scripts/build_indices.py first"
        )

    kg_path = get_kg_path()
    log.info("Loading KG from %s …", kg_path)
    kg = load_kg(kg_path)

    log.info("Loading BM25 index …")
    bm25 = BM25Index.load(BM25_INDEX_PATH)
    log.info("Loading dense index …")
    dense = DenseIndex.load(DENSE_FAISS_PATH, DENSE_META_PATH)

    pipeline = build_kg_hybrid_pipeline(
        kg=kg, bm25=bm25, dense=dense, registry=registry,
        top_k=args.top_k, k_each=args.k_each,
        expansion_terms_max=args.expansion_terms,
        kg_boost=args.kg_boost,
    )

    results = _run(
        pipeline, records=_benchmark_to_records(benchmark_path, registry),
        limit=args.limit,
        query_types=args.query_types,
        difficulty=args.difficulty,
        stratified=args.stratified,
        output_dir=output_dir,
        run_id=run_id,
    )

    strata = stratify(results)
    print_report(strata, title=f"KG-hybrid baseline  {run_id}")
    log.info("Done. Results in %s/%s/", output_dir, run_id)


if __name__ == "__main__":
    main()
