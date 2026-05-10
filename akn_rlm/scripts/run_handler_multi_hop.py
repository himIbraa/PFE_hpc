"""Run AlgerianLegalBench v3.0 multi_hop slice through the Phase-2 multi_hop handler.

This is the smoke runner for R2. It loads only what the handler needs
(BM25 index, dense index, registry, doc-router, LLM pool) and skips
everything the deterministic baselines also skip (KG, ColBERT, SPLADE,
temporal index, LangGraph pipeline, faithfulness gate). Output layout
matches ``run_baseline_*.py`` so ``scripts/compare_baselines.py`` can
read it directly:

    eval_results/{run_id}/predictions.jsonl
    eval_results/{run_id}/metrics.json
    eval_results/{run_id}/metrics.md
    eval_results/{run_id}/report.txt

Usage:
    $py = "C:\\Users\\21355\\.conda\\envs\\pfe_env\\python.exe"

    # Multi-hop only stratified smoke (default — what R2's gate cares about)
    & $py scripts\\run_handler_multi_hop.py --stratified 5 --query-types multi_hop \\
          --run-id rlm_multi_hop_strat5

    # Full multi_hop slice (26 questions on ALB v3.0)
    & $py scripts\\run_handler_multi_hop.py --query-types multi_hop \\
          --run-id rlm_multi_hop_full
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from akn_rlm.config import (  # noqa: E402
    BM25_INDEX_PATH,
    DENSE_FAISS_PATH,
    DENSE_META_PATH,
    SUB_LLM_MODEL,
    get_benchmark_path,
)
from akn_rlm.corpus.akn_parser import parse_all  # noqa: E402
from akn_rlm.corpus.article_registry import ArticleRegistry  # noqa: E402
from akn_rlm.eval.report import format_report, print_report  # noqa: E402
from akn_rlm.eval.runner import _answer_to_result, _format_markdown  # noqa: E402
from akn_rlm.eval.stratified import stratify  # noqa: E402
from akn_rlm.indexers.bm25 import BM25Index  # noqa: E402
from akn_rlm.indexers.dense import DenseIndex  # noqa: E402
from akn_rlm.llm.client import LLMPool  # noqa: E402
from akn_rlm.rlm.handlers import (  # noqa: E402
    DEFAULT_FINAL_TOP_K,
    DEFAULT_K_EACH,
    DEFAULT_MAX_SUB_QS,
    DEFAULT_TOP_K_PER_SUBQ,
    DEFAULT_VERIFY_THRESHOLD,
    DEFAULT_VERIFY_TOP_N,
    build_multi_hop_handler,
)
from akn_rlm.rlm.routing import build_doc_router  # noqa: E402

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
log = logging.getLogger("run_handler_multi_hop")


def _run(
    handler,
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

    log.info("Running %d questions through multi_hop handler …", len(questions))

    results: list[dict] = []
    for i, q in enumerate(questions):
        t0 = time.time()
        try:
            answer = handler.run(q["query"])
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

        if (i + 1) % 5 == 0 or (i + 1) == len(questions):
            avg = sum(r.get("latency_s", 0) for r in results) / len(results)
            log.info("  Progress: %d/%d  (%.1fs/q avg)", i + 1, len(questions), avg)

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
        format_report(strata, title=f"RLM multi_hop handler  {run_id}"),
        encoding="utf-8",
    )

    log.info("All results saved to %s/", out_dir)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AlgerianLegalBench v3.0 against the Phase-2 multi_hop handler"
    )
    parser.add_argument("--benchmark", default=None,
                        help="Path to AlgerianLegalBench JSON file (default: auto-detect)")
    parser.add_argument("--output-dir", default="eval_results")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--stratified", type=int, default=None)
    parser.add_argument(
        "--query-types", nargs="*",
        choices=[
            "rule_application", "exact_article", "multi_hop", "unanswerable",
            "layman", "long_context", "conceptual_definitional", "temporal_factual",
        ],
        # Default to multi_hop only — the gate the handler is designed to clear.
        default=["multi_hop"],
        help="Only evaluate these query types (default: multi_hop)",
    )
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default=None)
    parser.add_argument("--top-k-per-subq", type=int, default=DEFAULT_TOP_K_PER_SUBQ)
    parser.add_argument("--verify-top-n",   type=int, default=DEFAULT_VERIFY_TOP_N)
    parser.add_argument("--final-top-k",    type=int, default=DEFAULT_FINAL_TOP_K)
    parser.add_argument("--k-each",         type=int, default=DEFAULT_K_EACH)
    parser.add_argument("--max-sub-qs",     type=int, default=DEFAULT_MAX_SUB_QS)
    parser.add_argument("--verify-threshold", type=float, default=DEFAULT_VERIFY_THRESHOLD)
    parser.add_argument("--sub-model",      default=SUB_LLM_MODEL,
                        help=f"Sub-LM model name (default: {SUB_LLM_MODEL})")
    args = parser.parse_args()

    run_id = args.run_id or time.strftime("rlm_multi_hop_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    benchmark_path = Path(args.benchmark) if args.benchmark else get_benchmark_path()

    log.info("=" * 64)
    log.info("RLM multi_hop handler  run_id=%s", run_id)
    log.info("Benchmark : %s", benchmark_path)
    log.info("Output    : %s/%s/", output_dir, run_id)
    log.info("Sub model : %s", args.sub_model)
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
            f"Dense index missing — run scripts/build_indices.py first"
        )

    log.info("Loading BM25 index …")
    bm25 = BM25Index.load(BM25_INDEX_PATH)
    log.info("Loading dense index …")
    dense = DenseIndex.load(DENSE_FAISS_PATH, DENSE_META_PATH)

    log.info("Building doc-router …")
    router = build_doc_router(registry=registry, bm25=bm25)

    log.info("Connecting to LLM pool …")
    llm_pool = LLMPool.default()

    handler = build_multi_hop_handler(
        bm25=bm25, dense=dense, registry=registry,
        llm_pool=llm_pool, router=router,
        sub_model=args.sub_model,
        top_k_per_subq=args.top_k_per_subq,
        verify_top_n=args.verify_top_n,
        final_top_k=args.final_top_k,
        k_each=args.k_each,
        max_sub_qs=args.max_sub_qs,
        verify_threshold=args.verify_threshold,
    )

    results = _run(
        handler, records=_benchmark_to_records(benchmark_path, registry),
        limit=args.limit,
        query_types=args.query_types,
        difficulty=args.difficulty,
        stratified=args.stratified,
        output_dir=output_dir,
        run_id=run_id,
    )

    strata = stratify(results)
    print_report(strata, title=f"RLM multi_hop handler  {run_id}")
    log.info("Done. Results in %s/%s/", output_dir, run_id)


if __name__ == "__main__":
    main()
