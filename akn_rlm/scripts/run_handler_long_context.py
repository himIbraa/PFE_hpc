"""Run AlgerianLegalBench v3.0 long_context slice through the
Phase-2 long_context handler.

This is the smoke runner for R6.4. It loads:

  - the article registry (corpus parse → fast),
  - the BM25 index,
  - the dense (FAISS+e5-small) index,
  - the doc-router (R1),
  - and the LLM pool (sub-LM summariser only).

It does NOT load: ColBERT, SPLADE, the LangGraph pipeline, the KG, the
faithfulness gate. Sub-LM call budget = 1 summariser per query.
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
from akn_rlm.rlm.handlers import build_long_context_handler  # noqa: E402
from akn_rlm.rlm.handlers.long_context import (  # noqa: E402
    DEFAULT_FINAL_TOP_K,
    DEFAULT_K_EACH,
)
from akn_rlm.rlm.routing import build_doc_router  # noqa: E402

from scripts.run_benchmark import (  # noqa: E402
    _benchmark_to_records,
    _stratified_sample,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_handler_long_context")


def _run(handler, records, *, limit, query_types, difficulty, stratified,
         output_dir, run_id):
    questions = records
    if query_types:
        questions = [q for q in questions if q.get("query_type") in query_types]
    if difficulty:
        questions = [q for q in questions if q.get("difficulty") == difficulty]
    if stratified:
        questions = _stratified_sample(questions, stratified)
        log.info("Stratified sample: %d questions across %d types",
                 len(questions),
                 len({q.get("query_type") for q in questions}))
    if limit:
        questions = questions[:limit]

    log.info("Running %d questions through long_context handler …", len(questions))
    results = []
    for i, q in enumerate(questions):
        t0 = time.time()
        try:
            answer = handler.run(q["query"])
        except Exception as exc:
            log.error("Q%s failed: %s", q.get("id", i), exc)
            answer = {"answer_text": "", "abstention": True,
                      "abstention_reason": "pipeline_error", "citations": []}
        answer["_latency_s"] = time.time() - t0
        results.append(_answer_to_result(q, answer))
        if (i + 1) % 5 == 0 or (i + 1) == len(questions):
            avg = sum(r.get("latency_s", 0) for r in results) / len(results)
            log.info("  Progress: %d/%d  (%.3fs/q avg)", i + 1, len(questions), avg)

    out_dir = output_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

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
        format_report(strata, title=f"RLM long_context handler  {run_id}"),
        encoding="utf-8",
    )
    log.info("All results saved to %s/", out_dir)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AlgerianLegalBench v3.0 against the Phase-2 long_context handler"
    )
    parser.add_argument("--benchmark", default=None)
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
        default=["long_context"],
    )
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default=None)
    parser.add_argument("--final-top-k", type=int, default=DEFAULT_FINAL_TOP_K)
    parser.add_argument("--k-each",      type=int, default=DEFAULT_K_EACH)
    parser.add_argument("--sub-model",   default=SUB_LLM_MODEL)
    args = parser.parse_args()

    run_id = args.run_id or time.strftime("rlm_long_context_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    benchmark_path = Path(args.benchmark) if args.benchmark else get_benchmark_path()

    log.info("=" * 64)
    log.info("RLM long_context handler  run_id=%s", run_id)
    log.info("Benchmark : %s", benchmark_path)
    log.info("Output    : %s/%s/", output_dir, run_id)
    log.info("=" * 64)

    log.info("Parsing corpus to build registry …")
    registry = ArticleRegistry()
    registry.build(parse_all())

    if not BM25_INDEX_PATH.exists():
        raise FileNotFoundError(f"BM25 index missing at {BM25_INDEX_PATH}")
    if not DENSE_FAISS_PATH.exists() or not DENSE_META_PATH.exists():
        raise FileNotFoundError("Dense index missing")

    bm25 = BM25Index.load(BM25_INDEX_PATH)
    dense = DenseIndex.load(DENSE_FAISS_PATH, DENSE_META_PATH)

    router = build_doc_router(registry=registry, bm25=bm25)
    llm_pool = LLMPool.default()

    handler = build_long_context_handler(
        bm25=bm25, dense=dense, registry=registry,
        llm_pool=llm_pool, router=router,
        sub_model=args.sub_model,
        final_top_k=args.final_top_k,
        k_each=args.k_each,
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
    print_report(strata, title=f"RLM long_context handler  {run_id}")
    log.info("Done. Results in %s/%s/", output_dir, run_id)


if __name__ == "__main__":
    main()
