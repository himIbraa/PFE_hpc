"""Run AlgerianLegalBench v3.0 evaluation against the full AKN-RLM pipeline.

Reads the native JSON benchmark format (AlgerianLegalBench_v3.0_final.json),
resolves filename-based doc IDs to canonical FRBR IDs via the registry,
loads all built indices, runs every question through the pipeline, and saves
full results.

Usage:
    # Full run (244 questions — will take several hours with LLM calls)
    conda run -n pfe_env python scripts/run_benchmark.py --run-id run_001

    # Quick smoke (first 20 questions)
    conda run -n pfe_env python scripts/run_benchmark.py --limit 20 --run-id smoke_01

    # Only answerable questions
    conda run -n pfe_env python scripts/run_benchmark.py \\
        --query-types rule_application temporal multi_hop --run-id answerable_only

Output files (under eval_results/{run_id}/):
    predictions.jsonl   — one JSON object per question (full answer + metrics)
    metrics.json        — full stratified metrics tree
    metrics.md          — markdown table for quick review
    report.txt          — human-readable report with Δ vs baseline
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import sys
import time
from pathlib import Path

# UTF-8 console on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Ensure package root is importable
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from akn_rlm.config import (
    BM25_INDEX_PATH,
    COLBERT_INDEX_DIR,
    DENSE_FAISS_PATH,
    DENSE_META_PATH,
    ENABLE_COLBERT,
    ENABLE_DENSE,
    ENABLE_SPLADE,
    SPLADE_INDEX_PATH,
    TEMPORAL_INDEX_PATH,
    get_benchmark_path,
    get_kg_path,
)
from akn_rlm.corpus.akn_parser import parse_all
from akn_rlm.corpus.article_registry import ArticleRegistry
from akn_rlm.normalizers import canonical_article_ref
from akn_rlm.eval.report import print_report, save_report, format_report
from akn_rlm.eval.runner import _answer_to_result, _format_markdown
from akn_rlm.eval.stratified import stratify
from akn_rlm.llm.client import LLMPool
from akn_rlm.rlm.legal_env import LegalEnv
from akn_rlm.rlm.pipeline import build_pipeline
from akn_rlm.rlm.recursion_budget import RecursionBudget

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_benchmark")


# ---------------------------------------------------------------------------
# Benchmark format converter
# ---------------------------------------------------------------------------

def _resolve(registry: ArticleRegistry, doc_id: str) -> str:
    """Resolve a filename-based or canonical doc_id to canonical FRBR ID."""
    canonical = registry.resolve_alias(doc_id.lower()) or registry.resolve_alias(doc_id)
    if canonical:
        return canonical
    if registry.get_doc(doc_id):
        return doc_id
    return doc_id   # fall back: pass through unchanged


def _benchmark_to_records(benchmark_path: Path, registry: ArticleRegistry) -> list[dict]:
    """Load the native JSON benchmark and convert to runner-compatible records."""
    with benchmark_path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    raw_questions = data.get("questions", data) if isinstance(data, dict) else data
    log.info("Benchmark: %d questions loaded from %s", len(raw_questions), benchmark_path.name)

    records = []
    for q in raw_questions:
        # Resolve expected_documents (may use filename-based IDs)
        gold_doc_ids = [
            _resolve(registry, d) for d in q.get("expected_documents", [])
        ]

        # Build gold_citations + gold_article_ids from expected_articles.
        # article_ref is canonicalised so gold matches whatever the index/
        # citations side produces (1 == "الأولى" == "art_1", 9_bis == "9 مكرر",
        # etc.)
        gold_cites: list[dict] = []
        gold_art_ids: list[str] = []
        for art in q.get("expected_articles", []):
            doc_id = _resolve(registry, art.get("document_id", ""))
            art_ref_canon = canonical_article_ref(str(art.get("article_ref", "")))
            gold_cites.append({"doc_id": doc_id, "article_ref": art_ref_canon})
            art_key = f"{doc_id}#art_{art_ref_canon}"
            if art_key not in gold_art_ids:
                gold_art_ids.append(art_key)

        records.append({
            "id":                  q.get("id", ""),
            "query":               q.get("question", q.get("query", "")),
            "query_type":          q.get("query_type", "rule_application"),
            "legal_category":      q.get("category", q.get("legal_category", "unknown")),
            "difficulty":          q.get("difficulty", "unknown"),
            "language":            q.get("language", "ar"),
            "split":               q.get("split", "test"),
            "temporal_note":       q.get("temporal_note"),
            "gold_doc_ids":        gold_doc_ids,
            "gold_article_ids":    gold_art_ids,
            "gold_citations":      gold_cites,
            "gold_abstain":        not bool(q.get("answerable", True)),
            "gold_answer":         q.get("ground_truth_answer", ""),
            "gold_reasoning_chain": q.get("reasoning_chain", []),
        })

    return records


# ---------------------------------------------------------------------------
# Index loader
# ---------------------------------------------------------------------------

def _load_indices() -> dict:
    indices = {}

    if BM25_INDEX_PATH.exists():
        log.info("Loading BM25 index …")
        from akn_rlm.indexers.bm25 import BM25Index
        indices["bm25"] = BM25Index.load(BM25_INDEX_PATH)
        log.info("  BM25 loaded: %s documents", getattr(indices["bm25"], "_n_docs", "?"))
    else:
        log.warning("BM25 index not found at %s — run build_indices.py first", BM25_INDEX_PATH)

    if not ENABLE_SPLADE:
        log.info("SPLADE disabled (ENABLE_SPLADE=False) — skipping")
    elif SPLADE_INDEX_PATH.with_suffix(".npz").exists():
        log.info("Loading SPLADE index …")
        try:
            from akn_rlm.indexers.splade import SPLADEIndex
            indices["splade"] = SPLADEIndex.load(SPLADE_INDEX_PATH)
            # Warm up encoder with a short query to catch OOM before evaluation
            _probe = indices["splade"].search("test", k=1)
            log.info("  SPLADE loaded and warmed up (%d probe hits)", len(_probe))
        except MemoryError:
            log.warning("SPLADE load/warmup failed (MemoryError) — skipping")
            indices.pop("splade", None)
        except Exception as exc:
            log.warning("SPLADE load/warmup failed (%s) — skipping", exc)
            indices.pop("splade", None)
    else:
        log.warning("SPLADE index not found — skipping")

    if not ENABLE_DENSE:
        log.info("Dense retrieval disabled (ENABLE_DENSE=False) — skipping")
    elif DENSE_FAISS_PATH.exists() and DENSE_META_PATH.exists():
        log.info("Loading Dense (FAISS) index …")
        try:
            from akn_rlm.indexers.dense import DenseIndex
            indices["dense"] = DenseIndex.load(DENSE_FAISS_PATH, DENSE_META_PATH)
            _probe = indices["dense"].search("test", k=1)
            log.info("  Dense loaded and warmed up (%d probe hits)", len(_probe))
        except MemoryError:
            log.warning("Dense index load failed (MemoryError) — skipping; add RAM or use mmap")
            indices.pop("dense", None)
        except Exception as exc:
            log.warning("Dense index load failed (%s) — skipping", exc)
            indices.pop("dense", None)
    else:
        log.warning("Dense index not found — run build_indices.py first")

    if not ENABLE_COLBERT:
        log.info("ColBERT disabled (ENABLE_COLBERT=False) — skipping")
    elif COLBERT_INDEX_DIR.exists() and any(COLBERT_INDEX_DIR.iterdir()):
        log.info("Loading ColBERT index …")
        try:
            from akn_rlm.indexers.colbert import ColBERTIndex_load
            indices["colbert"] = ColBERTIndex_load(COLBERT_INDEX_DIR)
            _probe = indices["colbert"].search("test", k=1)
            log.info("  ColBERT loaded and warmed up (%d probe hits)", len(_probe))
        except MemoryError:
            log.warning("ColBERT index load failed (MemoryError) — skipping")
            indices.pop("colbert", None)
        except Exception as exc:
            log.warning("ColBERT index load failed (%s) — skipping", exc)
            indices.pop("colbert", None)
    else:
        log.warning("ColBERT index not found — skipping")

    log.info("Indices loaded: %s", list(indices.keys()))
    return indices


# ---------------------------------------------------------------------------
# Runner that works with pre-loaded records (bypasses JSONL expectation)
# ---------------------------------------------------------------------------

def _stratified_sample(records: list[dict], n_per_type: int) -> list[dict]:
    """Pick the first n_per_type questions of each query_type (stable order).

    Ensures the sample covers every query type the benchmark exposes, so a
    smoke run gives signal on multi_hop / temporal_factual / unanswerable etc.
    Returns up to ``n_per_type * |types|`` records.
    """
    by_type: dict[str, list[dict]] = {}
    for r in records:
        by_type.setdefault(r.get("query_type", "rule_application"), []).append(r)
    sample: list[dict] = []
    for qt in sorted(by_type):
        sample.extend(by_type[qt][:n_per_type])
    return sample


def _run_records(
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
        log.info("Stratified sample: %d questions across %d types",
                 len(questions),
                 len({q.get("query_type") for q in questions}))
    if limit:
        questions = questions[:limit]

    log.info("Running %d questions …", len(questions))

    results = []
    for i, q in enumerate(questions):
        t0 = time.time()
        try:
            answer = pipeline.run(q["query"])
        except Exception as exc:
            log.error("Q%s failed: %s", q.get("id", i), exc)
            answer = {
                "answer_text": "", "abstention": True,
                "abstention_reason": "pipeline_error", "citations": [],
            }
        answer["_latency_s"] = time.time() - t0
        results.append(_answer_to_result(q, answer))

        if (i + 1) % 10 == 0 or (i + 1) == len(questions):
            log.info("  Progress: %d/%d  (%.1fs/q avg)",
                     i + 1, len(questions),
                     sum(r.get("latency_s", 0) for r in results) / len(results))

    # Save all outputs
    out_dir = output_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # predictions.jsonl
    pred_path = out_dir / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    log.info("Predictions saved → %s", pred_path)

    # Stratified metrics
    strata = stratify(results)

    def _default(obj):
        if isinstance(obj, set):
            return sorted(obj)
        raise TypeError

    metrics_path = out_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as fh:
        json.dump(strata, fh, ensure_ascii=False, indent=2, default=_default)

    # Markdown metrics table
    from akn_rlm.eval.runner import _format_markdown
    md_path = out_dir / "metrics.md"
    md_path.write_text(_format_markdown(strata), encoding="utf-8")

    # Human report
    report_path = out_dir / "report.txt"
    report_path.write_text(format_report(strata), encoding="utf-8")

    log.info("All results saved to %s/", out_dir)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run AlgerianLegalBench v3.0 full evaluation"
    )
    parser.add_argument(
        "--benchmark",
        default=None,
        help="Path to AlgerianLegalBench JSON file (default: auto-detect via config)",
    )
    parser.add_argument(
        "--output-dir",
        default="eval_results",
        help="Parent directory for run outputs (default: eval_results/)",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Sub-directory name for this run (default: timestamp)",
    )
    parser.add_argument("--limit", type=int, default=None,
                        help="Stop after N questions (for quick testing)")
    parser.add_argument(
        "--stratified", type=int, default=None,
        help="Pick N questions per query_type (covers all 8 types). "
             "Pass --stratified 2 for a 16-question smoke that exercises every type.",
    )
    parser.add_argument(
        "--query-types", nargs="*",
        choices=[
            "rule_application", "exact_article", "multi_hop", "unanswerable",
            "layman", "long_context", "conceptual_definitional", "temporal_factual",
        ],
        default=None,
        help="Only evaluate these query types",
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        default=None,
        help="Only evaluate this difficulty level",
    )
    args = parser.parse_args()

    run_id = args.run_id or time.strftime("run_%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    benchmark_path = Path(args.benchmark) if args.benchmark else get_benchmark_path()

    log.info("=" * 64)
    log.info("AKN-RLM Benchmark  run_id=%s", run_id)
    log.info("Benchmark : %s", benchmark_path)
    log.info("Output    : %s/%s/", output_dir, run_id)
    log.info("=" * 64)

    # Build registry (parse XML corpus once — needed for doc_id resolution + pipeline)
    log.info("Parsing corpus to build registry …")
    registry = ArticleRegistry()
    registry.build(parse_all())

    # Convert benchmark questions
    records = _benchmark_to_records(benchmark_path, registry)

    # Load indices
    indices = _load_indices()

    # Load RDF knowledge graph (optional — KG primitives disabled if absent)
    kg = None
    try:
        from akn_rlm.corpus.kg_loader import load_kg
        kg_path = get_kg_path()
        kg = load_kg(kg_path)
        log.info("KG loaded: %s", kg_path.name)
    except FileNotFoundError:
        log.warning("KG TTL not found — env.kg_* primitives will be no-ops")
    except Exception as exc:
        log.warning("KG load failed (%s) — env.kg_* primitives disabled", exc)

    # Load temporal index (optional)
    temporal_index: dict = {}
    if TEMPORAL_INDEX_PATH.exists():
        try:
            with TEMPORAL_INDEX_PATH.open(encoding="utf-8") as fh:
                temporal_index = json.load(fh)
            log.info("Temporal index loaded: %d article chains", len(temporal_index))
        except Exception as exc:
            log.warning("Temporal index load failed (%s)", exc)
    else:
        log.warning("Temporal index not found at %s", TEMPORAL_INDEX_PATH)

    # Build LLM pool
    log.info("Connecting to LLM pool …")
    llm_pool = LLMPool.default()

    # Budget is shared across pipeline nodes within one query; _run_records
    # creates a fresh budget object per query (see below).
    budget = RecursionBudget(
        max_depth=2,
        max_sub_calls=12,
        max_total_tokens=200_000,
    )

    env = LegalEnv(
        registry=registry,
        kg=kg,
        indices=indices,
        llm_pool=llm_pool,
        budget=budget,
        temporal_index=temporal_index,
    )

    pipeline = build_pipeline(env)

    log.info("Starting evaluation …")
    results = _run_records(
        pipeline, records,
        limit=args.limit,
        query_types=args.query_types,
        difficulty=args.difficulty,
        stratified=args.stratified,
        output_dir=output_dir,
        run_id=run_id,
    )

    # Print summary to console
    strata = stratify(results)
    print_report(strata, title=f"AKN-RLM  {run_id}")
    log.info("Done. Results in %s/%s/", output_dir, run_id)


if __name__ == "__main__":
    main()
