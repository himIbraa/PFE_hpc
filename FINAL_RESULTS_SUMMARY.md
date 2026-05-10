# Final Results Summary

Date: 2026-05-01

This summary consolidates the measured performance of the current RAG app, with emphasis on retrieval recall and MRR. Unless noted otherwise, retrieval metrics come from the strict thesis evaluator and pipeline metrics come from the saved official benchmark-compatible reports.

## 1. Executive Takeaway

The best signal in this system is still dense retrieval. GraphRAG is weak as a standalone retriever, but the hybrid pipeline improves document-level ranking and the latest hardened full run materially improves citation safety. The main gap remains article-level retrieval quality and citation faithfulness.

## 2. Core Retrieval Results

| System | Scope | doc_recall@10 | art_recall@10 | doc_mrr | art_mrr | Note |
|---|---|---:|---:|---:|---:|---|
| Dense retriever | retrieval suite | 0.8219 | 0.3136 | 0.6894 | 0.2470 | Best single retriever on article recall@10 |
| BM25 + Dense + GraphRAG | retrieval suite | 0.8431 | 0.3126 | 0.6910 | 0.2357 | Best hybrid article recall@10 in the retrieval suite |
| Full hybrid | retrieval suite | 0.8358 | 0.2985 | 0.6920 | 0.2224 | Strong document ranking, weaker article ranking than dense-only |
| Strict current records | 120-question strict subset | 0.7833 | 0.2900 | 0.6774 | 0.3116 | Best strict article MRR seen in the saved subset |
| Hardened full run | full benchmark | 0.7384 | 0.2484 | 0.6351 | 0.1845 | Latest production-style run with no local LLM paths |

## 3. Measured Ablation Results

| Configuration | art_recall@10 | art_mrr | What it says |
|---|---:|---:|---|
| retrieval_dense_only | 0.2609 | 0.0825 | Dense is the strongest retrieval base in the node ablations |
| retrieval_full_hybrid | 0.1930 | 0.0682 | Hybrid helps, but not enough to beat dense on article ranking |
| crag_off | 0.1872 | 0.0648 | Baseline corrective RAG without thresholding |
| crag_threshold_060_025 | 0.1838 | 0.0694 | Small MRR gain, slight recall loss |
| crag_threshold_080_040 | 0.1590 | 0.0653 | Too strict; hurts recall and MRR |
| reranker_off_global | 0.1585 | 0.0608 | Reranker-off remains competitive |
| reranker_on_global | 0.1510 | 0.0594 | Global reranking did not improve retrieval here |
| temporal_forced_on | 0.1991 | 0.0711 | Temporal helps on its target subset, not overall |
| adu_baseline_no_rerank | 0.1202 | 0.0391 | ADU path is not helping retrieval quality |
| citeguard_strict | 0.1333 | 0.0483 | CiteGuard affects safety more than ranking |
| generator_aigrid_qwen30b_llm | 0.1920 | 0.0903 | Best generator-side ablation on retrieval-adjacent metrics |

## 4. Safety, Citation, and Answer Quality

| System | citation_f1 | P@1 | MRR | HCR | JIR | Notes |
|---|---:|---:|---:|---:|---:|---|
| Hardened full run | 0.2102 | 0.5820 | 0.6492 | 0.6680 | 0.1000 | Best current production-style safety profile |
| Best saved full RAG pipeline | 0.1124 | 0.5861 | 0.6885 | 0.8238 | 0.8500 | Strong ranking, weak safety on unanswerable questions |
| Qwen30B RAG generator subset | 0.1384 | 0.6600 | 0.7453 | 0.7700 | 0.3333 | Best subset generator run on ranking and answer quality |
| Gemini 2.5 Flash direct baseline | 0.1865 | 0.2541 | 0.2541 | 0.0082 | 0.3750 | Best direct citation F1 reference |
| Llama 3.3 70B direct baseline | 0.1001 | 0.6270 | 0.6557 | 0.8033 | 0.3500 | Best direct P@1/MRR reference |

## 5. Temporal Subset

Temporal retrieval is the strongest small-sample result, but it is only measured on 7 questions and should not be treated as a global system score.

| System | doc_recall@10 | art_recall@10 | doc_mrr | art_mrr | citation_f1 |
|---|---:|---:|---:|---:|---:|
| tf-on-smoke | 1.0000 | 0.7143 | 1.0000 | 0.6429 | 0.0000 |

## 6. What Matters For Architecture

- Dense retrieval is the best current anchor for both recall and MRR.
- GraphRAG alone is not a reliable retriever; it should be treated as a boost or repair path, not a replacement.
- Hybrid retrieval improves document ranking more consistently than article ranking.
- The latest hardened run shows that citation safety can be improved without collapsing ranking, but unanswerable handling remains the biggest gap.
- Temporal retrieval has strong local signal on temporal questions and should stay conditional, not always-on.
- ADU and aggressive reranking do not currently justify their latency cost in the measured runs.

## 7. Bottom Line

If the next architecture revision only changes one thing, improve article-level retrieval and citation grounding around the dense retriever instead of expanding GraphRAG as a standalone search path. The measured data already points to dense retrieval plus selective boosting as the most defensible direction.

## 8. Source Artifacts

- `code/results/tables/retrieval_suite_latest.md`
- `code/results/tables/strict_pipeline_metrics.md`
- `code/results/tables/hardened_vs_baselines.md`
- `code/results/analysis/direct_vs_rag_same_evaluator.md`
- `code/results/analysis/node_ablation/summary_rollup.md`
- `code/results/tables/temporal_evaluation.md`
