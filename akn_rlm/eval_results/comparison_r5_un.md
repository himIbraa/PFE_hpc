# AlgerianLegalBench — Phase 1 Comparison

Generated: 2026-05-09 11:50:37

## Runs included

| Pipeline | run_id | kind | n | Path |
| --- | --- | --- | --- | --- |
| BM25 | `baseline_bm25_un_full` | full | 40 | `D:\TRY_AGAIN\akn_rlm\eval_results\baseline_bm25_un_full` |
| Dense | `baseline_dense_un_full` | full | 40 | `D:\TRY_AGAIN\akn_rlm\eval_results\baseline_dense_un_full` |
| Hybrid (RRF) | `baseline_hybrid_un_full` | full | 40 | `D:\TRY_AGAIN\akn_rlm\eval_results\baseline_hybrid_un_full` |
| Hybrid+Rerank | `baseline_hybrid_rerank_un_full` | full | 40 | `D:\TRY_AGAIN\akn_rlm\eval_results\baseline_hybrid_rerank_un_full` |
| KG (SPARQL) | `baseline_kg_un_full` | full | 40 | `D:\TRY_AGAIN\akn_rlm\eval_results\baseline_kg_un_full` |
| KG+Hybrid | `baseline_kg_hybrid_un_full` | full | 40 | `D:\TRY_AGAIN\akn_rlm\eval_results\baseline_kg_hybrid_un_full` |
| RLM | `rlm_unanswerable_smoke` | smoke | 40 | `D:\TRY_AGAIN\akn_rlm\eval_results\rlm_unanswerable_smoke` |

## Headline — Cite F1 by query type

Article-level Citation F1 is the headline thesis metric. The RLM target
is to beat the best baseline on the hard types (`multi_hop`,
`temporal_factual`, `conceptual_definitional`, `unanswerable`).

| Query type | BM25 | Dense | Hybrid (RRF) | Hybrid+Rerank | KG (SPARQL) | KG+Hybrid | RLM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exact_article | — | — | — | — | — | — | — |
| rule_application | — | — | — | — | — | — | — |
| multi_hop | — | — | — | — | — | — | — |
| temporal_factual | — | — | — | — | — | — | — |
| conceptual_definitional | — | — | — | — | — | — | — |
| unanswerable | 0.000 | 0.000 | 0.008 | 0.008 | 0.033 | 0.017 | 0.525 |
| layman | — | — | — | — | — | — | — |
| long_context | — | — | — | — | — | — | — |
| **overall** | 0.000 | 0.000 | 0.008 | 0.008 | 0.033 | 0.017 | 0.525 |

## Overall metrics

| Pipeline | run_id | n | MRR@10 doc | MRR@10 art | Cite F1 | Doc Cite F1 | R@10 art | HCR↓ | JIR↓ | Abst F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | `baseline_bm25_un_full` | 40 | 0.360 | 0.000 | 0.000 | 0.138 | 0.000 | 0.000 | 0.000 | 0.000 |
| Dense | `baseline_dense_un_full` | 40 | 0.303 | 0.000 | 0.000 | 0.148 | 0.000 | 0.000 | 0.000 | 0.000 |
| Hybrid (RRF) | `baseline_hybrid_un_full` | 40 | 0.355 | 0.006 | 0.008 | 0.137 | 0.025 | 0.000 | 0.000 | 0.000 |
| Hybrid+Rerank | `baseline_hybrid_rerank_un_full` | 40 | 0.352 | 0.006 | 0.008 | 0.129 | 0.025 | 0.000 | 0.000 | 0.000 |
| KG (SPARQL) | `baseline_kg_un_full` | 40 | 0.067 | 0.013 | 0.033 | 0.068 | 0.025 | 0.000 | 0.000 | 0.140 |
| KG+Hybrid | `baseline_kg_hybrid_un_full` | 40 | 0.220 | 0.013 | 0.017 | 0.113 | 0.050 | 0.000 | 0.000 | 0.000 |
| RLM | `rlm_unanswerable_smoke` | 40 | 0.000 | 0.000 | 0.525 | 0.525 | 0.000 | 0.000 | 0.000 | 1.000 |

## Per query type

### unanswerable

| Pipeline | n | MRR@10 doc | MRR@10 art | Cite F1 | Doc Cite F1 | R@10 art | HCR↓ | JIR↓ | Abst F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 40 | 0.360 | 0.000 | 0.000 | 0.138 | 0.000 | 0.000 | 0.000 | 0.000 |
| Dense | 40 | 0.303 | 0.000 | 0.000 | 0.148 | 0.000 | 0.000 | 0.000 | 0.000 |
| Hybrid (RRF) | 40 | 0.355 | 0.006 | 0.008 | 0.137 | 0.025 | 0.000 | 0.000 | 0.000 |
| Hybrid+Rerank | 40 | 0.352 | 0.006 | 0.008 | 0.129 | 0.025 | 0.000 | 0.000 | 0.000 |
| KG (SPARQL) | 40 | 0.067 | 0.013 | 0.033 | 0.068 | 0.025 | 0.000 | 0.000 | 0.140 |
| KG+Hybrid | 40 | 0.220 | 0.013 | 0.017 | 0.113 | 0.050 | 0.000 | 0.000 | 0.000 |
| RLM | 40 | 0.000 | 0.000 | 0.525 | 0.525 | 0.000 | 0.000 | 0.000 | 1.000 |


_HCR↓ / JIR↓ — lower is better. Deterministic baselines (B1–B6)
have HCR=JIR=0 by construction (no LLM in the loop)._