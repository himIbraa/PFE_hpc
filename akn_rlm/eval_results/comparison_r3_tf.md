# AlgerianLegalBench — Phase 1 Comparison

Generated: 2026-05-08 23:46:06

## Runs included

| Pipeline | run_id | kind | n | Path |
| --- | --- | --- | --- | --- |
| BM25 | `baseline_bm25_smoke` | smoke | 16 | `D:\TRY_AGAIN\akn_rlm\eval_results\baseline_bm25_smoke` |
| Dense | `baseline_dense_smoke` | smoke | 16 | `D:\TRY_AGAIN\eval_results\baseline_dense_smoke` |
| Hybrid (RRF) | `baseline_hybrid_tf_full` | full | 7 | `D:\TRY_AGAIN\akn_rlm\eval_results\baseline_hybrid_tf_full` |
| Hybrid+Rerank | `baseline_hybrid_rerank_smoke` | smoke | 16 | `D:\TRY_AGAIN\eval_results\baseline_hybrid_rerank_smoke` |
| KG (SPARQL) | `baseline_kg_tf_full` | full | 7 | `D:\TRY_AGAIN\akn_rlm\eval_results\baseline_kg_tf_full` |
| KG+Hybrid | `baseline_kg_hybrid_tf_full` | full | 7 | `D:\TRY_AGAIN\akn_rlm\eval_results\baseline_kg_hybrid_tf_full` |
| RLM | `rlm_temporal_factual_default` | full | 7 | `D:\TRY_AGAIN\akn_rlm\eval_results\rlm_temporal_factual_default` |

## Headline — Cite F1 by query type

Article-level Citation F1 is the headline thesis metric. The RLM target
is to beat the best baseline on the hard types (`multi_hop`,
`temporal_factual`, `conceptual_definitional`, `unanswerable`).

| Query type | BM25 | Dense | Hybrid (RRF) | Hybrid+Rerank | KG (SPARQL) | KG+Hybrid | RLM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exact_article | 0.333 | 0.310 | — | 0.268 | — | — | — |
| rule_application | 0.000 | 0.000 | — | 0.000 | — | — | — |
| multi_hop | 0.000 | 0.000 | — | 0.000 | — | — | — |
| temporal_factual | 0.167 | 0.167 | 0.095 | 0.000 | 0.000 | 0.095 | 0.167 |
| conceptual_definitional | 0.000 | 0.000 | — | 0.000 | — | — | — |
| unanswerable | 0.000 | 0.000 | — | 0.167 | — | — | — |
| layman | 0.000 | 0.167 | — | 0.167 | — | — | — |
| long_context | 0.111 | 0.000 | — | 0.100 | — | — | — |
| **overall** | 0.076 | 0.080 | 0.095 | 0.088 | 0.000 | 0.095 | 0.167 |

## Overall metrics

| Pipeline | run_id | n | MRR@10 doc | MRR@10 art | Cite F1 | Doc Cite F1 | R@10 art | HCR↓ | JIR↓ | Abst F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | `baseline_bm25_smoke` | 16 | 0.552 | 0.203 | 0.076 | 0.433 | 0.127 | 0.000 | 0.000 | 0.000 |
| Dense | `baseline_dense_smoke` | 16 | 0.651 | 0.203 | 0.080 | 0.385 | 0.177 | 0.000 | 0.062 | 0.000 |
| Hybrid (RRF) | `baseline_hybrid_tf_full` | 7 | 0.636 | 0.171 | 0.095 | 0.481 | 0.286 | 0.000 | 0.000 | 0.000 |
| Hybrid+Rerank | `baseline_hybrid_rerank_smoke` | 16 | 0.620 | 0.266 | 0.088 | 0.442 | 0.190 | 0.000 | 0.000 | 0.000 |
| KG (SPARQL) | `baseline_kg_tf_full` | 7 | 0.071 | 0.000 | 0.000 | 0.095 | 0.000 | 0.000 | 0.000 | 0.000 |
| KG+Hybrid | `baseline_kg_hybrid_tf_full` | 7 | 0.143 | 0.095 | 0.095 | 0.176 | 0.286 | 0.000 | 0.000 | 0.000 |
| RLM | `rlm_temporal_factual_default` | 7 | 0.786 | 0.243 | 0.167 | 0.619 | 0.429 | 0.000 | 0.143 | 0.000 |

## Per query type

### exact_article

| Pipeline | n | MRR@10 doc | MRR@10 art | Cite F1 | Doc Cite F1 | R@10 art | HCR↓ | JIR↓ | Abst F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 2 | 1.000 | 1.000 | 0.333 | 0.667 | 0.417 | 0.000 | 0.000 | 0.000 |
| Dense | 2 | 0.667 | 0.625 | 0.310 | 0.450 | 0.417 | 0.000 | 0.000 | 0.000 |
| Hybrid (RRF) | 0 | — | — | — | — | — | — | — | — |
| Hybrid+Rerank | 2 | 1.000 | 1.000 | 0.268 | 0.500 | 0.417 | 0.000 | 0.000 | 0.000 |
| KG (SPARQL) | 0 | — | — | — | — | — | — | — | — |
| KG+Hybrid | 0 | — | — | — | — | — | — | — | — |
| RLM | 0 | — | — | — | — | — | — | — | — |

### rule_application

| Pipeline | n | MRR@10 doc | MRR@10 art | Cite F1 | Doc Cite F1 | R@10 art | HCR↓ | JIR↓ | Abst F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 2 | 0.125 | 0.000 | 0.000 | 0.167 | 0.000 | 0.000 | 0.000 | 0.000 |
| Dense | 2 | 0.167 | 0.000 | 0.000 | 0.250 | 0.000 | 0.000 | 0.000 | 0.000 |
| Hybrid (RRF) | 0 | — | — | — | — | — | — | — | — |
| Hybrid+Rerank | 2 | 0.250 | 0.000 | 0.000 | 0.200 | 0.000 | 0.000 | 0.000 | 0.000 |
| KG (SPARQL) | 0 | — | — | — | — | — | — | — | — |
| KG+Hybrid | 0 | — | — | — | — | — | — | — | — |
| RLM | 0 | — | — | — | — | — | — | — | — |

### multi_hop

| Pipeline | n | MRR@10 doc | MRR@10 art | Cite F1 | Doc Cite F1 | R@10 art | HCR↓ | JIR↓ | Abst F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 2 | 0.667 | 0.000 | 0.000 | 0.367 | 0.000 | 0.000 | 0.000 | 0.000 |
| Dense | 2 | 0.750 | 0.000 | 0.000 | 0.533 | 0.000 | 0.000 | 0.000 | 0.000 |
| Hybrid (RRF) | 0 | — | — | — | — | — | — | — | — |
| Hybrid+Rerank | 2 | 1.000 | 0.000 | 0.000 | 0.533 | 0.000 | 0.000 | 0.000 | 0.000 |
| KG (SPARQL) | 0 | — | — | — | — | — | — | — | — |
| KG+Hybrid | 0 | — | — | — | — | — | — | — | — |
| RLM | 0 | — | — | — | — | — | — | — | — |

### temporal_factual

| Pipeline | n | MRR@10 doc | MRR@10 art | Cite F1 | Doc Cite F1 | R@10 art | HCR↓ | JIR↓ | Abst F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 2 | 0.292 | 0.125 | 0.167 | 0.367 | 0.500 | 0.000 | 0.000 | 0.000 |
| Dense | 2 | 0.625 | 0.500 | 0.167 | 0.333 | 0.500 | 0.000 | 0.000 | 0.000 |
| Hybrid (RRF) | 7 | 0.636 | 0.171 | 0.095 | 0.481 | 0.286 | 0.000 | 0.000 | 0.000 |
| Hybrid+Rerank | 2 | 0.167 | 0.000 | 0.000 | 0.167 | 0.000 | 0.000 | 0.000 | 0.000 |
| KG (SPARQL) | 7 | 0.071 | 0.000 | 0.000 | 0.095 | 0.000 | 0.000 | 0.000 | 0.000 |
| KG+Hybrid | 7 | 0.143 | 0.095 | 0.095 | 0.176 | 0.286 | 0.000 | 0.000 | 0.000 |
| RLM | 7 | 0.786 | 0.243 | 0.167 | 0.619 | 0.429 | 0.000 | 0.143 | 0.000 |

### conceptual_definitional

| Pipeline | n | MRR@10 doc | MRR@10 art | Cite F1 | Doc Cite F1 | R@10 art | HCR↓ | JIR↓ | Abst F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 2 | 0.667 | 0.000 | 0.000 | 0.583 | 0.000 | 0.000 | 0.000 | 0.000 |
| Dense | 2 | 0.500 | 0.000 | 0.000 | 0.200 | 0.000 | 0.000 | 0.500 | 0.000 |
| Hybrid (RRF) | 0 | — | — | — | — | — | — | — | — |
| Hybrid+Rerank | 2 | 0.667 | 0.000 | 0.000 | 0.533 | 0.000 | 0.000 | 0.000 | 0.000 |
| KG (SPARQL) | 0 | — | — | — | — | — | — | — | — |
| KG+Hybrid | 0 | — | — | — | — | — | — | — | — |
| RLM | 0 | — | — | — | — | — | — | — | — |

### unanswerable

| Pipeline | n | MRR@10 doc | MRR@10 art | Cite F1 | Doc Cite F1 | R@10 art | HCR↓ | JIR↓ | Abst F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 2 | 0.500 | 0.000 | 0.000 | 0.583 | 0.000 | 0.000 | 0.000 | 0.000 |
| Dense | 2 | 0.750 | 0.000 | 0.000 | 0.450 | 0.000 | 0.000 | 0.000 | 0.000 |
| Hybrid (RRF) | 0 | — | — | — | — | — | — | — | — |
| Hybrid+Rerank | 2 | 0.250 | 0.125 | 0.167 | 0.367 | 0.500 | 0.000 | 0.000 | 0.000 |
| KG (SPARQL) | 0 | — | — | — | — | — | — | — | — |
| KG+Hybrid | 0 | — | — | — | — | — | — | — | — |
| RLM | 0 | — | — | — | — | — | — | — | — |

### layman

| Pipeline | n | MRR@10 doc | MRR@10 art | Cite F1 | Doc Cite F1 | R@10 art | HCR↓ | JIR↓ | Abst F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 2 | 0.500 | 0.000 | 0.000 | 0.200 | 0.000 | 0.000 | 0.000 | 0.000 |
| Dense | 2 | 0.750 | 0.500 | 0.167 | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 |
| Hybrid (RRF) | 0 | — | — | — | — | — | — | — | — |
| Hybrid+Rerank | 2 | 1.000 | 0.500 | 0.167 | 0.700 | 0.500 | 0.000 | 0.000 | 0.000 |
| KG (SPARQL) | 0 | — | — | — | — | — | — | — | — |
| KG+Hybrid | 0 | — | — | — | — | — | — | — | — |
| RLM | 0 | — | — | — | — | — | — | — | — |

### long_context

| Pipeline | n | MRR@10 doc | MRR@10 art | Cite F1 | Doc Cite F1 | R@10 art | HCR↓ | JIR↓ | Abst F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 2 | 0.667 | 0.500 | 0.111 | 0.533 | 0.100 | 0.000 | 0.000 | 0.000 |
| Dense | 2 | 1.000 | 0.000 | 0.000 | 0.367 | 0.000 | 0.000 | 0.000 | 0.000 |
| Hybrid (RRF) | 0 | — | — | — | — | — | — | — | — |
| Hybrid+Rerank | 2 | 0.625 | 0.500 | 0.100 | 0.533 | 0.100 | 0.000 | 0.000 | 0.000 |
| KG (SPARQL) | 0 | — | — | — | — | — | — | — | — |
| KG+Hybrid | 0 | — | — | — | — | — | — | — | — |
| RLM | 0 | — | — | — | — | — | — | — | — |


_HCR↓ / JIR↓ — lower is better. Deterministic baselines (B1–B6)
have HCR=JIR=0 by construction (no LLM in the loop)._