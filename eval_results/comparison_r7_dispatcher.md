# AlgerianLegalBench — Phase 1 Comparison

Generated: 2026-05-09 18:11:46

## Runs included

| Pipeline | run_id | kind | n | Path |
| --- | --- | --- | --- | --- |
| BM25 | `baseline_bm25_smoke` | smoke | 16 | `D:\TRY_AGAIN\akn_rlm\eval_results\baseline_bm25_smoke` |
| Dense | `baseline_dense_smoke` | smoke | 16 | `D:\TRY_AGAIN\eval_results\baseline_dense_smoke` |
| Hybrid (RRF) | `baseline_hybrid_smoke` | smoke | 16 | `D:\TRY_AGAIN\eval_results\baseline_hybrid_smoke` |
| Hybrid+Rerank | `baseline_hybrid_rerank_smoke` | smoke | 16 | `D:\TRY_AGAIN\eval_results\baseline_hybrid_rerank_smoke` |
| KG (SPARQL) | `baseline_kg_smoke` | smoke | 16 | `D:\TRY_AGAIN\akn_rlm\eval_results\baseline_kg_smoke` |
| KG+Hybrid | `baseline_kg_hybrid_smoke` | smoke | 16 | `D:\TRY_AGAIN\eval_results\baseline_kg_hybrid_smoke` |
| RLM | `rlm_dispatched_smoke` | smoke | 16 | `D:\TRY_AGAIN\eval_results\rlm_dispatched_smoke` |

## Headline — Cite F1 by query type

Article-level Citation F1 is the headline thesis metric. The RLM target
is to beat the best baseline on the hard types (`multi_hop`,
`temporal_factual`, `conceptual_definitional`, `unanswerable`).

| Query type | BM25 | Dense | Hybrid (RRF) | Hybrid+Rerank | KG (SPARQL) | KG+Hybrid | RLM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exact_article | 0.333 | 0.310 | 0.518 | 0.268 | 0.000 | 0.393 | 0.533 |
| rule_application | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| multi_hop | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| temporal_factual | 0.167 | 0.167 | 0.167 | 0.000 | 0.000 | 0.333 | 0.333 |
| conceptual_definitional | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.167 | 0.000 |
| unanswerable | 0.000 | 0.000 | 0.167 | 0.167 | 0.000 | 0.167 | 0.000 |
| layman | 0.000 | 0.167 | 0.167 | 0.167 | 0.000 | 0.167 | 0.900 |
| long_context | 0.111 | 0.000 | 0.200 | 0.100 | 0.100 | 0.100 | 0.133 |
| **overall** | 0.076 | 0.080 | 0.152 | 0.088 | 0.013 | 0.166 | 0.237 |

## Overall metrics

| Pipeline | run_id | n | MRR@10 doc | MRR@10 art | Cite F1 | Doc Cite F1 | R@10 art | HCR↓ | JIR↓ | Abst F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | `baseline_bm25_smoke` | 16 | 0.552 | 0.203 | 0.076 | 0.433 | 0.127 | 0.000 | 0.000 | 0.000 |
| Dense | `baseline_dense_smoke` | 16 | 0.651 | 0.203 | 0.080 | 0.385 | 0.177 | 0.000 | 0.062 | 0.000 |
| Hybrid (RRF) | `baseline_hybrid_smoke` | 16 | 0.797 | 0.297 | 0.152 | 0.465 | 0.306 | 0.000 | 0.000 | 0.000 |
| Hybrid+Rerank | `baseline_hybrid_rerank_smoke` | 16 | 0.620 | 0.266 | 0.088 | 0.442 | 0.190 | 0.000 | 0.000 | 0.000 |
| KG (SPARQL) | `baseline_kg_smoke` | 16 | 0.156 | 0.031 | 0.013 | 0.156 | 0.013 | 0.000 | 0.000 | 0.000 |
| KG+Hybrid | `baseline_kg_hybrid_smoke` | 16 | 0.536 | 0.223 | 0.166 | 0.425 | 0.398 | 0.000 | 0.000 | 0.000 |
| RLM | `rlm_dispatched_smoke` | 16 | 0.656 | 0.356 | 0.237 | 0.531 | 0.327 | 0.000 | 0.000 | 0.800 |

## Per query type

### exact_article

| Pipeline | n | MRR@10 doc | MRR@10 art | Cite F1 | Doc Cite F1 | R@10 art | HCR↓ | JIR↓ | Abst F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 2 | 1.000 | 1.000 | 0.333 | 0.667 | 0.417 | 0.000 | 0.000 | 0.000 |
| Dense | 2 | 0.667 | 0.625 | 0.310 | 0.450 | 0.417 | 0.000 | 0.000 | 0.000 |
| Hybrid (RRF) | 2 | 1.000 | 1.000 | 0.518 | 0.667 | 0.750 | 0.000 | 0.000 | 0.000 |
| Hybrid+Rerank | 2 | 1.000 | 1.000 | 0.268 | 0.500 | 0.417 | 0.000 | 0.000 | 0.000 |
| KG (SPARQL) | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| KG+Hybrid | 2 | 1.000 | 1.000 | 0.393 | 0.583 | 0.583 | 0.000 | 0.000 | 0.000 |
| RLM | 2 | 1.000 | 1.000 | 0.533 | 1.000 | 0.417 | 0.000 | 0.000 | 0.000 |

### rule_application

| Pipeline | n | MRR@10 doc | MRR@10 art | Cite F1 | Doc Cite F1 | R@10 art | HCR↓ | JIR↓ | Abst F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 2 | 0.125 | 0.000 | 0.000 | 0.167 | 0.000 | 0.000 | 0.000 | 0.000 |
| Dense | 2 | 0.167 | 0.000 | 0.000 | 0.250 | 0.000 | 0.000 | 0.000 | 0.000 |
| Hybrid (RRF) | 2 | 0.500 | 0.000 | 0.000 | 0.167 | 0.000 | 0.000 | 0.000 | 0.000 |
| Hybrid+Rerank | 2 | 0.250 | 0.000 | 0.000 | 0.200 | 0.000 | 0.000 | 0.000 | 0.000 |
| KG (SPARQL) | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| KG+Hybrid | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| RLM | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

### multi_hop

| Pipeline | n | MRR@10 doc | MRR@10 art | Cite F1 | Doc Cite F1 | R@10 art | HCR↓ | JIR↓ | Abst F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 2 | 0.667 | 0.000 | 0.000 | 0.367 | 0.000 | 0.000 | 0.000 | 0.000 |
| Dense | 2 | 0.750 | 0.000 | 0.000 | 0.533 | 0.000 | 0.000 | 0.000 | 0.000 |
| Hybrid (RRF) | 2 | 0.500 | 0.000 | 0.000 | 0.333 | 0.000 | 0.000 | 0.000 | 0.000 |
| Hybrid+Rerank | 2 | 1.000 | 0.000 | 0.000 | 0.533 | 0.000 | 0.000 | 0.000 | 0.000 |
| KG (SPARQL) | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| KG+Hybrid | 2 | 0.667 | 0.000 | 0.000 | 0.367 | 0.000 | 0.000 | 0.000 | 0.000 |
| RLM | 2 | 0.500 | 0.000 | 0.000 | 0.333 | 0.000 | 0.000 | 0.000 | 0.000 |

### temporal_factual

| Pipeline | n | MRR@10 doc | MRR@10 art | Cite F1 | Doc Cite F1 | R@10 art | HCR↓ | JIR↓ | Abst F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 2 | 0.292 | 0.125 | 0.167 | 0.367 | 0.500 | 0.000 | 0.000 | 0.000 |
| Dense | 2 | 0.625 | 0.500 | 0.167 | 0.333 | 0.500 | 0.000 | 0.000 | 0.000 |
| Hybrid (RRF) | 2 | 0.625 | 0.500 | 0.167 | 0.367 | 0.500 | 0.000 | 0.000 | 0.000 |
| Hybrid+Rerank | 2 | 0.167 | 0.000 | 0.000 | 0.167 | 0.000 | 0.000 | 0.000 | 0.000 |
| KG (SPARQL) | 2 | 0.250 | 0.000 | 0.000 | 0.333 | 0.000 | 0.000 | 0.000 | 0.000 |
| KG+Hybrid | 2 | 0.333 | 0.333 | 0.333 | 0.367 | 1.000 | 0.000 | 0.000 | 0.000 |
| RLM | 2 | 0.750 | 0.600 | 0.333 | 0.583 | 1.000 | 0.000 | 0.000 | 0.000 |

### conceptual_definitional

| Pipeline | n | MRR@10 doc | MRR@10 art | Cite F1 | Doc Cite F1 | R@10 art | HCR↓ | JIR↓ | Abst F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 2 | 0.667 | 0.000 | 0.000 | 0.583 | 0.000 | 0.000 | 0.000 | 0.000 |
| Dense | 2 | 0.500 | 0.000 | 0.000 | 0.200 | 0.000 | 0.000 | 0.500 | 0.000 |
| Hybrid (RRF) | 2 | 0.750 | 0.000 | 0.000 | 0.450 | 0.000 | 0.000 | 0.000 | 0.000 |
| Hybrid+Rerank | 2 | 0.667 | 0.000 | 0.000 | 0.533 | 0.000 | 0.000 | 0.000 | 0.000 |
| KG (SPARQL) | 2 | 0.750 | 0.000 | 0.000 | 0.667 | 0.000 | 0.000 | 0.000 | 0.000 |
| KG+Hybrid | 2 | 0.667 | 0.100 | 0.167 | 0.583 | 0.500 | 0.000 | 0.000 | 0.000 |
| RLM | 2 | 1.000 | 0.000 | 0.000 | 0.750 | 0.000 | 0.000 | 0.000 | 0.000 |

### unanswerable

| Pipeline | n | MRR@10 doc | MRR@10 art | Cite F1 | Doc Cite F1 | R@10 art | HCR↓ | JIR↓ | Abst F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 2 | 0.500 | 0.000 | 0.000 | 0.583 | 0.000 | 0.000 | 0.000 | 0.000 |
| Dense | 2 | 0.750 | 0.000 | 0.000 | 0.450 | 0.000 | 0.000 | 0.000 | 0.000 |
| Hybrid (RRF) | 2 | 1.000 | 0.125 | 0.167 | 0.583 | 0.500 | 0.000 | 0.000 | 0.000 |
| Hybrid+Rerank | 2 | 0.250 | 0.125 | 0.167 | 0.367 | 0.500 | 0.000 | 0.000 | 0.000 |
| KG (SPARQL) | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| KG+Hybrid | 2 | 0.750 | 0.125 | 0.167 | 0.667 | 0.500 | 0.000 | 0.000 | 0.000 |
| RLM | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1.000 |

### layman

| Pipeline | n | MRR@10 doc | MRR@10 art | Cite F1 | Doc Cite F1 | R@10 art | HCR↓ | JIR↓ | Abst F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 2 | 0.500 | 0.000 | 0.000 | 0.200 | 0.000 | 0.000 | 0.000 | 0.000 |
| Dense | 2 | 0.750 | 0.500 | 0.167 | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 |
| Hybrid (RRF) | 2 | 1.000 | 0.500 | 0.167 | 0.450 | 0.500 | 0.000 | 0.000 | 0.000 |
| Hybrid+Rerank | 2 | 1.000 | 0.500 | 0.167 | 0.700 | 0.500 | 0.000 | 0.000 | 0.000 |
| KG (SPARQL) | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| KG+Hybrid | 2 | 0.375 | 0.125 | 0.167 | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 |
| RLM | 2 | 1.000 | 1.000 | 0.900 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 |

### long_context

| Pipeline | n | MRR@10 doc | MRR@10 art | Cite F1 | Doc Cite F1 | R@10 art | HCR↓ | JIR↓ | Abst F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 2 | 0.667 | 0.500 | 0.111 | 0.533 | 0.100 | 0.000 | 0.000 | 0.000 |
| Dense | 2 | 1.000 | 0.000 | 0.000 | 0.367 | 0.000 | 0.000 | 0.000 | 0.000 |
| Hybrid (RRF) | 2 | 1.000 | 0.250 | 0.200 | 0.700 | 0.200 | 0.000 | 0.000 | 0.000 |
| Hybrid+Rerank | 2 | 0.625 | 0.500 | 0.100 | 0.533 | 0.100 | 0.000 | 0.000 | 0.000 |
| KG (SPARQL) | 2 | 0.250 | 0.250 | 0.100 | 0.250 | 0.100 | 0.000 | 0.000 | 0.000 |
| KG+Hybrid | 2 | 0.500 | 0.100 | 0.100 | 0.333 | 0.100 | 0.000 | 0.000 | 0.000 |
| RLM | 2 | 1.000 | 0.250 | 0.133 | 0.583 | 0.200 | 0.000 | 0.000 | 0.000 |


_HCR↓ / JIR↓ — lower is better. Deterministic baselines (B1–B6)
have HCR=JIR=0 by construction (no LLM in the loop)._