# AlgerianLegalBench — Phase 1 Comparison

Generated: 2026-05-08 18:28:28

## Runs included

| Pipeline | run_id | kind | n | Path |
| --- | --- | --- | --- | --- |
| BM25 | `baseline_bm25_smoke` | smoke | 16 | `D:\TRY_AGAIN\akn_rlm\eval_results\baseline_bm25_smoke` |
| Dense | `baseline_dense_smoke` | smoke | 16 | `D:\TRY_AGAIN\eval_results\baseline_dense_smoke` |
| Hybrid (RRF) | `baseline_hybrid_strat5` | smoke | 40 | `D:\TRY_AGAIN\eval_results\baseline_hybrid_strat5` |
| Hybrid+Rerank | `baseline_hybrid_rerank_strat5` | smoke | 40 | `D:\TRY_AGAIN\eval_results\baseline_hybrid_rerank_strat5` |
| KG (SPARQL) | `baseline_kg_smoke` | smoke | 16 | `D:\TRY_AGAIN\akn_rlm\eval_results\baseline_kg_smoke` |
| KG+Hybrid | `baseline_kg_hybrid_smoke` | smoke | 16 | `D:\TRY_AGAIN\eval_results\baseline_kg_hybrid_smoke` |
| RLM | `phase0_smoke2` | smoke | 10 | `D:\TRY_AGAIN\akn_rlm\eval_results\phase0_smoke2` |

## Headline — Cite F1 by query type

Article-level Citation F1 is the headline thesis metric. The RLM target
is to beat the best baseline on the hard types (`multi_hop`,
`temporal_factual`, `conceptual_definitional`, `unanswerable`).

| Query type | BM25 | Dense | Hybrid (RRF) | Hybrid+Rerank | KG (SPARQL) | KG+Hybrid | RLM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exact_article | 0.333 | 0.310 | 0.264 | 0.221 | 0.000 | 0.393 | 0.222 |
| rule_application | 0.000 | 0.000 | 0.214 | 0.157 | 0.000 | 0.000 | 0.095 |
| multi_hop | 0.000 | 0.000 | 0.050 | 0.050 | 0.000 | 0.000 | — |
| temporal_factual | 0.167 | 0.167 | 0.133 | 0.067 | 0.000 | 0.333 | — |
| conceptual_definitional | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.167 | — |
| unanswerable | 0.000 | 0.000 | 0.067 | 0.067 | 0.000 | 0.167 | — |
| layman | 0.000 | 0.167 | 0.067 | 0.067 | 0.000 | 0.167 | — |
| long_context | 0.111 | 0.000 | 0.120 | 0.120 | 0.100 | 0.100 | — |
| **overall** | 0.076 | 0.080 | 0.114 | 0.094 | 0.013 | 0.166 | 0.133 |

## Overall metrics

| Pipeline | run_id | n | MRR@10 doc | MRR@10 art | Cite F1 | Doc Cite F1 | R@10 art | HCR↓ | JIR↓ | Abst F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | `baseline_bm25_smoke` | 16 | 0.552 | 0.203 | 0.076 | 0.433 | 0.127 | 0.000 | 0.000 | 0.000 |
| Dense | `baseline_dense_smoke` | 16 | 0.651 | 0.203 | 0.080 | 0.385 | 0.177 | 0.000 | 0.062 | 0.000 |
| Hybrid (RRF) | `baseline_hybrid_strat5` | 40 | 0.670 | 0.216 | 0.114 | 0.386 | 0.215 | 0.000 | 0.000 | 0.000 |
| Hybrid+Rerank | `baseline_hybrid_rerank_strat5` | 40 | 0.573 | 0.226 | 0.094 | 0.374 | 0.173 | 0.000 | 0.050 | 0.000 |
| KG (SPARQL) | `baseline_kg_smoke` | 16 | 0.156 | 0.031 | 0.013 | 0.156 | 0.013 | 0.000 | 0.000 | 0.000 |
| KG+Hybrid | `baseline_kg_hybrid_smoke` | 16 | 0.536 | 0.223 | 0.166 | 0.425 | 0.398 | 0.000 | 0.000 | 0.000 |
| RLM | `phase0_smoke2` | 10 | 0.300 | 0.200 | 0.133 | 0.300 | 0.100 | 0.300 | 0.000 | 0.400 |

## Per query type

### exact_article

| Pipeline | n | MRR@10 doc | MRR@10 art | Cite F1 | Doc Cite F1 | R@10 art | HCR↓ | JIR↓ | Abst F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 2 | 1.000 | 1.000 | 0.333 | 0.667 | 0.417 | 0.000 | 0.000 | 0.000 |
| Dense | 2 | 0.667 | 0.625 | 0.310 | 0.450 | 0.417 | 0.000 | 0.000 | 0.000 |
| Hybrid (RRF) | 5 | 0.667 | 0.600 | 0.264 | 0.427 | 0.400 | 0.000 | 0.000 | 0.000 |
| Hybrid+Rerank | 5 | 0.867 | 0.600 | 0.221 | 0.567 | 0.367 | 0.000 | 0.000 | 0.000 |
| KG (SPARQL) | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| KG+Hybrid | 2 | 1.000 | 1.000 | 0.393 | 0.583 | 0.583 | 0.000 | 0.000 | 0.000 |
| RLM | 3 | 0.333 | 0.333 | 0.222 | 0.333 | 0.167 | 0.333 | 0.000 | 0.500 |

### rule_application

| Pipeline | n | MRR@10 doc | MRR@10 art | Cite F1 | Doc Cite F1 | R@10 art | HCR↓ | JIR↓ | Abst F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 2 | 0.125 | 0.000 | 0.000 | 0.167 | 0.000 | 0.000 | 0.000 | 0.000 |
| Dense | 2 | 0.167 | 0.000 | 0.000 | 0.250 | 0.000 | 0.000 | 0.000 | 0.000 |
| Hybrid (RRF) | 5 | 0.800 | 0.300 | 0.214 | 0.333 | 0.333 | 0.000 | 0.000 | 0.000 |
| Hybrid+Rerank | 5 | 0.700 | 0.400 | 0.157 | 0.347 | 0.233 | 0.000 | 0.000 | 0.000 |
| KG (SPARQL) | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| KG+Hybrid | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| RLM | 7 | 0.286 | 0.143 | 0.095 | 0.286 | 0.071 | 0.286 | 0.000 | 0.333 |

### multi_hop

| Pipeline | n | MRR@10 doc | MRR@10 art | Cite F1 | Doc Cite F1 | R@10 art | HCR↓ | JIR↓ | Abst F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 2 | 0.667 | 0.000 | 0.000 | 0.367 | 0.000 | 0.000 | 0.000 | 0.000 |
| Dense | 2 | 0.750 | 0.000 | 0.000 | 0.533 | 0.000 | 0.000 | 0.000 | 0.000 |
| Hybrid (RRF) | 5 | 0.500 | 0.200 | 0.050 | 0.400 | 0.067 | 0.000 | 0.000 | 0.000 |
| Hybrid+Rerank | 5 | 0.667 | 0.067 | 0.050 | 0.473 | 0.067 | 0.000 | 0.000 | 0.000 |
| KG (SPARQL) | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| KG+Hybrid | 2 | 0.667 | 0.000 | 0.000 | 0.367 | 0.000 | 0.000 | 0.000 | 0.000 |
| RLM | 0 | — | — | — | — | — | — | — | — |

### temporal_factual

| Pipeline | n | MRR@10 doc | MRR@10 art | Cite F1 | Doc Cite F1 | R@10 art | HCR↓ | JIR↓ | Abst F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 2 | 0.292 | 0.125 | 0.167 | 0.367 | 0.500 | 0.000 | 0.000 | 0.000 |
| Dense | 2 | 0.625 | 0.500 | 0.167 | 0.333 | 0.500 | 0.000 | 0.000 | 0.000 |
| Hybrid (RRF) | 5 | 0.590 | 0.240 | 0.133 | 0.373 | 0.400 | 0.000 | 0.000 | 0.000 |
| Hybrid+Rerank | 5 | 0.317 | 0.040 | 0.067 | 0.280 | 0.200 | 0.000 | 0.000 | 0.000 |
| KG (SPARQL) | 2 | 0.250 | 0.000 | 0.000 | 0.333 | 0.000 | 0.000 | 0.000 | 0.000 |
| KG+Hybrid | 2 | 0.333 | 0.333 | 0.333 | 0.367 | 1.000 | 0.000 | 0.000 | 0.000 |
| RLM | 0 | — | — | — | — | — | — | — | — |

### conceptual_definitional

| Pipeline | n | MRR@10 doc | MRR@10 art | Cite F1 | Doc Cite F1 | R@10 art | HCR↓ | JIR↓ | Abst F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 2 | 0.667 | 0.000 | 0.000 | 0.583 | 0.000 | 0.000 | 0.000 | 0.000 |
| Dense | 2 | 0.500 | 0.000 | 0.000 | 0.200 | 0.000 | 0.000 | 0.500 | 0.000 |
| Hybrid (RRF) | 5 | 0.600 | 0.000 | 0.000 | 0.413 | 0.000 | 0.000 | 0.000 | 0.000 |
| Hybrid+Rerank | 5 | 0.367 | 0.000 | 0.000 | 0.293 | 0.000 | 0.000 | 0.000 | 0.000 |
| KG (SPARQL) | 2 | 0.750 | 0.000 | 0.000 | 0.667 | 0.000 | 0.000 | 0.000 | 0.000 |
| KG+Hybrid | 2 | 0.667 | 0.100 | 0.167 | 0.583 | 0.500 | 0.000 | 0.000 | 0.000 |
| RLM | 0 | — | — | — | — | — | — | — | — |

### unanswerable

| Pipeline | n | MRR@10 doc | MRR@10 art | Cite F1 | Doc Cite F1 | R@10 art | HCR↓ | JIR↓ | Abst F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 2 | 0.500 | 0.000 | 0.000 | 0.583 | 0.000 | 0.000 | 0.000 | 0.000 |
| Dense | 2 | 0.750 | 0.000 | 0.000 | 0.450 | 0.000 | 0.000 | 0.000 | 0.000 |
| Hybrid (RRF) | 5 | 0.600 | 0.050 | 0.067 | 0.233 | 0.200 | 0.000 | 0.000 | 0.000 |
| Hybrid+Rerank | 5 | 0.300 | 0.050 | 0.067 | 0.147 | 0.200 | 0.000 | 0.000 | 0.000 |
| KG (SPARQL) | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| KG+Hybrid | 2 | 0.750 | 0.125 | 0.167 | 0.667 | 0.500 | 0.000 | 0.000 | 0.000 |
| RLM | 0 | — | — | — | — | — | — | — | — |

### layman

| Pipeline | n | MRR@10 doc | MRR@10 art | Cite F1 | Doc Cite F1 | R@10 art | HCR↓ | JIR↓ | Abst F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 2 | 0.500 | 0.000 | 0.000 | 0.200 | 0.000 | 0.000 | 0.000 | 0.000 |
| Dense | 2 | 0.750 | 0.500 | 0.167 | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 |
| Hybrid (RRF) | 5 | 0.800 | 0.200 | 0.067 | 0.280 | 0.200 | 0.000 | 0.000 | 0.000 |
| Hybrid+Rerank | 5 | 0.650 | 0.200 | 0.067 | 0.360 | 0.200 | 0.000 | 0.200 | 0.000 |
| KG (SPARQL) | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| KG+Hybrid | 2 | 0.375 | 0.125 | 0.167 | 0.500 | 0.500 | 0.000 | 0.000 | 0.000 |
| RLM | 0 | — | — | — | — | — | — | — | — |

### long_context

| Pipeline | n | MRR@10 doc | MRR@10 art | Cite F1 | Doc Cite F1 | R@10 art | HCR↓ | JIR↓ | Abst F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 2 | 0.667 | 0.500 | 0.111 | 0.533 | 0.100 | 0.000 | 0.000 | 0.000 |
| Dense | 2 | 1.000 | 0.000 | 0.000 | 0.367 | 0.000 | 0.000 | 0.000 | 0.000 |
| Hybrid (RRF) | 5 | 0.800 | 0.140 | 0.120 | 0.627 | 0.120 | 0.000 | 0.000 | 0.000 |
| Hybrid+Rerank | 5 | 0.717 | 0.450 | 0.120 | 0.527 | 0.120 | 0.000 | 0.200 | 0.000 |
| KG (SPARQL) | 2 | 0.250 | 0.250 | 0.100 | 0.250 | 0.100 | 0.000 | 0.000 | 0.000 |
| KG+Hybrid | 2 | 0.500 | 0.100 | 0.100 | 0.333 | 0.100 | 0.000 | 0.000 | 0.000 |
| RLM | 0 | — | — | — | — | — | — | — | — |


_HCR↓ / JIR↓ — lower is better. Deterministic baselines (B1–B6)
have HCR=JIR=0 by construction (no LLM in the loop)._