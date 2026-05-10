# AlgerianLegalBench — Phase 1 Comparison

Generated: 2026-05-09 18:44:02

## Runs included

| Pipeline | run_id | kind | n | Path |
| --- | --- | --- | --- | --- |
| BM25 | `baseline_bm25_strat5` | smoke | 40 | `D:\TRY_AGAIN\akn_rlm\eval_results\baseline_bm25_strat5` |
| Dense | `baseline_dense_strat5` | smoke | 40 | `D:\TRY_AGAIN\akn_rlm\eval_results\baseline_dense_strat5` |
| Hybrid (RRF) | `baseline_hybrid_strat5` | smoke | 40 | `D:\TRY_AGAIN\eval_results\baseline_hybrid_strat5` |
| Hybrid+Rerank | `baseline_hybrid_rerank_strat5` | smoke | 40 | `D:\TRY_AGAIN\eval_results\baseline_hybrid_rerank_strat5` |
| KG (SPARQL) | `baseline_kg_strat5` | smoke | 40 | `D:\TRY_AGAIN\akn_rlm\eval_results\baseline_kg_strat5` |
| KG+Hybrid | `baseline_kg_hybrid_strat5` | smoke | 40 | `D:\TRY_AGAIN\akn_rlm\eval_results\baseline_kg_hybrid_strat5` |
| RLM | `rlm_dispatched_strat5` | smoke | 40 | `D:\TRY_AGAIN\akn_rlm\eval_results\rlm_dispatched_strat5` |

## Headline — Cite F1 by query type

Article-level Citation F1 is the headline thesis metric. The RLM target
is to beat the best baseline on the hard types (`multi_hop`,
`temporal_factual`, `conceptual_definitional`, `unanswerable`).

| Query type | BM25 | Dense | Hybrid (RRF) | Hybrid+Rerank | KG (SPARQL) | KG+Hybrid | RLM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exact_article | 0.248 | 0.124 | 0.264 | 0.221 | 0.000 | 0.214 | 0.413 |
| rule_application | 0.157 | 0.000 | 0.214 | 0.157 | 0.000 | 0.164 | 0.267 |
| multi_hop | 0.050 | 0.094 | 0.050 | 0.050 | 0.000 | 0.050 | 0.150 |
| temporal_factual | 0.067 | 0.133 | 0.133 | 0.067 | 0.000 | 0.133 | 0.233 |
| conceptual_definitional | 0.067 | 0.000 | 0.000 | 0.000 | 0.000 | 0.133 | 0.067 |
| unanswerable | 0.000 | 0.000 | 0.067 | 0.067 | 0.000 | 0.067 | 0.600 |
| layman | 0.000 | 0.067 | 0.067 | 0.067 | 0.000 | 0.067 | 0.333 |
| long_context | 0.084 | 0.000 | 0.120 | 0.120 | 0.040 | 0.080 | 0.080 |
| **overall** | 0.084 | 0.052 | 0.114 | 0.094 | 0.005 | 0.114 | 0.268 |

## Overall metrics

| Pipeline | run_id | n | MRR@10 doc | MRR@10 art | Cite F1 | Doc Cite F1 | R@10 art | HCR↓ | JIR↓ | Abst F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | `baseline_bm25_strat5` | 40 | 0.575 | 0.200 | 0.084 | 0.379 | 0.143 | 0.000 | 0.025 | 0.000 |
| Dense | `baseline_dense_strat5` | 40 | 0.510 | 0.144 | 0.052 | 0.299 | 0.110 | 0.000 | 0.025 | 0.000 |
| Hybrid (RRF) | `baseline_hybrid_strat5` | 40 | 0.670 | 0.216 | 0.114 | 0.386 | 0.215 | 0.000 | 0.000 | 0.000 |
| Hybrid+Rerank | `baseline_hybrid_rerank_strat5` | 40 | 0.573 | 0.226 | 0.094 | 0.374 | 0.173 | 0.000 | 0.050 | 0.000 |
| KG (SPARQL) | `baseline_kg_strat5` | 40 | 0.087 | 0.013 | 0.005 | 0.062 | 0.005 | 0.000 | 0.000 | 0.000 |
| KG+Hybrid | `baseline_kg_hybrid_strat5` | 40 | 0.490 | 0.204 | 0.114 | 0.318 | 0.243 | 0.000 | 0.000 | 0.000 |
| RLM | `rlm_dispatched_strat5` | 40 | 0.613 | 0.282 | 0.268 | 0.604 | 0.255 | 0.000 | 0.000 | 0.632 |

## Per query type

### exact_article

| Pipeline | n | MRR@10 doc | MRR@10 art | Cite F1 | Doc Cite F1 | R@10 art | HCR↓ | JIR↓ | Abst F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 5 | 0.700 | 0.600 | 0.248 | 0.447 | 0.367 | 0.000 | 0.000 | 0.000 |
| Dense | 5 | 0.417 | 0.250 | 0.124 | 0.327 | 0.167 | 0.000 | 0.000 | 0.000 |
| Hybrid (RRF) | 5 | 0.667 | 0.600 | 0.264 | 0.427 | 0.400 | 0.000 | 0.000 | 0.000 |
| Hybrid+Rerank | 5 | 0.867 | 0.600 | 0.221 | 0.567 | 0.367 | 0.000 | 0.000 | 0.000 |
| KG (SPARQL) | 5 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| KG+Hybrid | 5 | 0.800 | 0.600 | 0.214 | 0.393 | 0.333 | 0.000 | 0.000 | 0.000 |
| RLM | 5 | 0.800 | 0.600 | 0.413 | 0.800 | 0.367 | 0.000 | 0.000 | 0.400 |

### rule_application

| Pipeline | n | MRR@10 doc | MRR@10 art | Cite F1 | Doc Cite F1 | R@10 art | HCR↓ | JIR↓ | Abst F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 5 | 0.650 | 0.400 | 0.157 | 0.367 | 0.233 | 0.000 | 0.000 | 0.000 |
| Dense | 5 | 0.167 | 0.000 | 0.000 | 0.180 | 0.000 | 0.000 | 0.000 | 0.000 |
| Hybrid (RRF) | 5 | 0.800 | 0.300 | 0.214 | 0.333 | 0.333 | 0.000 | 0.000 | 0.000 |
| Hybrid+Rerank | 5 | 0.700 | 0.400 | 0.157 | 0.347 | 0.233 | 0.000 | 0.000 | 0.000 |
| KG (SPARQL) | 5 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| KG+Hybrid | 5 | 0.467 | 0.400 | 0.164 | 0.293 | 0.267 | 0.000 | 0.000 | 0.000 |
| RLM | 5 | 0.600 | 0.400 | 0.267 | 0.533 | 0.233 | 0.000 | 0.000 | 0.000 |

### multi_hop

| Pipeline | n | MRR@10 doc | MRR@10 art | Cite F1 | Doc Cite F1 | R@10 art | HCR↓ | JIR↓ | Abst F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 5 | 0.533 | 0.200 | 0.050 | 0.307 | 0.067 | 0.000 | 0.000 | 0.000 |
| Dense | 5 | 0.900 | 0.400 | 0.094 | 0.480 | 0.117 | 0.000 | 0.000 | 0.000 |
| Hybrid (RRF) | 5 | 0.500 | 0.200 | 0.050 | 0.400 | 0.067 | 0.000 | 0.000 | 0.000 |
| Hybrid+Rerank | 5 | 0.667 | 0.067 | 0.050 | 0.473 | 0.067 | 0.000 | 0.000 | 0.000 |
| KG (SPARQL) | 5 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| KG+Hybrid | 5 | 0.533 | 0.200 | 0.050 | 0.307 | 0.067 | 0.000 | 0.000 | 0.000 |
| RLM | 5 | 0.700 | 0.300 | 0.150 | 0.600 | 0.117 | 0.000 | 0.000 | 0.000 |

### temporal_factual

| Pipeline | n | MRR@10 doc | MRR@10 art | Cite F1 | Doc Cite F1 | R@10 art | HCR↓ | JIR↓ | Abst F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 5 | 0.483 | 0.050 | 0.067 | 0.373 | 0.200 | 0.000 | 0.000 | 0.000 |
| Dense | 5 | 0.350 | 0.300 | 0.133 | 0.233 | 0.400 | 0.000 | 0.000 | 0.000 |
| Hybrid (RRF) | 5 | 0.590 | 0.240 | 0.133 | 0.373 | 0.400 | 0.000 | 0.000 | 0.000 |
| Hybrid+Rerank | 5 | 0.317 | 0.040 | 0.067 | 0.280 | 0.200 | 0.000 | 0.000 | 0.000 |
| KG (SPARQL) | 5 | 0.100 | 0.000 | 0.000 | 0.133 | 0.000 | 0.000 | 0.000 | 0.000 |
| KG+Hybrid | 5 | 0.133 | 0.133 | 0.133 | 0.147 | 0.400 | 0.000 | 0.000 | 0.000 |
| RLM | 5 | 0.700 | 0.340 | 0.233 | 0.533 | 0.600 | 0.000 | 0.000 | 0.000 |

### conceptual_definitional

| Pipeline | n | MRR@10 doc | MRR@10 art | Cite F1 | Doc Cite F1 | R@10 art | HCR↓ | JIR↓ | Abst F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 5 | 0.467 | 0.050 | 0.067 | 0.500 | 0.200 | 0.000 | 0.000 | 0.000 |
| Dense | 5 | 0.500 | 0.000 | 0.000 | 0.380 | 0.000 | 0.000 | 0.200 | 0.000 |
| Hybrid (RRF) | 5 | 0.600 | 0.000 | 0.000 | 0.413 | 0.000 | 0.000 | 0.000 | 0.000 |
| Hybrid+Rerank | 5 | 0.367 | 0.000 | 0.000 | 0.293 | 0.000 | 0.000 | 0.000 | 0.000 |
| KG (SPARQL) | 5 | 0.300 | 0.000 | 0.000 | 0.267 | 0.000 | 0.000 | 0.000 | 0.000 |
| KG+Hybrid | 5 | 0.433 | 0.090 | 0.133 | 0.413 | 0.400 | 0.000 | 0.000 | 0.000 |
| RLM | 5 | 0.600 | 0.067 | 0.067 | 0.500 | 0.200 | 0.000 | 0.000 | 0.000 |

### unanswerable

| Pipeline | n | MRR@10 doc | MRR@10 art | Cite F1 | Doc Cite F1 | R@10 art | HCR↓ | JIR↓ | Abst F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 5 | 0.400 | 0.000 | 0.000 | 0.233 | 0.000 | 0.000 | 0.000 | 0.000 |
| Dense | 5 | 0.600 | 0.000 | 0.000 | 0.180 | 0.000 | 0.000 | 0.000 | 0.000 |
| Hybrid (RRF) | 5 | 0.600 | 0.050 | 0.067 | 0.233 | 0.200 | 0.000 | 0.000 | 0.000 |
| Hybrid+Rerank | 5 | 0.300 | 0.050 | 0.067 | 0.147 | 0.200 | 0.000 | 0.000 | 0.000 |
| KG (SPARQL) | 5 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| KG+Hybrid | 5 | 0.500 | 0.050 | 0.067 | 0.267 | 0.200 | 0.000 | 0.000 | 0.000 |
| RLM | 5 | 0.000 | 0.000 | 0.600 | 0.600 | 0.000 | 0.000 | 0.000 | 1.000 |

### layman

| Pipeline | n | MRR@10 doc | MRR@10 art | Cite F1 | Doc Cite F1 | R@10 art | HCR↓ | JIR↓ | Abst F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 5 | 0.600 | 0.000 | 0.000 | 0.213 | 0.000 | 0.000 | 0.200 | 0.000 |
| Dense | 5 | 0.500 | 0.200 | 0.067 | 0.300 | 0.200 | 0.000 | 0.000 | 0.000 |
| Hybrid (RRF) | 5 | 0.800 | 0.200 | 0.067 | 0.280 | 0.200 | 0.000 | 0.000 | 0.000 |
| Hybrid+Rerank | 5 | 0.650 | 0.200 | 0.067 | 0.360 | 0.200 | 0.000 | 0.200 | 0.000 |
| KG (SPARQL) | 5 | 0.200 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| KG+Hybrid | 5 | 0.450 | 0.050 | 0.067 | 0.300 | 0.200 | 0.000 | 0.000 | 0.000 |
| RLM | 5 | 0.600 | 0.400 | 0.333 | 0.600 | 0.400 | 0.000 | 0.000 | 0.000 |

### long_context

| Pipeline | n | MRR@10 doc | MRR@10 art | Cite F1 | Doc Cite F1 | R@10 art | HCR↓ | JIR↓ | Abst F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 5 | 0.767 | 0.300 | 0.084 | 0.593 | 0.080 | 0.000 | 0.000 | 0.000 |
| Dense | 5 | 0.650 | 0.000 | 0.000 | 0.313 | 0.000 | 0.000 | 0.000 | 0.000 |
| Hybrid (RRF) | 5 | 0.800 | 0.140 | 0.120 | 0.627 | 0.120 | 0.000 | 0.000 | 0.000 |
| Hybrid+Rerank | 5 | 0.717 | 0.450 | 0.120 | 0.527 | 0.120 | 0.000 | 0.200 | 0.000 |
| KG (SPARQL) | 5 | 0.100 | 0.100 | 0.040 | 0.100 | 0.040 | 0.000 | 0.000 | 0.000 |
| KG+Hybrid | 5 | 0.600 | 0.107 | 0.080 | 0.427 | 0.080 | 0.000 | 0.000 | 0.000 |
| RLM | 5 | 0.900 | 0.150 | 0.080 | 0.667 | 0.120 | 0.000 | 0.000 | 0.000 |


_HCR↓ / JIR↓ — lower is better. Deterministic baselines (B1–B6)
have HCR=JIR=0 by construction (no LLM in the loop)._