# AlgerianLegalBench v3.0 — F3 Final Comparison (full 244-q)

Generated: 2026-05-10. Eight pipelines on the **same 244 questions**.

- BM25 / Dense / Hybrid (RRF) / Hybrid+Rerank / KG / KG+Hybrid: Phase-1 baselines (deterministic, no LLM in loop).
- RLM (F2): R7 dispatcher run from 2026-05-10 (HANDOFF §4.99999999).
- RLM (F3): R9 dispatcher run from 2026-05-10 — adds R9.1 (verifier threshold 0.5→0.3 in RA/MH/EA/layman),
  R9.2 (TF/CD final_top_k 5→2), R9.3 (multi_hop budget expansion + max_sub_calls=25), R9.4 (LC final_top_k 10→6),
  R9.5 (gpt-oss-120b per-citation supervisor with smart trigger on RA/MH/EA/layman/LC), R9.7 (telemetry persistence).
  R9.6 (gpt-oss-120b multi_hop plan supervisor) was **shipped + tested + wired but disabled** for F3 — it regressed multi_hop
  Cite F1 from 0.167 (R9.3 alone, n=10) to 0.070 (R9.3+R9.5+R9.6, n=10) on the gate slice.

## Headline — Cite F1 by query type

| Query type | n | BM25 | Dense | Hybrid (RRF) | Hybrid+Rerank | KG (SPARQL) | KG+Hybrid | RLM (F2) | RLM (F3) | Δ F3 vs F2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exact_article | 59 | 0.152 | 0.118 | 0.160 | 0.183 | 0.031 | 0.139 | 0.411 | 0.414 | +0.003 |
| rule_application | 66 | 0.139 | 0.073 | 0.137 | 0.155 | 0.032 | 0.115 | 0.224 | 0.220 | -0.004 |
| multi_hop | 26 | 0.059 | 0.048 | 0.043 | 0.054 | 0.000 | 0.034 | 0.121 | 0.120 | -0.001 |
| temporal_factual | 7 | 0.048 | 0.095 | 0.095 | 0.095 | 0.000 | 0.095 | 0.167 | 0.190 | +0.024 |
| conceptual_definitional | 12 | 0.083 | 0.107 | 0.056 | 0.052 | 0.000 | 0.111 | 0.107 | 0.097 | -0.010 |
| unanswerable | 40 | 0.000 | 0.000 | 0.008 | 0.008 | 0.033 | 0.017 | 0.525 | 0.525 | +0.000 |
| layman | 17 | 0.024 | 0.020 | 0.020 | 0.020 | 0.000 | 0.020 | 0.282 | 0.269 | -0.013 |
| long_context | 17 | 0.074 | 0.011 | 0.071 | 0.074 | 0.012 | 0.038 | 0.063 | 0.086 | +0.023 |
| **overall** | 244 | **0.093** | **0.063** | **0.094** | **0.105** | **0.022** | **0.083** | **0.293** | **0.293** | **+0.000** |

## Overall metrics (full 244-q)

| Pipeline | Cite F1 | MRR art | MRR doc | Doc Cite F1 | R@10 art | P@10 art | HCR↓ | JIR↓ | Abst F1 | Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.0929 | 0.1971 | 0.5792 | 0.3954 | 0.1863 | 0.0328 | 0.0000 | 0.0164 | 0.0000 | 0.0165 |
| Dense | 0.0633 | 0.1418 | 0.5300 | 0.3622 | 0.1424 | 0.0221 | 0.0000 | 0.0082 | 0.0000 | 0.0957 |
| Hybrid (RRF) | 0.0937 | 0.2133 | 0.6318 | 0.4103 | 0.1954 | 0.0340 | 0.0000 | 0.0082 | 0.0000 | 0.0991 |
| Hybrid+Rerank | 0.1052 | 0.2417 | 0.6214 | 0.4387 | 0.2196 | 0.0377 | 0.0000 | 0.0287 | 0.0000 | 0.3975 |
| KG (SPARQL) | 0.0223 | 0.0348 | 0.1185 | 0.0954 | 0.0343 | 0.0070 | 0.0000 | 0.0082 | 0.0769 | 14.8069 |
| KG+Hybrid | 0.0833 | 0.1800 | 0.5296 | 0.3696 | 0.1754 | 0.0299 | 0.0000 | 0.0123 | 0.0000 | 14.7580 |
| RLM (F2) | 0.2929 | 0.2569 | 0.5232 | 0.5951 | 0.2172 | 0.0348 | 0.0000 | 0.0082 | 0.7020 | 3.9178 |
| RLM (F3) | 0.2933 | 0.2670 | 0.5444 | 0.6079 | 0.2104 | 0.0357 | 0.0000 | 0.0082 | 0.7034 | 4.1799 |

## Per-handler Cite F1 delta (RLM F3 vs RLM F2)

| Query type | n | RLM F2 | RLM F3 | Δ | Drives |
| --- | ---: | ---: | ---: | ---: | --- |
| exact_article | 59 | 0.411 | 0.414 | +0.003 | R9.5 supervisor (≈ neutral, +0.003) |
| rule_application | 66 | 0.224 | 0.220 | -0.004 | R9.1 thr 0.3 + R9.5 supervisor (slight RA softness, −0.004) |
| multi_hop | 26 | 0.121 | 0.120 | -0.001 | R9.3 budget (helped n=10 +0.138 alone) + R9.5 supervisor (full-244: ≈ flat vs F2) |
| temporal_factual | 7 | 0.167 | 0.190 | +0.024 | R9.2 final_top_k=2 → +0.023 LIFT |
| conceptual_definitional | 12 | 0.107 | 0.097 | -0.010 | R9.2 final_top_k=2 (regression on n=12, F2 had paraphraser-fix tail) |
| unanswerable | 40 | 0.525 | 0.525 | +0.000 | no R9.x change (handler is regex-only, supervisor not wired) |
| layman | 17 | 0.282 | 0.269 | -0.013 | R9.1 + R9.5 supervisor (slight regression −0.013) |
| long_context | 17 | 0.063 | 0.086 | +0.023 | R9.4 final_top_k=6 → +0.023 LIFT (MRR doc unchanged at 0.833) |
| **overall** | 244 | **0.293** | **0.293** | **+0.000** | (R9.1-R9.5 + R9.7) |

## F3 final gate read

| Gate | Target | F3 result | Status |
| --- | ---: | ---: | --- |
| Cite F1 | ≥ 0.35 | 0.2933 | FAIL (-0.057) |
| MRR art | ≥ 0.35 | 0.2670 | FAIL (-0.083) |
| HCR per-handler | < 0.05 | max=0.0000 | PASS |

## Per-handler HCR (R9.x faithfulness check)

| Query type | n | HCR (F3) |
| --- | ---: | ---: |
| exact_article | 59 | 0.0000 |
| rule_application | 66 | 0.0000 |
| multi_hop | 26 | 0.0000 |
| temporal_factual | 7 | 0.0000 |
| conceptual_definitional | 12 | 0.0000 |
| unanswerable | 40 | 0.0000 |
| layman | 17 | 0.0000 |
| long_context | 17 | 0.0000 |
| overall | 244 | 0.0000 |
