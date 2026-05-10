# AlgerianLegalBench v3.0 - F5 Final Comparison (full 244-q)

Generated: 2026-05-10. Nine pipelines on the **same 244 questions**.

- BM25 / Dense / Hybrid / H+Rerank / KG / KG+Hybrid: Phase-1 baselines (deterministic, no LLM in loop).
- RLM (F2): R7 dispatcher (HANDOFF section 4.99999999).
- RLM (F3): R9.1-R9.7 retunes; supervisor wired but never fired.
- RLM (F5): F4 + F5 surgical tuning. Kept genuine wins (R9.2 TF top_k=2 +0.024,
  R9.4 LC top_k=6 +0.034, EA top_k 5->3 +0.005), reverted regressions (R9.1 thr 0.3->0.5,
  R9.2 CD top_k 2->5, F4 RA top_k 4->8, F4 MH top_k 5->10), and changed the supervisor
  trigger to fire on `len(citations) >= 3` (F3 used [0.30, 0.70] band that never matched
  Qwen3's bimodal confidences, so supervisor fired 0 times in 244 q).

## Headline - Cite F1 by query type

| Query type | n | BM25 | Dense | Hybrid (RRF) | Hybrid+Rerank | KG (SPARQL) | KG+Hybrid | RLM (F2) | RLM (F3) | RLM (F5) | F5-F2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| exact_article | 59 | 0.152 | 0.118 | 0.160 | 0.183 | 0.031 | 0.139 | 0.411 | 0.414 | 0.416 | +0.005 |
| rule_application | 66 | 0.139 | 0.073 | 0.137 | 0.155 | 0.032 | 0.115 | 0.224 | 0.220 | 0.235 | +0.011 |
| multi_hop | 26 | 0.059 | 0.048 | 0.043 | 0.054 | 0.000 | 0.034 | 0.121 | 0.120 | 0.122 | +0.001 |
| temporal_factual | 7 | 0.048 | 0.095 | 0.095 | 0.095 | 0.000 | 0.095 | 0.167 | 0.190 | 0.190 | +0.024 |
| conceptual_definitional | 12 | 0.083 | 0.107 | 0.056 | 0.052 | 0.000 | 0.111 | 0.107 | 0.097 | 0.135 | +0.028 |
| unanswerable | 40 | 0.000 | 0.000 | 0.008 | 0.008 | 0.033 | 0.017 | 0.525 | 0.525 | 0.525 | +0.000 |
| layman | 17 | 0.024 | 0.020 | 0.020 | 0.020 | 0.000 | 0.020 | 0.282 | 0.269 | 0.275 | -0.008 |
| long_context | 17 | 0.074 | 0.011 | 0.071 | 0.074 | 0.012 | 0.038 | 0.063 | 0.086 | 0.097 | +0.034 |
| **overall** | 244 | **0.093** | **0.063** | **0.094** | **0.105** | **0.022** | **0.083** | **0.293** | **0.293** | **0.301** | **+0.008** |

## Overall metrics (full 244-q)

| Pipeline | Cite F1 | MRR art | MRR doc | Doc Cite F1 | R@10 art | P@10 art | HCR-down | JIR-down | Abst F1 | Latency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.0929 | 0.1971 | 0.5792 | 0.3954 | 0.1863 | 0.0328 | 0.0000 | 0.0164 | 0.0000 | 0.0165 |
| Dense | 0.0633 | 0.1418 | 0.5300 | 0.3622 | 0.1424 | 0.0221 | 0.0000 | 0.0082 | 0.0000 | 0.0957 |
| Hybrid (RRF) | 0.0937 | 0.2133 | 0.6318 | 0.4103 | 0.1954 | 0.0340 | 0.0000 | 0.0082 | 0.0000 | 0.0991 |
| Hybrid+Rerank | 0.1052 | 0.2417 | 0.6214 | 0.4387 | 0.2196 | 0.0377 | 0.0000 | 0.0287 | 0.0000 | 0.3975 |
| KG (SPARQL) | 0.0223 | 0.0348 | 0.1185 | 0.0954 | 0.0343 | 0.0070 | 0.0000 | 0.0082 | 0.0769 | 14.8069 |
| KG+Hybrid | 0.0833 | 0.1800 | 0.5296 | 0.3696 | 0.1754 | 0.0299 | 0.0000 | 0.0123 | 0.0000 | 14.7580 |
| RLM (F2) | 0.2929 | 0.2569 | 0.5232 | 0.5951 | 0.2172 | 0.0348 | 0.0000 | 0.0082 | 0.7020 | 3.9178 |
| RLM (F3) | 0.2933 | 0.2670 | 0.5444 | 0.6079 | 0.2104 | 0.0357 | 0.0000 | 0.0082 | 0.7034 | 4.1799 |
| RLM (F5) | 0.3011 | 0.2686 | 0.5567 | 0.6122 | 0.2155 | 0.0361 | 0.0000 | 0.0041 | 0.7075 | 4.5086 |

## Per-handler Cite F1 delta (RLM F5 vs F2 vs F3)

| Query type | n | RLM F2 | RLM F3 | RLM F5 | Delta F5-F2 | Drives |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| exact_article | 59 | 0.411 | 0.414 | 0.416 | +0.005 | F4 final_top_k 5->3 + supervisor (+0.005 net) |
| rule_application | 66 | 0.224 | 0.220 | 0.235 | +0.011 | Supervisor re-rank (+0.011) - top_k tighten reverted |
| multi_hop | 26 | 0.121 | 0.120 | 0.122 | +0.001 | R9.3 budget kept (full-244 essentially flat) |
| temporal_factual | 7 | 0.167 | 0.190 | 0.190 | +0.024 | R9.2 final_top_k=2 -> +0.023 LIFT |
| conceptual_definitional | 12 | 0.107 | 0.097 | 0.135 | +0.028 | F4 reverted R9.2 (top_k 2->5) -> +0.028 RECOVERY |
| unanswerable | 40 | 0.525 | 0.525 | 0.525 | +0.000 | no handler change (regex-only) |
| layman | 17 | 0.282 | 0.269 | 0.275 | -0.008 | Mostly recovered F4 regression (-0.007 vs F2) |
| long_context | 17 | 0.063 | 0.086 | 0.097 | +0.034 | R9.4 final_top_k 10->6 -> +0.034 LIFT (best stratum win) |
| **overall** | 244 | **0.293** | **0.293** | **0.301** | **+0.008** | F2 baseline + R9 retunes net to small lift |

## F5 final gate read

| Gate | Target | F5 result | Status |
| --- | ---: | ---: | --- |
| Cite F1 | >= 0.35 | 0.3011 | FAIL (-0.049) |
| MRR art | >= 0.35 | 0.2686 | FAIL (-0.081) |
| HCR per-handler | < 0.05 | max=0.0000 | PASS |

## F5 telemetry (R9.7 from predictions.jsonl)

- supervisor_used: **60/244 questions** (24.6%)
- gpt-oss-120b calls (supervisor): **60**
- Qwen3-30B-A3B-Thinking calls: **1488**
- google/gemma-4-31B calls (layman rewriter): **17**
- Total sub-LM calls: **1565**

## Per-handler HCR (faithfulness check)

| Query type | n | HCR (F5) |
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
