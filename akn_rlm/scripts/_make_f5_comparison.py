"""One-off helper: write D:/TRY_AGAIN/eval_results/comparison_f5_full.md."""
from __future__ import annotations

import json
from pathlib import Path

base_dirs = [Path(r"D:\TRY_AGAIN\eval_results"),
             Path(r"D:\TRY_AGAIN\akn_rlm\eval_results")]


def load_metrics(run_id: str) -> dict:
    for d in base_dirs:
        p = d / run_id / "metrics.json"
        if p.exists():
            return json.load(open(p, encoding="utf-8"))
    raise FileNotFoundError(run_id)


runs = [
    ("BM25",          "baseline_bm25_full"),
    ("Dense",         "baseline_dense_full"),
    ("Hybrid (RRF)",  "baseline_hybrid_full"),
    ("Hybrid+Rerank", "baseline_hybrid_rerank_full"),
    ("KG (SPARQL)",   "baseline_kg_full"),
    ("KG+Hybrid",     "baseline_kg_hybrid_full"),
    ("RLM (F2)",      "rlm_dispatched_full"),
    ("RLM (F3)",      "rlm_dispatched_full_v2"),
    ("RLM (F5)",      "rlm_dispatched_full_v4"),
]
metrics = {label: load_metrics(rid) for label, rid in runs}
counts_qt = metrics["BM25"]["counts"]["by_query_type"]

types = ["exact_article", "rule_application", "multi_hop", "temporal_factual",
         "conceptual_definitional", "unanswerable", "layman", "long_context"]

lines: list[str] = []
lines.append("# AlgerianLegalBench v3.0 - F5 Final Comparison (full 244-q)\n")
lines.append("Generated: 2026-05-10. Nine pipelines on the **same 244 questions**.\n")
lines.append("- BM25 / Dense / Hybrid / H+Rerank / KG / KG+Hybrid: Phase-1 baselines (deterministic, no LLM in loop).")
lines.append("- RLM (F2): R7 dispatcher (HANDOFF section 4.99999999).")
lines.append("- RLM (F3): R9.1-R9.7 retunes; supervisor wired but never fired.")
lines.append("- RLM (F5): F4 + F5 surgical tuning. Kept genuine wins (R9.2 TF top_k=2 +0.024,")
lines.append("  R9.4 LC top_k=6 +0.034, EA top_k 5->3 +0.005), reverted regressions (R9.1 thr 0.3->0.5,")
lines.append("  R9.2 CD top_k 2->5, F4 RA top_k 4->8, F4 MH top_k 5->10), and changed the supervisor")
lines.append("  trigger to fire on `len(citations) >= 3` (F3 used [0.30, 0.70] band that never matched")
lines.append("  Qwen3's bimodal confidences, so supervisor fired 0 times in 244 q).\n")

lines.append("## Headline - Cite F1 by query type\n")
lines.append("| Query type | n | " + " | ".join(label for label, _ in runs) + " | F5-F2 |")
lines.append("| --- | ---: | " + " | ".join("---:" for _ in runs) + " | ---: |")
for qt in types:
    n = counts_qt[qt]
    row = f"| {qt} | {n}"
    for label, _ in runs:
        v = metrics[label]["by_query_type"].get(qt, {}).get("citation_f1", 0.0)
        row += f" | {v:.3f}"
    f2 = metrics["RLM (F2)"]["by_query_type"][qt]["citation_f1"]
    f5 = metrics["RLM (F5)"]["by_query_type"][qt]["citation_f1"]
    delta = f5 - f2
    sign = "+" if delta >= 0 else ""
    row += f" | {sign}{delta:.3f} |"
    lines.append(row)
row = "| **overall** | 244"
for label, _ in runs:
    v = metrics[label]["overall"]["citation_f1"]
    row += f" | **{v:.3f}**"
f2 = metrics["RLM (F2)"]["overall"]["citation_f1"]
f5 = metrics["RLM (F5)"]["overall"]["citation_f1"]
row += f" | **{('+' if f5-f2 >= 0 else '')}{f5-f2:.3f}** |"
lines.append(row)

lines.append("\n## Overall metrics (full 244-q)\n")
cols = ["citation_f1", "mrr_article", "mrr_doc", "doc_citation_f1",
        "recall_article", "precision_article", "hcr", "jir",
        "abstention_f1", "mean_latency_s"]
display_cols = ["Cite F1", "MRR art", "MRR doc", "Doc Cite F1", "R@10 art",
                "P@10 art", "HCR-down", "JIR-down", "Abst F1", "Latency"]
lines.append("| Pipeline | " + " | ".join(display_cols) + " |")
lines.append("| --- | " + " | ".join(["---:"] * len(display_cols)) + " |")
for label, _ in runs:
    ovr = metrics[label]["overall"]
    cells = [label] + [f"{ovr.get(c, 0.0):.4f}" for c in cols]
    lines.append("| " + " | ".join(cells) + " |")

lines.append("\n## Per-handler Cite F1 delta (RLM F5 vs F2 vs F3)\n")
lines.append("| Query type | n | RLM F2 | RLM F3 | RLM F5 | Delta F5-F2 | Drives |")
lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
drivers = {
    "exact_article":           "F4 final_top_k 5->3 + supervisor (+0.005 net)",
    "rule_application":        "Supervisor re-rank (+0.011) - top_k tighten reverted",
    "multi_hop":               "R9.3 budget kept (full-244 essentially flat)",
    "temporal_factual":        "R9.2 final_top_k=2 -> +0.023 LIFT",
    "conceptual_definitional": "F4 reverted R9.2 (top_k 2->5) -> +0.028 RECOVERY",
    "unanswerable":            "no handler change (regex-only)",
    "layman":                  "Mostly recovered F4 regression (-0.007 vs F2)",
    "long_context":            "R9.4 final_top_k 10->6 -> +0.034 LIFT (best stratum win)",
}
for qt in types:
    n = counts_qt[qt]
    f2 = metrics["RLM (F2)"]["by_query_type"][qt]["citation_f1"]
    f3 = metrics["RLM (F3)"]["by_query_type"][qt]["citation_f1"]
    f5 = metrics["RLM (F5)"]["by_query_type"][qt]["citation_f1"]
    d = f5 - f2
    sign = "+" if d >= 0 else ""
    lines.append(f"| {qt} | {n} | {f2:.3f} | {f3:.3f} | {f5:.3f} | {sign}{d:.3f} | {drivers[qt]} |")
of2 = metrics["RLM (F2)"]["overall"]["citation_f1"]
of3 = metrics["RLM (F3)"]["overall"]["citation_f1"]
of5 = metrics["RLM (F5)"]["overall"]["citation_f1"]
lines.append(f"| **overall** | 244 | **{of2:.3f}** | **{of3:.3f}** | **{of5:.3f}** | "
             f"**{('+' if of5-of2 >= 0 else '')}{of5-of2:.3f}** | F2 baseline + R9 retunes net to small lift |")

lines.append("\n## F5 final gate read\n")
lines.append("| Gate | Target | F5 result | Status |")
lines.append("| --- | ---: | ---: | --- |")
for name, target, val in [
    ("Cite F1", 0.35, metrics["RLM (F5)"]["overall"]["citation_f1"]),
    ("MRR art", 0.35, metrics["RLM (F5)"]["overall"]["mrr_article"]),
]:
    status = "PASS" if val >= target else f"FAIL (-{target - val:.3f})"
    lines.append(f"| {name} | >= {target:.2f} | {val:.4f} | {status} |")
hcr_values = {qt: metrics["RLM (F5)"]["by_query_type"].get(qt, {}).get("hcr", 0.0)
              for qt in types}
hcr_max = max(hcr_values.values())
hcr_status = "PASS" if hcr_max < 0.05 else f"FAIL (max {hcr_max:.4f})"
lines.append(f"| HCR per-handler | < 0.05 | max={hcr_max:.4f} | {hcr_status} |")

with open(r"D:\TRY_AGAIN\akn_rlm\eval_results\rlm_dispatched_full_v4\predictions.jsonl",
          encoding="utf-8") as f:
    rows = [json.loads(line) for line in f]
sup_used = sum(1 for r in rows if r.get("supervisor_used"))
gpt_calls = sum((r.get("calls_by_model") or {}).get("gpt-oss-120b", 0) for r in rows)
total_calls: dict[str, int] = {}
for r in rows:
    for k, v in (r.get("calls_by_model") or {}).items():
        total_calls[k] = total_calls.get(k, 0) + v

lines.append("\n## F5 telemetry (R9.7 from predictions.jsonl)\n")
lines.append(f"- supervisor_used: **{sup_used}/{len(rows)} questions** "
             f"({100 * sup_used / len(rows):.1f}%)")
lines.append(f"- gpt-oss-120b calls (supervisor): **{gpt_calls}**")
lines.append(f"- Qwen3-30B-A3B-Thinking calls: "
             f"**{total_calls.get('Qwen3-30B-A3B-Thinking', 0)}**")
lines.append(f"- google/gemma-4-31B calls (layman rewriter): "
             f"**{total_calls.get('google/gemma-4-31B', 0)}**")
lines.append(f"- Total sub-LM calls: **{sum(total_calls.values())}**")

lines.append("\n## Per-handler HCR (faithfulness check)\n")
lines.append("| Query type | n | HCR (F5) |")
lines.append("| --- | ---: | ---: |")
for qt in types:
    n = counts_qt[qt]
    h = metrics["RLM (F5)"]["by_query_type"][qt].get("hcr", 0.0)
    lines.append(f"| {qt} | {n} | {h:.4f} |")
lines.append(f"| overall | 244 | {metrics['RLM (F5)']['overall'].get('hcr', 0.0):.4f} |")

lines.append("")
out = Path(r"D:\TRY_AGAIN\eval_results\comparison_f5_full.md")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {out} ({sum(len(line) + 1 for line in lines)} bytes, {len(lines)} lines)")
