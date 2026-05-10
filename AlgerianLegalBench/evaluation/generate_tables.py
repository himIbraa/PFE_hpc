#!/usr/bin/env python3
"""
Generate paper-ready tables from evaluation results.
Usage: python generate_tables.py --results-dir results/ --output paper/tables.md
"""

import json, os, argparse
from pathlib import Path
from collections import defaultdict

def load_results(results_dir):
    """Load all evaluation result files."""
    models = []
    for f in sorted(Path(results_dir).glob('eval_*.json')):
        with open(f, encoding='utf-8') as fh:
            data = json.load(fh)
        models.append(data)
    return models

def generate_tables(results_dir, output_path):
    models = load_results(results_dir)
    if not models:
        print("No result files found. Run evaluations first.")
        return

    lines = []
    lines.append("# AlgerianLegalBench v3.0 — Paper Tables\n")
    lines.append(f"*Generated from {len(models)} model evaluations*\n")

    # ============================================================
    # TABLE 1: Overall Results
    # ============================================================
    lines.append("## Table 1: Overall Results\n")
    lines.append("| Model | Cit. F1 | Cit. P | Cit. R | P@1 | MRR | HCR ↓ | JIR ↓ |")
    lines.append("|---|---|---|---|---|---|---|---|")

    for m in sorted(models, key=lambda x: -x.get('aggregate_metrics',{}).get('overall',{}).get('citation_f1',0)):
        o = m.get('aggregate_metrics', {}).get('overall', {})
        ji = m.get('aggregate_metrics', {}).get('jurisdictional_infection', {})
        name = m['metadata']['model']
        lines.append(f"| {name} | {o.get('citation_f1',0):.3f} | {o.get('citation_precision',0):.3f} | {o.get('citation_recall',0):.3f} | {o.get('precision_at_1',0):.3f} | {o.get('mrr',0):.3f} | {o.get('hallucinated_citation_rate',0):.3f} | {ji.get('infection_rate',0):.3f} |")

    # ============================================================
    # TABLE 2: Results by Query Type
    # ============================================================
    lines.append("\n## Table 2: Citation F1 by Query Type\n")

    qt_order = ['exact_article','conceptual_definitional','rule_application','temporal_factual',
                'multi_hop','long_context','layman','unanswerable']
    header = "| Model | " + " | ".join(qt_order) + " |"
    sep = "|---|" + "|".join(["---"]*len(qt_order)) + "|"
    lines.append(header)
    lines.append(sep)

    for m in models:
        name = m['metadata']['model']
        by_qt = m.get('aggregate_metrics', {}).get('by_query_type', {})
        vals = [f"{by_qt.get(qt,{}).get('citation_f1',0):.3f}" for qt in qt_order]
        lines.append(f"| {name} | " + " | ".join(vals) + " |")

    # ============================================================
    # TABLE 3: Results by Difficulty
    # ============================================================
    lines.append("\n## Table 3: Citation F1 by Difficulty\n")
    lines.append("| Model | Easy | Medium | Hard |")
    lines.append("|---|---|---|---|")

    for m in models:
        name = m['metadata']['model']
        by_d = m.get('aggregate_metrics', {}).get('by_difficulty', {})
        e = by_d.get('easy', {}).get('citation_f1', 0)
        med = by_d.get('medium', {}).get('citation_f1', 0)
        h = by_d.get('hard', {}).get('citation_f1', 0)
        lines.append(f"| {name} | {e:.3f} | {med:.3f} | {h:.3f} |")

    # ============================================================
    # TABLE 4: Jurisdictional Infection Detail
    # ============================================================
    lines.append("\n## Table 4: Jurisdictional Infection Rate by Model\n")
    lines.append("| Model | Total UN Qs | Infected | JIR |")
    lines.append("|---|---|---|---|")

    for m in models:
        name = m['metadata']['model']
        ji = m.get('aggregate_metrics', {}).get('jurisdictional_infection', {})
        lines.append(f"| {name} | {ji.get('n',0)} | {ji.get('infected_count',0)} | {ji.get('infection_rate',0):.3f} |")

    # ============================================================
    # TABLE 5: Infection Examples (from per-question results)
    # ============================================================
    lines.append("\n## Table 5: Infection Examples (for paper §5.5)\n")
    lines.append("*Select 2-3 most striking examples per model for the paper.*\n")

    for m in models:
        name = m['metadata']['model']
        per_q = m.get('per_question_results', [])
        infected = [r for r in per_q if r.get('metrics', {}).get('jurisdictional_infection')]
        if infected:
            lines.append(f"### {name} — {len(infected)} infections\n")
            for r in infected[:5]:  # Show max 5
                lines.append(f"**{r['id']}** ({r['category']})")
                lines.append(f"> Q: {r.get('response','')[:200]}...")
                lines.append("")

    # ============================================================
    # TABLE 6: Results by Category (top 10)
    # ============================================================
    lines.append("\n## Table 6: Citation F1 by Category (per model)\n")

    all_cats = set()
    for m in models:
        all_cats.update(m.get('aggregate_metrics', {}).get('by_category', {}).keys())
    cats_sorted = sorted(all_cats)

    header = "| Category | " + " | ".join(m['metadata']['model'][:15] for m in models) + " |"
    sep = "|---|" + "|".join(["---"]*len(models)) + "|"
    lines.append(header)
    lines.append(sep)

    for cat in cats_sorted:
        vals = []
        for m in models:
            cf1 = m.get('aggregate_metrics',{}).get('by_category',{}).get(cat,{}).get('citation_f1',0)
            vals.append(f"{cf1:.3f}")
        lines.append(f"| {cat} | " + " | ".join(vals) + " |")

    # Write output
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"Tables written to {output_path}")
    print(f"Models: {len(models)}")
    print(f"Tables generated: 6")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='results')
    parser.add_argument('--output', default='paper/tables.md')
    args = parser.parse_args()
    generate_tables(args.results_dir, args.output)
