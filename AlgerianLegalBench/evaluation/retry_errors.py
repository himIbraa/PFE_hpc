#!/usr/bin/env python3
"""
Retry ONLY the failed questions and merge into existing results.
Usage:
  python retry_errors.py --backend groq --model llama-3.3-70b-versatile --original results/eval_llama-3_3-70b-versatile_20260417_154127.json
  python retry_errors.py --backend google --model gemini-2.5-flash --original results/eval_gemini-2_5-flash_20260417_190512.json
"""
import json, os, sys, time, argparse

# Import the evaluation functions from the main script
sys.path.insert(0, os.path.dirname(__file__))
from evaluate_baselines import (
    build_prompt, query_model, extract_article_citations, extract_law_references,
    citation_f1, document_precision_at_1, mrr, detect_jurisdictional_infection,
    detect_hallucinated_citation
)

def retry(original_file, benchmark_file, backend, model, api_key=None):
    # Load original results
    with open(original_file, encoding='utf-8') as f:
        orig = json.load(f)
    
    # Load benchmark
    with open(benchmark_file, encoding='utf-8') as f:
        bench = json.load(f)
    bench_map = {q['id']: q for q in bench['questions']}
    
    # Find failed question IDs
    failed_ids = [r['id'] for r in orig['per_question_results'] if r.get('error') or not r.get('response')]
    print(f"Retrying {len(failed_ids)} failed questions for {model}...")
    
    # Retry each
    fixed = 0
    still_failed = 0
    for i, qid in enumerate(failed_ids):
        q = bench_map.get(qid)
        if not q:
            print(f"  [{i+1}/{len(failed_ids)}] {qid} — NOT IN BENCHMARK, skipping")
            continue
        
        print(f"  [{i+1}/{len(failed_ids)}] {qid} ...", end=' ', flush=True)
        system, user = build_prompt(q)
        
        response = query_model(backend, model, system, user, api_key)
        
        if response.startswith('[') and 'ERROR' in response:
            print(f"STILL FAILED: {response[:80]}")
            still_failed += 1
            if '429' in response or 'rate' in response.lower() or 'quota' in response.lower():
                print("  Rate limit hit — stopping. Re-run tomorrow for remaining.")
                break
            continue
        
        # Compute metrics
        pred_articles = extract_article_citations(response)
        pred_docs = extract_law_references(response)
        cf1 = citation_f1(pred_articles, q['expected_articles'])
        p1 = document_precision_at_1(pred_docs, q['expected_documents'])
        mrr_s = mrr(pred_docs, q['expected_documents'])
        infected = detect_jurisdictional_infection(response, q) if not q['answerable'] else False
        hall, hall_arts = detect_hallucinated_citation(response, q['expected_articles'])
        
        # Update the result in-place
        for r in orig['per_question_results']:
            if r['id'] == qid:
                r.pop('error', None)
                r['response'] = response
                r['category'] = q['category']
                r['query_type'] = q['query_type']
                r['difficulty'] = q['difficulty']
                r['answerable'] = q['answerable']
                r['language'] = q.get('language', 'ar')
                r['predicted_articles'] = list(pred_articles)
                r['predicted_documents'] = list(pred_docs)
                r['expected_articles'] = [a['article_ref'] for a in q['expected_articles']]
                r['expected_documents'] = q['expected_documents']
                r['metrics'] = {
                    'citation_precision': cf1['precision'],
                    'citation_recall': cf1['recall'],
                    'citation_f1': cf1['f1'],
                    'precision_at_1': p1,
                    'mrr': mrr_s,
                    'jurisdictional_infection': infected,
                    'hallucinated_citation': hall,
                    'hallucinated_articles': list(hall_arts),
                }
                break
        
        fixed += 1
        print(f"CF1={cf1['f1']:.2f} ✓")
        
        if backend in ('groq', 'google'):
            time.sleep(3)
    
    # Recompute aggregates
    valid = [r for r in orig['per_question_results'] if r.get('metrics') and 'citation_f1' in r.get('metrics',{})]
    if valid:
        orig['aggregate_metrics'] = {'overall': {
            'n': len(valid),
            'citation_f1': round(sum(r['metrics']['citation_f1'] for r in valid)/len(valid), 4),
            'citation_precision': round(sum(r['metrics']['citation_precision'] for r in valid)/len(valid), 4),
            'citation_recall': round(sum(r['metrics']['citation_recall'] for r in valid)/len(valid), 4),
            'precision_at_1': round(sum(r['metrics']['precision_at_1'] for r in valid)/len(valid), 4),
            'mrr': round(sum(r['metrics']['mrr'] for r in valid)/len(valid), 4),
            'hallucinated_citation_rate': round(sum(r['metrics']['hallucinated_citation'] for r in valid)/len(valid), 4),
        }}
        un = [r for r in valid if not r.get('answerable', True)]
        if un:
            orig['aggregate_metrics']['jurisdictional_infection'] = {
                'n': len(un),
                'infection_rate': round(sum(r['metrics']['jurisdictional_infection'] for r in un)/len(un), 4),
                'infected_count': sum(r['metrics']['jurisdictional_infection'] for r in un),
            }
    
    new_errors = sum(1 for r in orig['per_question_results'] if r.get('error'))
    orig['metadata']['errors'] = new_errors
    
    # Save
    outpath = original_file.replace('.json', '_complete.json')
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(orig, f, ensure_ascii=False, indent=2)
    
    o = orig['aggregate_metrics']['overall']
    ji = orig['aggregate_metrics'].get('jurisdictional_infection', {})
    print(f"\nFixed {fixed}, still failed {still_failed}")
    print(f"Total OK: {o['n']}/244")
    print(f"Citation F1: {o['citation_f1']:.4f}")
    print(f"P@1: {o['precision_at_1']:.4f}")
    print(f"Hall: {o['hallucinated_citation_rate']:.4f}")
    if ji:
        print(f"JIR: {ji['infection_rate']:.4f} ({ji['infected_count']}/{ji['n']})")
    print(f"Saved: {outpath}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--original', required=True, help='Path to the original eval_*.json with errors')
    parser.add_argument('--benchmark', default='AlgerianLegalBench_v3.0_final.json')
    parser.add_argument('--api-key', default=None)
    args = parser.parse_args()
    api_key = args.api_key or os.environ.get(f'{args.backend.upper()}_API_KEY')
    retry(args.original, args.benchmark, args.backend, args.model, api_key)
