#!/usr/bin/env python3
"""
AlgerianLegalBench v3.0 — Baseline Evaluation Script
=====================================================
Runs LLMs on the benchmark and computes all metrics.

FREE backends supported:
  1. Ollama (local)       — pip install ollama      → runs llama3, mistral, qwen2, gemma2 locally
  2. Groq (free API)      — pip install groq        → 14,400 req/day free (llama-3.1-70b, mixtral, gemma2)
  3. Google AI Studio     — pip install google-genai → Gemini 1.5 Flash free tier (1500 req/day)
  4. HuggingFace (free)   — pip install huggingface_hub → free inference API

Usage:
  python evaluate_baselines.py --benchmark AlgerianLegalBench_v3.0_final.json --backend ollama --model llama3.1
  python evaluate_baselines.py --benchmark ... --backend groq --model llama-3.1-70b-versatile --api-key $GROQ_API_KEY
  python evaluate_baselines.py --benchmark ... --backend google --model gemini-1.5-flash --api-key $GOOGLE_API_KEY
  python evaluate_baselines.py --benchmark ... --backend huggingface --model mistralai/Mistral-7B-Instruct-v0.3
"""

import json, os, sys, time, re, argparse, hashlib
from datetime import datetime
from pathlib import Path
from collections import Counter, defaultdict

# ============================================================
# METRICS
# ============================================================

def extract_article_citations(text):
    """Extract article references from model output.
    IMPORTANT: First removes law-reference numbers (e.g. رقم 08-04, قانون 90-11)
    to avoid counting them as article citations."""
    citations = set()

    # Step 1: Remove law/decree reference numbers that aren't articles
    cleaned = text
    cleaned = re.sub(r'رقم\s*\d{2,4}[-/]\d{2,4}', '', cleaned)
    cleaned = re.sub(r'(?:قانون|القانون|أمر|الأمر|مرسوم)\s*(?:رقم\s*)?\d{2,4}[-/]\d{2,4}', '', cleaned)
    cleaned = re.sub(r'[Ll]oi\s*(?:n[°o]?\s*)?\d{2,4}[-/]\d{2,4}', '', cleaned)
    cleaned = re.sub(r'\d{2,4}[-/]\d{2,4}[-/]\d{2,4}', '', cleaned)  # dates
    cleaned = re.sub(r'\b\d{2}-\d{2,4}\b', '', cleaned)  # IDs like 08-04

    # Step 2: Extract real article citations
    patterns = [
        r'(?:المادة|المادّة|مادة)\s+(\d+(?:\s*(?:مكرر|مكرّر)(?:\s*\d+)?)?)',
        r'المواد\s+(\d+)',
        r'[Aa]rt(?:icle)?\.?\s*(\d+(?:\s*(?:bis|ter|quater))?)',
    ]
    for p in patterns:
        for m in re.finditer(p, cleaned):
            ref = m.group(1).strip()
            ref = re.sub(r'\s+', ' ', ref)
            citations.add(ref)
    return citations

def extract_law_references(text):
    """Extract law/document references from model output."""
    refs = set()
    # Patterns: قانون الأسرة, القانون المدني, قانون العقوبات, Code civil, etc.
    law_patterns = [
        (r'قانون الأسرة', '84-11_1984-06-09'),
        (r'القانون المدني', '75-8_1975-09-26'),
        (r'القانون التجاري', '1975_1975-09-26'),
        (r'قانون العقوبات', '66-156_1966-06-08'),
        (r'قانون الإجراءات المدنية', '08-09_2008-02-25'),
        (r'قانون الإجراءات الجزائية', '25-14_2025-08-03'),
        (r'قانون الاستثمار', '22-18_2022-07-24'),
        (r'(?:الدستور|دستور)', '2020_2020-12-30'),
        (r'قانون علاقات العمل|قانون العمل|90-11', '90-11_1990-04-21'),
        (r'قانون مكافحة الفساد|06-01', '06-01_2006-02-20'),
        (r'قانون حماية البيئة|03-10', '03-10_2003-07-19'),
        (r'قانون حماية المستهلك|09-03', '09-03_2009-02-25'),
        (r'قانون البلدية|11-10', '11-10_2011-06-22'),
        (r'التأمينات الاجتماعية|83-11', '83-11_1983-07-02'),
        (r'حقوق المؤلف|03-05', '03-05_2003-07-19'),
        (r'الصفقات العمومية|15-247', '15-247_2015-09-16'),
        (r'التجارة الإلكترونية|18-05', '18-05_2018-05-10'),
        (r'الترقية العقارية|11-04', '11-04_2011-02-17'),
        (r'Code (?:de la )?famille', '84-11_1984-06-09'),
        (r'Code civil', '75-8_1975-09-26'),
        (r'Code (?:de )?commerce', '1975_1975-09-26'),
        (r'Code pénal', '66-156_1966-06-08'),
    ]
    for pattern, doc_id in law_patterns:
        if re.search(pattern, text):
            refs.add(doc_id)
    return refs

def citation_f1(predicted_articles, expected_articles):
    """Compute Citation Precision, Recall, F1."""
    pred = set()
    for ref in predicted_articles:
        # Normalize: remove spaces, convert Arabic numerals
        normalized = re.sub(r'\s+', '', str(ref))
        pred.add(normalized)

    exp = set()
    for art in expected_articles:
        ref = art.get('article_ref', '')
        normalized = re.sub(r'\s+', '', str(ref))
        exp.add(normalized)

    if not exp:
        return {'precision': 1.0 if not pred else 0.0, 'recall': 1.0, 'f1': 1.0 if not pred else 0.0}

    tp = len(pred & exp)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(exp) if exp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {'precision': round(precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}

def document_precision_at_1(predicted_docs, expected_docs):
    """Check if the first predicted document is correct."""
    if not expected_docs:
        return 1.0 if not predicted_docs else 0.0
    if not predicted_docs:
        return 0.0
    return 1.0 if list(predicted_docs)[0] in expected_docs else 0.0

def mrr(predicted_docs, expected_docs):
    """Mean Reciprocal Rank for document retrieval."""
    if not expected_docs:
        return 1.0
    pred_list = list(predicted_docs)
    for i, doc in enumerate(pred_list):
        if doc in expected_docs:
            return 1.0 / (i + 1)
    return 0.0

def detect_jurisdictional_infection(response, question_data):
    """Detect if the model imported foreign law concepts."""
    if question_data['answerable']:
        return False  # Only check unanswerable questions

    # If the model gives a confident affirmative answer to an unanswerable question = infection
    negative_markers = [
        'لا وجود', 'لا يوجد', 'لا ينص', 'غير موجود', 'غير معروف',
        'لا يعترف', 'محظور', 'ممنوع', 'مفهوم أجنبي', 'مستورد',
        "n'existe pas", "ne reconnaît pas", "non", "pas de",
        'لا يطبق', 'لا يسري', 'خارج', 'مخالف',
    ]
    response_lower = response.lower()

    # If model correctly identifies the concept doesn't exist → no infection
    if any(marker in response for marker in negative_markers):
        return False

    # If model gives a substantive affirmative answer → infection
    affirmative_markers = [
        'نعم', 'يجوز', 'يحق', 'يمكن', 'ينص القانون', 'وفق المادة',
        'المادة', 'طبقاً', 'بموجب',
        'oui', 'il est possible', 'la loi prévoit',
    ]
    if any(marker in response for marker in affirmative_markers):
        return True

    return False  # Ambiguous → not counted as infection

def detect_hallucinated_citation(response, expected_articles):
    """Check if model cites non-existent articles."""
    predicted = extract_article_citations(response)
    expected_refs = set()
    for art in expected_articles:
        ref = re.sub(r'\s+', '', str(art.get('article_ref', '')))
        expected_refs.add(ref)

    hallucinated = set()
    for pred in predicted:
        pred_norm = re.sub(r'\s+', '', pred)
        if pred_norm not in expected_refs and expected_refs:
            hallucinated.add(pred)

    return len(hallucinated) > 0, hallucinated

# ============================================================
# LLM BACKENDS
# ============================================================

def build_prompt(question_data):
    """Build the evaluation prompt."""
    q = question_data['question']
    lang = question_data.get('language', 'ar')

    if lang == 'fr':
        system = """Vous êtes un expert en droit algérien. Répondez uniquement sur la base du droit algérien en vigueur.
Citez les articles de loi pertinents avec leur numéro et le nom de la loi.
Si le concept juridique mentionné n'existe pas en droit algérien, dites-le clairement."""
        user = f"Question juridique:\n{q}\n\nRépondez en citant les articles de loi algériens applicables."
    else:
        system = """أنت خبير في القانون الجزائري. أجب فقط بناءً على القانون الجزائري المعمول به حالياً.
اذكر المواد القانونية ذات الصلة بأرقامها واسم القانون.
إذا كان المفهوم القانوني المذكور غير موجود في القانون الجزائري، صرّح بذلك بوضوح."""
        user = f"السؤال القانوني:\n{q}\n\nأجب مع ذكر المواد القانونية الجزائرية المطبقة."

    return system, user

def query_ollama(system, user, model):
    """Query local Ollama instance (uses GPU automatically if available)."""
    try:
        import ollama
        response = ollama.chat(model=model, messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ], options={"temperature": 0.1, "num_predict": 1024, "num_gpu": 99})
        return response['message']['content']
    except Exception as e:
        err = str(e)
        if 'not found' in err.lower() or 'pull' in err.lower():
            return f"[OLLAMA ERROR: Model '{model}' not downloaded. Run: ollama pull {model}]"
        if 'connection' in err.lower() or 'refused' in err.lower():
            return f"[OLLAMA ERROR: Ollama not running. Start it with: ollama serve]"
        return f"[OLLAMA ERROR: {e}]"

def query_groq(system, user, model, api_key):
    """Query Groq free API with retry on rate limits."""
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1, max_tokens=1024,
        )
        return response.choices[0].message.content
    except Exception as e:
        err = str(e)
        if '429' in err or 'rate' in err.lower():
            print("(rate limit, waiting 60s...)", end=' ', flush=True)
            time.sleep(60)
            try:
                client = Groq(api_key=api_key)
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.1, max_tokens=1024,
                )
                return response.choices[0].message.content
            except Exception as e2:
                return f"[GROQ ERROR (retry failed): {e2}]"
        if 'authentication' in err.lower() or 'api_key' in err.lower():
            return f"[GROQ ERROR: Invalid API key. Check GROQ_API_KEY env variable]"
        return f"[GROQ ERROR: {e}]"

def query_google(system, user, model, api_key):
    """Query Google Gemini — supports both AI Studio (AIza keys) and Vertex AI (gcloud auth).
    
    For Vertex AI: set environment variables:
      GOOGLE_CLOUD_PROJECT=your-project-id
      GOOGLE_CLOUD_LOCATION=us-central1
      GOOGLE_GENAI_USE_VERTEXAI=True
    Then run: gcloud auth application-default login
    """
    from google import genai
    from google.genai import types
    
    use_vertex = os.environ.get('GOOGLE_GENAI_USE_VERTEXAI', '').lower() == 'true'
    
    try:
        if use_vertex:
            # Vertex AI mode — uses gcloud credentials, no API key needed
            project = os.environ.get('GOOGLE_CLOUD_PROJECT', '')
            location = os.environ.get('GOOGLE_CLOUD_LOCATION', 'us-central1')
            if not project:
                return "[GOOGLE ERROR: Set GOOGLE_CLOUD_PROJECT env variable to your project ID]"
            client = genai.Client(
                vertexai=True,
                project=project,
                location=location,
            )
        else:
            # AI Studio mode — needs AIza API key
            if not api_key or not api_key.startswith('AIza'):
                return "[GOOGLE ERROR: Invalid key. For Cloud Console users, set these env vars instead:\n  GOOGLE_CLOUD_PROJECT=your-project-id\n  GOOGLE_CLOUD_LOCATION=us-central1\n  GOOGLE_GENAI_USE_VERTEXAI=True\nThen run: gcloud auth application-default login]"
            client = genai.Client(api_key=api_key)
        
        response = client.models.generate_content(
            model=model,
            contents=f"{system}\n\n{user}",
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=1024,
            ),
        )
        return response.text
    except Exception as e:
        err = str(e)
        if '429' in err or 'quota' in err.lower() or 'resource' in err.lower():
            print("(quota limit, waiting 30s...)", end=' ', flush=True)
            time.sleep(30)
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=f"{system}\n\n{user}",
                    config=types.GenerateContentConfig(temperature=0.1, max_output_tokens=1024),
                )
                return response.text
            except Exception as e2:
                return f"[GOOGLE ERROR (retry): {e2}]"
        if 'not found' in err.lower() or 'not supported' in err.lower():
            return f"[GOOGLE ERROR: Model '{model}' not available in your region. Try: gemini-2.0-flash-001]"
        return f"[GOOGLE ERROR: {err[:200]}]"

def query_huggingface(system, user, model):
    """Query HuggingFace free Inference API."""
    try:
        from huggingface_hub import InferenceClient
        client = InferenceClient()
        response = client.chat_completion(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=1024, temperature=0.1,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[HF ERROR: {e}]"

def query_model(backend, model, system, user, api_key=None):
    """Route to the correct backend."""
    if backend == 'ollama':
        return query_ollama(system, user, model)
    elif backend == 'groq':
        return query_groq(system, user, model, api_key)
    elif backend == 'google':
        return query_google(system, user, model, api_key)
    elif backend == 'huggingface':
        return query_huggingface(system, user, model)
    else:
        raise ValueError(f"Unknown backend: {backend}")

# ============================================================
# EVALUATION LOOP
# ============================================================

def evaluate(benchmark_path, backend, model, api_key=None, split='test', limit=None, output_dir='results'):
    """Run full evaluation."""
    with open(benchmark_path, encoding='utf-8') as f:
        data = json.load(f)

    questions = [q for q in data['questions'] if split == 'all' or q['split'] == split]
    if limit:
        questions = questions[:limit]

    print(f"\n{'='*60}")
    print(f"AlgerianLegalBench v3.0 — Baseline Evaluation")
    print(f"{'='*60}")
    print(f"Backend: {backend} | Model: {model}")
    print(f"Split: {split} | Questions: {len(questions)}")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"{'='*60}\n")

    results = []
    errors = 0

    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] {q['id']} ({q['category']}, {q['query_type']}) ...", end=' ', flush=True)

        system, user = build_prompt(q)

        start_time = time.time()
        response = query_model(backend, model, system, user, api_key)
        elapsed = time.time() - start_time

        if response.startswith('[') and 'ERROR' in response:
            print(f"ERROR: {response[1:80]}... ({elapsed:.1f}s)")
            errors += 1
            result = {
                'id': q['id'], 'error': response, 'response': '',
                'metrics': {}, 'elapsed': elapsed,
            }
        else:
            # Compute metrics
            pred_articles = extract_article_citations(response)
            pred_docs = extract_law_references(response)
            cf1 = citation_f1(pred_articles, q['expected_articles'])
            p_at_1 = document_precision_at_1(pred_docs, q['expected_documents'])
            mrr_score = mrr(pred_docs, q['expected_documents'])
            infected = detect_jurisdictional_infection(response, q) if not q['answerable'] else False
            hallucinated, hall_arts = detect_hallucinated_citation(response, q['expected_articles'])

            result = {
                'id': q['id'],
                'category': q['category'],
                'query_type': q['query_type'],
                'difficulty': q['difficulty'],
                'answerable': q['answerable'],
                'language': q.get('language', 'ar'),
                'response': response,
                'predicted_articles': list(pred_articles),
                'predicted_documents': list(pred_docs),
                'expected_articles': [a['article_ref'] for a in q['expected_articles']],
                'expected_documents': q['expected_documents'],
                'metrics': {
                    'citation_precision': cf1['precision'],
                    'citation_recall': cf1['recall'],
                    'citation_f1': cf1['f1'],
                    'precision_at_1': p_at_1,
                    'mrr': mrr_score,
                    'jurisdictional_infection': infected,
                    'hallucinated_citation': hallucinated,
                    'hallucinated_articles': list(hall_arts),
                },
                'elapsed': round(elapsed, 2),
            }
            status = f"CF1={cf1['f1']:.2f} P@1={p_at_1:.0f} {'🔴INF' if infected else ''} ({elapsed:.1f}s)"
            print(status)

        results.append(result)

        # Rate limiting per backend
        if backend == 'groq':
            time.sleep(2.5)  # 30 req/min = 1 per 2s, add buffer
        elif backend == 'google':
            time.sleep(2)    # ~30 req/min free tier
        elif backend == 'huggingface':
            time.sleep(1)

    # ============================================================
    # AGGREGATE METRICS
    # ============================================================
    valid = [r for r in results if 'citation_f1' in r.get('metrics', {})]

    agg = {}
    if valid:
        agg['overall'] = {
            'n': len(valid),
            'citation_f1': round(sum(r['metrics']['citation_f1'] for r in valid) / len(valid), 4),
            'citation_precision': round(sum(r['metrics']['citation_precision'] for r in valid) / len(valid), 4),
            'citation_recall': round(sum(r['metrics']['citation_recall'] for r in valid) / len(valid), 4),
            'precision_at_1': round(sum(r['metrics']['precision_at_1'] for r in valid) / len(valid), 4),
            'mrr': round(sum(r['metrics']['mrr'] for r in valid) / len(valid), 4),
            'hallucinated_citation_rate': round(sum(r['metrics']['hallucinated_citation'] for r in valid) / len(valid), 4),
        }

        # Jurisdictional Infection Rate (only on unanswerable questions)
        unanswerable = [r for r in valid if not r['answerable']]
        if unanswerable:
            agg['jurisdictional_infection'] = {
                'n': len(unanswerable),
                'infection_rate': round(sum(r['metrics']['jurisdictional_infection'] for r in unanswerable) / len(unanswerable), 4),
                'infected_count': sum(r['metrics']['jurisdictional_infection'] for r in unanswerable),
            }

        # By category
        agg['by_category'] = {}
        for cat in sorted(set(r['category'] for r in valid)):
            cat_results = [r for r in valid if r['category'] == cat]
            agg['by_category'][cat] = {
                'n': len(cat_results),
                'citation_f1': round(sum(r['metrics']['citation_f1'] for r in cat_results) / len(cat_results), 4),
                'precision_at_1': round(sum(r['metrics']['precision_at_1'] for r in cat_results) / len(cat_results), 4),
            }

        # By query type
        agg['by_query_type'] = {}
        for qt in sorted(set(r['query_type'] for r in valid)):
            qt_results = [r for r in valid if r['query_type'] == qt]
            agg['by_query_type'][qt] = {
                'n': len(qt_results),
                'citation_f1': round(sum(r['metrics']['citation_f1'] for r in qt_results) / len(qt_results), 4),
                'mrr': round(sum(r['metrics']['mrr'] for r in qt_results) / len(qt_results), 4),
            }

        # By difficulty
        agg['by_difficulty'] = {}
        for diff in ['easy', 'medium', 'hard']:
            diff_results = [r for r in valid if r['difficulty'] == diff]
            if diff_results:
                agg['by_difficulty'][diff] = {
                    'n': len(diff_results),
                    'citation_f1': round(sum(r['metrics']['citation_f1'] for r in diff_results) / len(diff_results), 4),
                }

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    os.makedirs(output_dir, exist_ok=True)
    model_safe = model.replace('/', '_').replace(':', '_')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    output = {
        'metadata': {
            'benchmark': 'AlgerianLegalBench v3.0',
            'backend': backend,
            'model': model,
            'split': split,
            'timestamp': timestamp,
            'total_questions': len(questions),
            'errors': errors,
        },
        'aggregate_metrics': agg,
        'per_question_results': results,
    }

    outpath = os.path.join(output_dir, f'eval_{model_safe}_{timestamp}.json')
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY — {model}")
    print(f"{'='*60}")
    if agg.get('overall'):
        o = agg['overall']
        print(f"  Citation F1:          {o['citation_f1']:.4f}")
        print(f"  Citation Precision:   {o['citation_precision']:.4f}")
        print(f"  Citation Recall:      {o['citation_recall']:.4f}")
        print(f"  Precision@1:          {o['precision_at_1']:.4f}")
        print(f"  MRR:                  {o['mrr']:.4f}")
        print(f"  Hallucinated Cit.:    {o['hallucinated_citation_rate']:.4f}")
    if agg.get('jurisdictional_infection'):
        ji = agg['jurisdictional_infection']
        print(f"  Jurisd. Infection:    {ji['infection_rate']:.4f} ({ji['infected_count']}/{ji['n']})")
    print(f"  Errors:               {errors}/{len(questions)}")
    print(f"\n  Saved: {outpath}")

    return output

# ============================================================
# COMPARE MODELS
# ============================================================

def compare_results(result_dir='results'):
    """Compare all evaluation results in a directory."""
    files = sorted(Path(result_dir).glob('eval_*.json'))
    if not files:
        print("No result files found.")
        return

    print(f"\n{'='*80}")
    print(f"MODEL COMPARISON — AlgerianLegalBench v3.0")
    print(f"{'='*80}")

    models = []
    for f in files:
        with open(f, encoding='utf-8') as fh:
            data = json.load(fh)
        agg = data.get('aggregate_metrics', {}).get('overall', {})
        ji = data.get('aggregate_metrics', {}).get('jurisdictional_infection', {})
        models.append({
            'model': data['metadata']['model'],
            'cf1': agg.get('citation_f1', 0),
            'p1': agg.get('precision_at_1', 0),
            'mrr': agg.get('mrr', 0),
            'hall': agg.get('hallucinated_citation_rate', 0),
            'infect': ji.get('infection_rate', 0),
            'n': agg.get('n', 0),
        })

    # Print comparison table
    print(f"\n{'Model':<35s} {'Cit.F1':>7s} {'P@1':>6s} {'MRR':>6s} {'Hall%':>6s} {'Infect%':>8s} {'N':>4s}")
    print('-' * 80)
    for m in sorted(models, key=lambda x: -x['cf1']):
        print(f"{m['model']:<35s} {m['cf1']:>7.3f} {m['p1']:>6.3f} {m['mrr']:>6.3f} {m['hall']:>6.3f} {m['infect']:>8.3f} {m['n']:>4d}")

# ============================================================
# CLI
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AlgerianLegalBench v3.0 Evaluation')
    sub = parser.add_subparsers(dest='command')

    # Evaluate
    ev = sub.add_parser('evaluate', help='Run evaluation')
    ev.add_argument('--benchmark', required=True, help='Path to benchmark JSON')
    ev.add_argument('--backend', required=True, choices=['ollama', 'groq', 'google', 'huggingface'])
    ev.add_argument('--model', required=True, help='Model name/ID')
    ev.add_argument('--api-key', default=None, help='API key (or set env var)')
    ev.add_argument('--split', default='all', choices=['test', 'dev', 'all'], help='Which split to evaluate (default: all)')
    ev.add_argument('--limit', type=int, default=None, help='Limit number of questions')
    ev.add_argument('--output-dir', default='results')

    # Compare
    cmp = sub.add_parser('compare', help='Compare results')
    cmp.add_argument('--results-dir', default='results')

    # Recompute metrics on existing results (after regex fix)
    recomp = sub.add_parser('recompute', help='Recompute metrics on existing results with fixed regex')
    recomp.add_argument('--result-file', required=True, help='Path to eval_*.json result file')
    recomp.add_argument('--benchmark', required=True, help='Path to benchmark JSON')

    args = parser.parse_args()

    if args.command == 'evaluate':
        api_key = args.api_key or os.environ.get(f'{args.backend.upper()}_API_KEY')
        evaluate(args.benchmark, args.backend, args.model, api_key, args.split, args.limit, args.output_dir)
    elif args.command == 'compare':
        compare_results(args.results_dir)
    elif args.command == 'recompute':
        # Recompute metrics with fixed regex
        with open(args.result_file, encoding='utf-8') as f:
            result_data = json.load(f)
        with open(args.benchmark, encoding='utf-8') as f:
            bench_data = json.load(f)
        bench_map = {q['id']: q for q in bench_data['questions']}

        valid = []
        for r in result_data['per_question_results']:
            if not r.get('response') or r['response'].startswith('['):
                continue
            q = bench_map.get(r['id'], {})
            response = r['response']

            pred_articles = extract_article_citations(response)
            pred_docs = extract_law_references(response)
            cf1 = citation_f1(pred_articles, q.get('expected_articles', []))
            p1 = document_precision_at_1(pred_docs, q.get('expected_documents', []))
            mrr_s = mrr(pred_docs, q.get('expected_documents', []))
            infected = detect_jurisdictional_infection(response, q) if not q.get('answerable', True) else False
            hall, hall_arts = detect_hallucinated_citation(response, q.get('expected_articles', []))

            r['predicted_articles'] = list(pred_articles)
            r['predicted_documents'] = list(pred_docs)
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
            valid.append(r)

        # Recompute aggregates
        if valid:
            agg = {'overall': {
                'n': len(valid),
                'citation_f1': round(sum(r['metrics']['citation_f1'] for r in valid)/len(valid), 4),
                'citation_precision': round(sum(r['metrics']['citation_precision'] for r in valid)/len(valid), 4),
                'citation_recall': round(sum(r['metrics']['citation_recall'] for r in valid)/len(valid), 4),
                'precision_at_1': round(sum(r['metrics']['precision_at_1'] for r in valid)/len(valid), 4),
                'mrr': round(sum(r['metrics']['mrr'] for r in valid)/len(valid), 4),
                'hallucinated_citation_rate': round(sum(r['metrics']['hallucinated_citation'] for r in valid)/len(valid), 4),
            }}
            un = [r for r in valid if not bench_map.get(r['id'],{}).get('answerable',True)]
            if un:
                agg['jurisdictional_infection'] = {
                    'n': len(un),
                    'infection_rate': round(sum(r['metrics']['jurisdictional_infection'] for r in un)/len(un), 4),
                    'infected_count': sum(r['metrics']['jurisdictional_infection'] for r in un),
                }
            result_data['aggregate_metrics'] = agg

        outpath = args.result_file.replace('.json', '_fixed.json')
        with open(outpath, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)

        o = result_data.get('aggregate_metrics',{}).get('overall',{})
        ji = result_data.get('aggregate_metrics',{}).get('jurisdictional_infection',{})
        print(f"\nRECOMPUTED METRICS ({result_data['metadata']['model']})")
        print(f"  Citation F1:       {o.get('citation_f1',0):.4f}")
        print(f"  Citation Prec:     {o.get('citation_precision',0):.4f}")
        print(f"  Citation Recall:   {o.get('citation_recall',0):.4f}")
        print(f"  Precision@1:       {o.get('precision_at_1',0):.4f}")
        print(f"  MRR:               {o.get('mrr',0):.4f}")
        print(f"  Hallucinated:      {o.get('hallucinated_citation_rate',0):.4f}")
        if ji:
            print(f"  JIR:               {ji.get('infection_rate',0):.4f} ({ji.get('infected_count',0)}/{ji.get('n',0)})")
        print(f"  Saved: {outpath}")
    else:
        parser.print_help()
