# AlgerianLegalBench v3.0

**A Native Benchmark for Evaluating Jurisdictional Faithfulness in Algerian Legal NLP**

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Dataset on HuggingFace](https://img.shields.io/badge/🤗-Dataset-yellow.svg)](https://huggingface.co/datasets/YOUR_USERNAME/AlgerianLegalBench)

## Overview

AlgerianLegalBench is a benchmark of **244 expert-annotated questions** for evaluating legal information retrieval and answer generation on Algerian law. The benchmark focuses on whether systems cite the correct Algerian legal sources and avoid importing rules from other jurisdictions.

The current evaluator in [`evaluation/evaluate_baselines.py`](evaluation/evaluate_baselines.py) computes:

- **Citation faithfulness**: Citation Precision, Citation Recall, and Citation F1.
- **Document retrieval correctness**: Precision@1 and Mean Reciprocal Rank (MRR) over recognized Algerian legal documents.
- **Jurisdictional faithfulness**: Jurisdictional Infection Rate (JIR), measured on unanswerable/foreign-law trap questions.
- **Citation safety**: Hallucinated Citation Rate (HCR), measured as unexpected article citations relative to the gold articles.

All reported scores are proportions in `[0, 1]`. Higher is better for Citation Precision/Recall/F1, Precision@1, and MRR. Lower is better for HCR and JIR.

## Key Results (7 Models)

| Model | Params | Cit.F1 | Cit.P | Cit.R | P@1 | MRR | HCR↓ | JIR↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gemini 2.5 Flash | - | **0.186** | **0.189** | 0.186 | 0.254 | 0.254 | **0.008** | 0.375 |
| Llama 3.2 | 3B | 0.129 | 0.129 | 0.213 | 0.385 | 0.393 | 0.492 | 0.400 |
| Gemma 2 | 9B | 0.120 | 0.123 | 0.219 | 0.389 | 0.408 | 0.725 | **0.175** |
| Llama 3.3 | 70B | 0.100 | 0.088 | **0.300** | **0.627** | **0.656** | 0.803 | 0.350 |
| Qwen 2 | 7B | 0.085 | 0.087 | 0.194 | 0.361 | 0.383 | 0.701 | 0.675 |
| Llama 3.1 | 8B | 0.068 | 0.063 | 0.270 | 0.430 | 0.446 | 0.775 | 0.500 |
| Aya Expanse | 8B | 0.030 | 0.025 | 0.208 | 0.557 | 0.592 | 0.811 | 0.325 |

**No model exceeds 19% Citation F1.** Qwen 2 imports foreign law on **67.5%** of unanswerable questions.

## Evaluation Metrics

### Metric Quick Reference

| Metric | Output field | Scope | Meaning |
|---|---|---|---|
| Citation Precision | `citation_precision` | All valid questions | Fraction of extracted article citations that match a gold article reference. |
| Citation Recall | `citation_recall` | All valid questions | Fraction of gold article references recovered by the model. |
| Citation F1 | `citation_f1` | All valid questions | Harmonic mean of Citation Precision and Citation Recall. |
| Precision@1 | `precision_at_1` | All valid questions | Whether the first extracted legal document is among the gold documents. |
| MRR | `mrr` | All valid questions | Reciprocal rank of the first extracted legal document that appears in the gold document set. |
| Hallucinated Citation Rate (HCR) | `hallucinated_citation_rate` | All valid questions | Fraction of questions where the model cites at least one article outside the gold article set. |
| Jurisdictional Infection Rate (JIR) | `infection_rate` under `jurisdictional_infection` | Unanswerable questions only | Fraction of unanswerable questions where the model gives an affirmative Algerian-law answer to a foreign or absent legal concept. |

### Citation Precision, Recall, and F1

The evaluator first extracts article references from the model response with Arabic and Latin patterns such as `المادة 54`, `مادة 124`, `Art. 30`, and `Article 30`. Before extraction, it removes law/decree identifiers and dates such as `قانون 90-11`, `رقم 08-04`, and `2020-12-30` so those numbers are not misread as article citations.

Gold articles come from each question's `expected_articles[].article_ref` field.

For one question:

```text
TP = number of predicted article refs also present in the gold article refs
Citation Precision = TP / number of predicted article refs
Citation Recall    = TP / number of gold article refs
Citation F1        = 2 * Precision * Recall / (Precision + Recall)
```

Article references are normalized by removing whitespace before comparison. If a question has no gold articles, the citation score is perfect only when the model also cites no articles; otherwise precision and F1 become `0.0`.

### Precision@1 and MRR

Document-level metrics are computed from legal document references extracted from the model response. The extractor maps known law names and aliases to document IDs, for example:

- `قانون الأسرة` -> `84-11_1984-06-09`
- `القانون المدني` -> `75-8_1975-09-26`
- `قانون العقوبات` -> `66-156_1966-06-08`
- `Code civil` -> `75-8_1975-09-26`

Gold documents come from each question's `expected_documents` field.

**Precision@1** is `1.0` if the first extracted document is in the gold document set, otherwise `0.0`. If a question has no gold documents, Precision@1 is `1.0` only when the model also produces no recognized document reference.

**MRR** is the reciprocal rank of the first extracted document that appears in the gold document set:

```text
MRR = 1 / rank_of_first_correct_document
```

If no extracted document is correct, MRR is `0.0`. In the current implementation, questions with no gold documents return MRR `1.0`.

Implementation note: `extract_law_references()` currently returns a Python `set`. That makes P@1 and MRR useful as coarse document-hit indicators, but not as strict ranked-retrieval metrics unless the extractor is changed to preserve mention order.

### Hallucinated Citation Rate (HCR)

HCR is a binary per-question safety metric. A question is marked as having a hallucinated citation when the model cites at least one article reference that is not present in the gold article set.

```text
HCR = questions_with_hallucinated_citation / valid_questions
```

Lower is better. The aggregate field is `aggregate_metrics.overall.hallucinated_citation_rate`.

Important nuance: the current implementation checks whether cited articles are outside the question's expected article set. It does **not** verify article existence against the full document registry. If a question has an empty gold article set, `detect_hallucinated_citation()` currently does not flag hallucination even if the model cites an article.

### Jurisdictional Infection Rate (JIR)

JIR measures whether a model imports foreign or absent legal concepts into an Algerian-law answer. It is computed **only on unanswerable questions**, where `answerable` is `false`.

An unanswerable question is marked as infected when the model gives a substantive affirmative answer instead of saying that the concept does not exist or is not recognized in Algerian law. The detector is heuristic:

- It treats negative/absence markers such as `لا يوجد`, `لا ينص`, `لا يعترف`, `غير موجود`, `n'existe pas`, and `ne reconnaît pas` as non-infected.
- It treats affirmative/legal-rule markers such as `نعم`, `يجوز`, `يمكن`, `ينص القانون`, `وفق المادة`, `طبقاً`, and `la loi prévoit` as infected when no negative marker is present.
- Ambiguous responses are not counted as infected.

```text
JIR = infected_unanswerable_questions / valid_unanswerable_questions
```

Lower is better. The aggregate fields are:

- `aggregate_metrics.jurisdictional_infection.n`
- `aggregate_metrics.jurisdictional_infection.infected_count`
- `aggregate_metrics.jurisdictional_infection.infection_rate`

Example failure: if a question asks about US-style punitive damages and the model answers that Algerian courts can award punitive damages under Article 124 of the Civil Code, that is jurisdictional infection. A faithful answer should state that Algerian law provides compensatory reparation and does not recognize punitive damages.

### Aggregation

The evaluator ignores backend/API errors when computing aggregate metrics. A result is considered valid if it has a populated `metrics` object with `citation_f1`.

The output JSON contains:

- `aggregate_metrics.overall`: mean Citation Precision, Citation Recall, Citation F1, Precision@1, MRR, and HCR over valid questions.
- `aggregate_metrics.jurisdictional_infection`: JIR over valid unanswerable questions only.
- `aggregate_metrics.by_category`: Citation F1 and Precision@1 by legal category.
- `aggregate_metrics.by_query_type`: Citation F1 and MRR by query type.
- `aggregate_metrics.by_difficulty`: Citation F1 by difficulty level.

### Metrics Listed in the Dataset Metadata

The dataset metadata lists broader metric families, including `Precision@K`, `Recall@K`, `MAP`, `nDCG@K`, `ROUGE-L`, `BERTScore`, and semantic similarity. These are **not computed by the current baseline evaluator**.

Those metrics require additional outputs that `evaluate_baselines.py` does not currently produce:

- `Precision@K`, `Recall@K`, `MAP`, and `nDCG@K` require a ranked retrieval list.
- `ROUGE-L`, `BERTScore`, and semantic similarity require reference-answer comparison logic.

## Quick Start

```bash
# Install
pip install groq google-genai

# Set API key
export GROQ_API_KEY="gsk_..."

# Run evaluation
python evaluation/evaluate_baselines.py evaluate \
    --benchmark data/AlgerianLegalBench_v3.0.json \
    --backend groq \
    --model llama-3.3-70b-versatile \
    --output-dir results

# Compare models
python evaluation/evaluate_baselines.py compare --results-dir results
```

## Dataset Structure

```text
data/
└── AlgerianLegalBench_v3.0.json    # 244 questions, full benchmark
evaluation/
├── evaluate_baselines.py           # Main evaluation script
├── retry_errors.py                 # Retry failed API calls
├── generate_tables.py              # Generate paper tables
└── requirements.txt
results/                            # Raw evaluation outputs (7 models)
style_guide/
└── annotation_guide.md             # Annotation protocol (10 worked examples)
```

## Benchmark Statistics

- **244** questions across **23** legal categories
- **8** query types: exact article, rule application, multi-hop, long context, unanswerable, layman/Darja, conceptual, temporal
- **40** unanswerable questions probing **7+** foreign or absent-law sources: FR, US, EG, TN, UK, Gulf, international/other, and genuine Algerian absence cases
- **2** recorded languages: MSA/Arabic (`ar`, 232) and Legal French (`fr`, 12); the Arabic set includes layman/Darja-style prompts
- **30** laws in document registry (1966 Penal Code -> 2025 Criminal Procedure Code)
- Inter-annotator κ = **0.829**

## Citation

```bibtex
@inproceedings{attia2026algerianlegal,
  title={AlgerianLegalBench: A Native Benchmark for Evaluating Jurisdictional Faithfulness in Algerian Legal NLP},
  author={Attia, Ibrahim El Khalil and Choui, Maab and Slimi, Islam and Slimi, Ziad and Dahak, Fouad},
  booktitle={Proceedings of the Fourth Arabic Natural Language Processing Conference (ArabicNLP 2026)},
  year={2026}
}
```

## License

CC BY 4.0
