# AlgerianLegalBench v3.0 — Complete Project Context

**Project:** AlgerianLegalBench — A native Algerian legal IR evaluation benchmark

---

## 1. What Was Built

### 1.1 Benchmark File
`AlgerianLegalBench_v3.0_final.json` — 244 questions total

### 1.2 Schema (v3.0)
Every question object contains:
```json
{
  "id": "fam_ra_q01",           // {domain}_{query_type}_q{nn}
  "version": "3.0",
  "source": "benchmark-v3",
  "split": "test|dev",
  "language": "ar|fr",
  "category": "family_law",
  "query_type": "rule_application",
  "difficulty": "easy|medium|hard",
  "question": "...",
  "answerable": true|false,
  "partially_answerable": false,
  "temporal_note": null | {amendment_law, amendment_date, pre/post_amendment_rule, model_failure_risk},
  "expected_documents": ["84-11_1984-06-09"],
  "expected_articles": [{document_id, article_ref, article_ref_disambig, law_name_ar, in_dataset}],
  "ground_truth_answer": "...",
  "reasoning_chain": ["Step 1...", "Step 2...", "Step 3...", "Step 4..."],
  "annotation": {annotator_1: "Islam Slimi", annotator_2: "Ziad Slimi", kappa, agreement_on_answer/articles/chain, ...}
}
```

### 1.3 ID Convention
`{domain_abbr}_{query_type_abbr}_q{nn}` — e.g., `fam_ra_q01`, `xdom_mh_q03`, `con_tf_q01`

Domain abbreviations: fam, civ, crim, com, cpro, cpp, inv, lab, xdom, tax, hous, env, ip, soc, cust, pfn, traf, con, adm, acor, ecom, cofl, cons

Query type abbreviations: ea, cd, ra, tf, mh, lc, un, lm

---

## 2. Benchmark Statistics

### 2.1 Overall
| Property | Value |
|---|---|
| Total questions | 244 |
| Answerable | 204 (83.6%) |
| Unanswerable (infection traps) | 40 (16.4%) |
| Categories | 23 |
| Query types | 8 |
| Languages | Arabic 232, French 12 |
| Test split | 182 (κ ≥ 0.60) |
| Dev split | 62 (κ < 0.60) |
| Laws in registry | 30 |
| Inter-annotator κ | 0.829 |

### 2.2 By Query Type
| Query Type | Count | % | Purpose |
|---|---|---|---|
| Rule Application (RA) | 66 | 27.0% | Apply law to facts |
| Exact Article (EA) | 59 | 24.2% | Pure retrieval |
| Unanswerable (UN) | 40 | 16.4% | Infection traps |
| Multi-Hop (MH) | 26 | 10.7% | Chain 2–3 codes |
| Long Context (LC) | 17 | 7.0% | Synthesize 4+ articles |
| Layman/Darja (LM) | 17 | 7.0% | Non-technical queries |
| Conceptual/Def. (CD) | 12 | 4.9% | Legal definitions |
| Temporal (TF) | 7 | 2.9% | Amendment tracking |

### 2.3 By Difficulty
| Difficulty | Count | % |
|---|---|---|
| Easy | 58 | 24% |
| Medium | 82 | 34% |
| Hard | 104 | 43% |

### 2.4 By Category (23 categories)
| Category | Count |
|---|---|
| Family Law | 35 |
| Commercial Law | 30 |
| Civil Procedure | 23 |
| Civil Law | 20 |
| Criminal Law | 19 |
| Cross-Domain | 12 |
| Investment Law | 11 |
| Criminal Procedure | 10 |
| Labor Law | 10 |
| Constitutional Law | 9 |
| Administrative Law | 9 |
| Tax Law | 8 |
| Housing Law | 7 |
| Conflict of Laws | 6 |
| Anti-Corruption | 6 |
| Environmental Law | 6 |
| Social Security | 6 |
| IP Law | 5 |
| Consumer Law | 2 |
| Customs, E-Commerce, Public Function, Traffic | 1 each |

### 2.5 Unanswerable Questions — By Infection Source
| Source | Count | Example Concepts |
|---|---|---|
| French | 13 | ISF wealth tax, rent control (contrôle des loyers), EURL, SAS, rupture conventionnelle, garde à vue 24h, crédit d'impôt remboursable, notaire monopoly, imprévision (Bordeaux 1916), action de groupe, droit au maintien |
| US | 10 | at-will employment, punitive damages, fair use, class action, plea bargaining, jury trial, stare decisis, pre-trial discovery, Second Amendment, LWOP |
| Egyptian | 4 | Mandatory bequest for grandfather (وصية واجبة), civil marriage officer, automatic penalty clause reduction (Art.224), 2-year pretrial detention |
| Tunisian | 3 | Absolute polygamy ban (Chapter 18 PSC), adoption plénière, independent commercial courts |
| UK | 3 | Consideration doctrine, tort of negligence (duty of care), civil jury |
| Gulf | 2 | Kafeel/local sponsor system, exit permit |
| DZ Absence | 3 | Constitutional auto-saisine, municipal taxing power, death penalty for corruption |
| International | 1 | ICSID direct arbitration |
| Asia | 1 | Death penalty for economic crimes |

### 2.6 Temporal Questions (7)
| ID | Category | Law | Pre-amendment | Post-amendment |
|---|---|---|---|---|
| lab_tf_q01 | Labor | 90-11 (1990) | Socialist management system | Freedom of contract |
| con_tf_q01 | Constitutional | Const 2020 | Unlimited terms (2008) | Two terms only |
| tax_tf_q01 | Tax/Investment | 22-18 (2022) | 51/49 rule mandatory | Abolished except strategic |
| fam_tf_q01 | Family | 84-11 Art.54 | Khul' requires husband consent | Khul' without consent (2005) |
| cpp_tf_q01 | Criminal Proc. | 25-14 (2025) | Limited lawyer access | Lawyer from first hour |
| com_tf_q01 | Commercial | Ccom Art.566 | SARL min capital 100,000 DZD | Symbolic minimum (2015) |
| crim_tf_q01 | Criminal | 25-14 (2025) | Limited defense rights | Enhanced guarantees |

### 2.7 Document Registry (30 laws)
| ID | Short | Arabic Name | Refs |
|---|---|---|---|
| 84-11_1984-06-09 | Cfam | قانون الأسرة | 36 |
| 75-8_1975-09-26 | Cciv | القانون المدني | 34 |
| 1975_1975-09-26 | Ccom | القانون التجاري | 34 |
| 08-09_2008-02-25 | CPCA | قانون الإجراءات المدنية والإدارية | 31 |
| 22-18_2022-07-24 | Cinv | قانون الاستثمار | 16 |
| 25-14_2025-08-03 | CPP | قانون الإجراءات الجزائية الجديد | 11 |
| 90-11_1990-04-21 | Ctrav | قانون علاقات العمل | 10 |
| 2020_2020-12-30 | Const | الدستور | 9 |
| 83-11_1983-07-02 | CCNAS | قانون التأمينات الاجتماعية | 8 |
| 06-01_2006-02-20 | Cacor | قانون مكافحة الفساد | 8 |
| 03-10_2003-07-19 | Cenv | قانون حماية البيئة | 6 |
| 11-10_2011-06-22 | Cbld | قانون البلدية | 5 |
| 66-155_1966-06-08 | aCPP | قانون الإجراءات الجزائية القديم (ملغى) | 4 |
| 09-03_2009-02-25 | Ccons | قانون حماية المستهلك | 4 |
| 03-05_2003-07-19 | CDA | حقوق المؤلف | 4 |
| 05-01_2005-02-06 | CAML | قانون مكافحة تبييض الأموال | 3 |
| 11-04_2011-02-17 | Cimm | قانون الترقية العقارية | 3 |
| 66-156_1966-06-08 | CP | قانون العقوبات | 2 |
| 15-247_2015-09-16 | DSP | مرسوم الصفقات العمومية | 2 |
| 18-05_2018-05-10 | Cecom | قانون التجارة الإلكترونية | 2 |
| + 10 more with 0 refs | | (nationality, military justice, prisons, health, drugs, media, etc.) | |

### 2.8 Annotation Results
| Metric | Value |
|---|---|
| Annotator 1 | Islam Slimi |
| Annotator 2 | Ziad Slimi |
| Resolver | Senior Expert (Prof.) |
| Overall κ | 0.829 |
| κ (easy) | 0.979 |
| κ (medium) | 0.904 |
| κ (hard) | 0.659 |
| κ (exact_article) | 0.967 |
| κ (multi_hop) | 0.582 |
| κ (long_context) | 0.574 |
| κ threshold for test | 0.60 |
| Questions resolved by expert | 57 |

---

## 3. Reasoning Chain Templates

### Template A — Standard (EA, CD, RA, LM)
1. Legal issue — Name the precise sub-issue
2. Applicable rule — Cite exact article, quote operative phrase
3. Application/Subsumption — Map facts to rule elements
4. Conclusion — State outcome; flag judicial discretion

### Template B — Temporal (TF)
1. Version identification — Which version in force?
2. Amendment timeline — What changed, when, cite amending law
3. Correct version applied — Apply right version
4. Model failure description — What pre-trained model would say

### Template C — Multi-Hop (MH)
1. Name the chain — List all codes/articles in traversal order
2. Hop 1 — Apply first code; intermediate conclusion
3. Hop 2 — Apply next code using Hop 1 output
4. Synthesis + failure point — Where does single-code retrieval fail?

### Template D — Unanswerable (UN)
1. Failure mode — (A) Jurisdictional infection, (B) Fictitious citation, (C) Genuine absence
2. Demonstrate the absence — Cite sections that do NOT contain the rule
3. Incorrect model answer — What infected model would say
4. Correct response — "This concept does not exist under Algerian law..."

---

## 4. Baseline Evaluation Results

### 4.1 Models Tested (7 models × 244 questions = 1,708 evaluations, ALL complete)
| Model | Params | Backend | Notes |
|---|---|---|---|
| Gemini 2.5 Flash | — | Google Vertex AI | Commercial, best Arabic |
| Llama 3.3 | 70B | Groq (cloud) | Largest open model |
| Gemma 2 | 9B | Ollama (local GPU) | Google open model |
| Qwen 2 | 7B | Ollama (local GPU) | Chinese lab, multilingual |
| Llama 3.2 | 3B | Ollama (local GPU) | Small baseline |
| Aya Expanse | 8B | Ollama (local GPU) | Arabic-focused multilingual |
| Llama 3.1 | 8B | Ollama (local GPU) | Mid-size baseline |

### 4.2 Overall Results Table
| Model | Params | CF1 | Cit.P | Cit.R | P@1 | MRR | HCR↓ | JIR↓ |
|---|---|---|---|---|---|---|---|---|
| **Gemini 2.5 Flash** | — | **0.186** | **0.189** | 0.186 | 0.254 | 0.254 | **0.008** | 0.375 |
| Llama 3.2 | 3B | 0.129 | 0.129 | 0.213 | 0.385 | 0.393 | 0.492 | 0.400 |
| Gemma 2 | 9B | 0.120 | 0.123 | 0.219 | 0.389 | 0.408 | 0.725 | **0.175** |
| Llama 3.3 | 70B | 0.100 | 0.088 | **0.300** | **0.627** | **0.656** | 0.803 | 0.350 |
| Qwen 2 | 7B | 0.085 | 0.087 | 0.194 | 0.361 | 0.383 | 0.701 | 0.675 |
| Llama 3.1 | 8B | 0.068 | 0.063 | 0.270 | 0.430 | 0.446 | 0.775 | 0.500 |
| Aya Expanse | 8B | 0.030 | 0.025 | 0.208 | 0.557 | 0.592 | 0.811 | 0.325 |

### 4.3 Infection Detail by Jurisdiction (selected models)
| Source | N | Gemma 2 | Qwen 2 | Llama 3.3 |
|---|---|---|---|---|
| French | 13 | 1/13 | 7/13 | 4/13 |
| US | 10 | 3/10 | 6/10 | 3/10 |
| Egyptian | 4 | 0/4 | 3/4 | 1/4 |
| Tunisian | 3 | 0/3 | 3/3 | 2/3 |
| UK | 2 | 0/2 | 1/2 | 0/2 |
| Gulf | 2 | 0/2 | 2/2 | 1/2 |
| DZ absence | 2 | 2/2 | 2/2 | 2/2 |
| **Total** | **40** | **7/40 (17.5%)** | **27/40 (67.5%)** | **14/40 (35%)** |

### 4.4 Key Findings (9)
1. **No model exceeds 19% Citation F1** — cannot reliably cite Algerian articles
2. **Qwen 2 imports foreign law 67.5%** — worst jurisdictional faithfulness
3. **Gemma 2 resists at 17.5%** — best at rejecting foreign concepts
4. **Gemini paradox**: near-zero hallucination (0.8%) but lowest P@1 (25%) — avoids citing articles entirely (236/244 responses have zero article citations)
5. **Aya paradox**: highest P@1 (56%) + MRR (59%) but lowest CF1 (3%) — knows the right law but cites wrong article
6. **Size ≠ quality**: Llama 3.3 70B has best P@1 (63%) but only 10% CF1
7. **US infections most successful**: at-will, punitive damages, fair use catch most models
8. **French infections subtler**: models confuse Algerian/French civil law silently
9. **DZ absence traps are hardest**: auto-saisine and municipal taxing fool ALL models

### 4.5 Qualitative Infection Examples (for paper §5.5)

**Example 1 — US infection (at-will employment):**
Qwen 2 invents "employment on benefit" as Algerian translation of US at-will doctrine, fabricates Article 140 of non-existent law. Correct: Law 90-11 Art.73 requires just cause.

**Example 2 — French infection (rent control):**
Qwen 2 cites "Rental Law No. 06/12" and "Article 49 on rental price control" — neither exists. Imports French encadrement des loyers framework.

**Example 3 — Hallucination cascade:**
Llama 3.3 70B invents "Stamp Tax Act" (قانون الطوابع), then hallucinates Articles 2-17 sequentially with plausible but fabricated content.

---

## 5. Evaluation Metrics

### Standard
- **Citation F1**: Harmonic mean of citation precision and recall against expected articles
- **Citation Precision**: % of model-cited articles that are correct
- **Citation Recall**: % of expected articles the model cited
- **Precision@1**: Whether first identified law document is correct
- **MRR**: Reciprocal rank of first correct document

---
