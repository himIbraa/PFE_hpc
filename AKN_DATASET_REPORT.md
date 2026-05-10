# AKN Dataset Results

Generated on **2026-05-02**.  
**Conflicts resolved and registry implemented: 2026-05-04.**

## Summary

| Metric | Value |
|---|---:|
| Live AKN XML files | 45 |
| Live per-document RDF TTL files | 45 |
| Total RDF TTL files | 46 |
| Unique canonical document IDs after normalization | 44 |
| Canonical ID collisions | 1 |
| Files processed in live RDF extraction report | 45 |
| KG triples | 765,215 |
| Articles (raw parsed) | 9,004 |
| Articles (indexable, non-empty) | 8,925 |
| Terms | 12,028 |
| Definitions | 611 |
| Rights | 2,592 |
| Obligations | 6,649 |
| Conditions | 5,253 |
| Permissions | 2,806 |
| Prohibitions | 436 |
| Total amendment events | 9,091 |
| Article versions | 9,075 |

## Conflict Resolution Status

All 4 data problems from the original report have been resolved architecturally in
`akn_rlm/corpus/article_registry.py`. No XML files were modified.

### Problem 1 — Wrong file under `06-01_2006-02-20.xml`  ✓ RESOLVED

**Original:** filename claims Anti-Corruption Law 06-01 but XML FRBR metadata + content = AML law 05-01.

**Resolution:**
- `FILENAME_COLLISIONS = {"06-01_2006-02-20": ("06-01_2006-02-20", "05-01_2005-02-06")}`
- Registry emits a one-time WARNING at init: `"06-01_2006-02-20.xml content is AML law 05-01, not Anti-Corruption 06-01"`
- `resolve_alias("anti_corruption")` → `None` (abstain signal)
- `resolve_alias("06-01_2006-02-20")` → `None` (abstain signal)
- `resolve_alias("05-01_2005-02-06")` → `"05-01_2005-02-06"` (AML law accessible normally)
- Any benchmark question requiring Anti-Corruption Law citations triggers mandatory abstention
- Real Anti-Corruption Law 06-01 added to `MISSING_FROM_CORPUS` set

### Problem 2 — Missing `constitution_2020.xml` alias  ✓ RESOLVED

**Original:** file exists but was missing from `SOURCE_TO_XML_ID` alias map.

**Resolution:** `_STATIC_ALIASES` in `article_registry.py` includes all constitutional texts:
- `"2020"` → `"2020_2020-12-30"` (current constitution)
- `"constitution"` / `"const"` → `"2020_2020-12-30"`
- All historical constitutions accessible by their canonical IDs

### Problem 3 — Registry coverage gap (29 vs 44 canonical IDs)  ✓ RESOLVED

**Original:** `coverage.py` alias dict covered only 29 of 44 canonical IDs.

**Resolution:** `_STATIC_ALIASES` in `article_registry.py` covers all 44 in-corpus documents
plus common abbreviations for all 23 benchmark legal categories:
- Arabic abbreviations: قمم (Civil Code), قسرة (Family Code), قتجارة (Commercial Code), etc.
- French abbreviations: Cciv, Cfam, Ccom, CPCA, CPP, Cinv, Const, etc.
- Full Arabic names for each law
- Numeric shortcuts: "75-58", "84-11", "66-156", etc.
- Coverage verified: all 30 `expected_documents` in AlgerianLegalBench v3.0 resolve correctly
  except the 7 flagged as missing/collision

### Problem 4 — Internal metadata/type mismatches  ✓ RESOLVED (non-fatal)

**Original:** 3 files have docNumber/type inconsistencies between filename, FRBR, and AKN root element.

**Resolution:** `METADATA_MISMATCHES` dict documents all 3; registry logs a one-time WARNING per
file at build time but continues loading normally. Articles from these docs are fully accessible.

| File | Mismatch | Effect |
|---|---|---|
| `15-247_2015-09-16.xml` | docNumber says presidential-decree, root uses `<act name='law'>` | WARNING only |
| `12-2003_2012-11-28.xml` | filename `12-2003` but FRBR id `03-12_2012-11-28` | WARNING + filename alias added |
| `ordonnance-03-07-ar.xml` | title says Order 03-07, root uses `<act name='presidential-decree'>` | WARNING only |

### Problem 5 — Stale KG size comment  ✓ NOTED

The `rdf_kg.py` comment said 21.74 MB; actual live TTL is 73.44 MB (765K triples).
This is a comment-only issue in the baseline code; not relevant to `akn_rlm/`.

## Documents Missing From Corpus

The following canonical IDs are known from the benchmark but have no live AKN XML file:

| Canonical ID | Law | Impact |
|---|---|---|
| `06-01_2006-02-20` | Anti-Corruption Law 06-01 (collision) | Abstain on acor questions |
| `06-15_2006-05-11` | Law 06-15 on Prevention and Combating Trafficking | Abstain |
| `66-155_1966-06-08` | Former CPP (superseded by 25-14) | Abstain |
| `03-05_2003-07-19` | Law 03-05 | Abstain |
| `03-10_2003-07-19` | Law 03-10 | Abstain |
| `11-04_2011-02-17` | Law 11-04 | Abstain |
| `83-11_1983-07-02` | Law 83-11 | Abstain |

## Metadata Mismatch (Original Report)

| Source | AKN files | RDF files / processed docs | Triples |
|---|---:|---:|---:|
| Live dataset folders | 45 | 46 RDF files / 45 processed docs | 765,215 |
| `code/Dataset/metadata/dataset_summary.md` | 24 | 25 RDF files | - |
| `code/Dataset/metadata/extraction_report.json` | - | 23 processed docs | 258,556 |

## Script Alias Dictionary (Updated)

The canonical source of truth is now `akn_rlm/corpus/article_registry.py::_STATIC_ALIASES`
(199 keys). The table below shows the primary entries.

| English alias | Canonical doc id | Live file | Status |
|---|---|---|---|
| Civil Code / Cciv | `75-58_1975-09-26` | `75-8_1975-09-26.xml` | OK |
| Family Code / Cfam | `84-11_1984-06-09` | `84-11_1984-06-09.xml` | OK |
| Commercial Code / Ccom | `75-59_1975-09-26` | `1975_1975-09-26.xml` | OK |
| Civil Procedure / CPCA | `08-09_2008-02-25` | `08-09_2008-02-25.xml` | OK |
| Criminal Procedure / CPP | `25-14_2025-08-03` | `25-14_2025-08-03.xml` | OK |
| Penal Code | `66-156_1966-06-08` | `6-5_1966-06-08.xml` | OK |
| Investment Law / Cinv | `22-18_2022-07-24` | `22-18_2022-07-24.xml` | OK |
| AML 2005 | `05-01_2005-02-06` | `05-01_2005-02-06.xml` | OK |
| AML 2012 | `03-12_2012-11-28` | `12-2003_2012-11-28.xml` | OK (FRBR mismatch logged) |
| Anti-Corruption | `06-01_2006-02-20` | `06-01_2006-02-20.xml` | COLLISION -> abstain |
| Constitution / Const | `2020_2020-12-30` | `2020_2020-12-30.xml` | OK |
| Nationality Code | `70-86_1970-12-15` | `70-86_1970-12-15.xml` | OK |
| Military Justice | `71-28_1971-04-22` | `13-12_1971-04-22.xml` | OK |
| Prison Organisation | `05-04_2005-02-06` | `05-2004_2005-02-06.xml` | OK |
| Labour Law | `90-11_1990-04-21` | `labor_law_90-11.xml` | OK |
| Former CPP 66-155 | — | — | MISSING -> abstain |




Step 1 — Build indices (GPU step, run once)                                                                                 
                                                                                                                              
  cd D:\TRY_AGAIN\akn_rlm                                                                                                     
  conda run -n pfe_env python scripts/build_indices.py --quick-check                                                          

  This builds BM25 + SPLADE + Dense (FAISS) + ColBERT and runs a sanity check that Family Code Art. 54 lands in top-10 for    
  each retriever. Expect 30–90 min depending on GPU. If an index already exists it is skipped; use --force to rebuild.        
                                                                                                                              
  You can also build one at a time:                                                                                           
  conda run -n pfe_env python scripts/build_indices.py --index bm25                                                           
  conda run -n pfe_env python scripts/build_indices.py --index dense --quick-check                                            

  ---
  Step 2 — Full benchmark run (long-running, keep terminal open)

  conda run -n pfe_env python scripts/run_benchmark.py --run-id run_001

  This runs all 244 questions through the full pipeline (RLM + 3 gates + corrective retry). With LLM API calls it will take
  2–6 hours. All output lands in eval_results/run_001/:

  ┌───────────────────┬────────────────────────────────────────────────────────────────────────────┐
  │       File        │                                  Contents                                  │
  ├───────────────────┼────────────────────────────────────────────────────────────────────────────┤
  │ predictions.jsonl │ One JSON per question — answer, citations, latency, gate_results, HCR, JIR │
  ├───────────────────┼────────────────────────────────────────────────────────────────────────────┤
  │ metrics.json      │ Full stratified metrics tree (all strata, all Phase I metrics)             │
  ├───────────────────┼────────────────────────────────────────────────────────────────────────────┤
  │ metrics.md        │ Markdown table for quick review                                            │
  ├───────────────────┼────────────────────────────────────────────────────────────────────────────┤
  │ report.txt        │ Human report with Δ vs baseline + ✓/✗ target markers                       │
  └───────────────────┴────────────────────────────────────────────────────────────────────────────┘

  Cheaper smoke test first (20 questions, ~15 min):
  conda run -n pfe_env python scripts/run_benchmark.py --limit 20 --run-id smoke_01

  Only answerable questions (204 questions, skips unanswerable):
  conda run -n pfe_env python scripts/run_benchmark.py --query-types rule_application temporal multi_hop --run-id
  answerable_only

  When the run finishes, come back and I'll do the full report — delta vs baseline, per-category breakdown, ablation
  suggestions based on where the numbers land.


  Replace the four-model story (gte-Qwen2 + SPLADE + AraBERT-ColBERT + reranker) with a two-model story: "We use bge-m3, a multilingual model purpose-built for hybrid retrieval that produces dense, sparse, and multi-vector representations from a single forward pass, paired with bge-reranker-v2-m3 for cross-encoder reranking and BM25 for exact-identifier matching."

  bge-m3 is designed as a unified multi-functionality model — it produces dense + sparse (SPLADE-style) + multi-vector (ColBERT-style) representations from a single forward pass. It's built on XLM-RoBERTa, was trained on 100+ languages including Arabic, and the Chen et al. 2024 paper shows its sparse mode beats BM25 across all evaluated languages, and its multi-vector mode adds further lift when used to re-rank top-200 dense candidates. There's also a published 2025 study by Alsubhi et al. specifically on Arabic RAG showing bge-m3 outperforms general multilingual alternatives. 