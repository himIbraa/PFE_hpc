# AKN-RLM HPC run — ceiling-breakers

Run the AKN-RLM dispatcher on the full 244-q ALB v3.0 benchmark with the
five ceiling-breaker upgrades documented in `HANDOFF.md` §"What would
actually break the ceiling":

1. **BGE-m3 dense retriever** (1024-dim, replaces e5-small at 384-dim)
2. **BAAI/bge-reranker-v2-m3** (Arabic-capable, replaces mMARCO mMiniLM)
3. **Per-citation NLI as live verifier** (replaces the LLM verifier in
   rule_application / multi_hop / exact_article)
4. **Doc-router LLM tie-breaker** (gpt-oss-120b judges top-N candidates
   on alias-channel ties)
5. **Concept→amendment SPARQL helper** (catches definitions that live
   inside amending decrees)

Activation: a single CLI flag (`--ceiling-breakers`) or env var
(`AKN_CEILING_BREAKERS=1`).

---

## 1 — Hardware requirements

| Resource | Build indices | Run benchmark |
|---|---|---|
| **GPU** | 1× ≥16 GB VRAM (A100 40 GB / V100 32 GB / RTX 6000 ideal) | same |
| **System RAM** | **64 GB** (BGE-m3 fp16 + corpus + FAISS build peak) | **64 GB** (758k-triple rdflib KG + dense index + reranker + corpus) |
| **Scratch / disk** | 30 GB (HF cache for BGE-m3 ≈ 2.3 GB + reranker ≈ 2.3 GB + corpus + indices) | same |
| **Wall-clock** | 15–25 min on A100 | 25–40 min on A100 (244 q × 5–8 s/q) |
| **Network** | First run downloads ~5 GB from HuggingFace | none after caches warm |

**Why 64 GB RAM, not 16 GB:** the F5 result was capped on a 16 GB Windows
laptop because BGE-m3 + the 758k-triple rdflib KG + the dense index can't
co-reside in 16 GB. On HPC with 64 GB the entire pipeline lives in memory
with headroom; this is the constraint that actually gates the +0.05 Cite F1
ceiling on 244 q.

A 32 GB node will work if the KG handlers (`temporal_factual`,
`conceptual_definitional`) are run in a separate job after the rest, but
it's easier to just request 64 GB.

---

## 2 — Install (one-time)

```bash
# Clone the HPC repo
git clone https://github.com/himIbraa/PFE_hpc.git akn_rlm_hpc
cd akn_rlm_hpc

# Create the conda env (~5 minutes; downloads PyTorch + CUDA wheels)
conda env create -f hpc/environment.yml

# Activate
conda activate akn_rlm_hpc

# Install the akn_rlm package itself in editable mode.
# IMPORTANT: --no-deps tells pip to skip the dependency list inside
# pyproject.toml — those deps (torch, faiss, transformers, …) are already
# installed in the conda env from environment.yml. Without --no-deps pip
# would re-install CPU-only wheels on top of the conda CUDA build.
pip install -e ./akn_rlm --no-deps

# Configure LLM API credentials. Either export in your shell rc:
echo 'export AI_GRID_API_KEY="your_key_here"' >> ~/.bashrc
echo 'export AI_GRID_BASE_URL="http://app.ai-grid.io:4000/v1"' >> ~/.bashrc
# OR put them in akn_rlm/.env (gitignored). dotenv is loaded by config.py.
cat > akn_rlm/.env <<'EOF'
AI_GRID_API_KEY=your_key_here
AI_GRID_BASE_URL=http://app.ai-grid.io:4000/v1
EOF

# Smoke check that the env imports without errors
python -c "from akn_rlm.rlm.dispatcher import build_dispatcher; print('imports OK')"

# Run the existing 762-test pytest suite (should pass on any node with the
# conda env, no GPU / no LLM creds needed for unit tests)
cd akn_rlm
python -m pytest akn_rlm/tests/ -q
cd ..
```

---

## 3 — Build the indices (must run BEFORE the benchmark)

The repo ships with the **F5 baseline** indices (e5-small dense FAISS,
14 MB) inside `akn_rlm/data/indices/`. To swap to BGE-m3 you have to
re-embed the corpus.

```bash
# Submit on SLURM (see hpc/build_indices.sbatch for resource spec)
sbatch hpc/build_indices.sbatch

# Watch progress
squeue -u $USER
tail -f logs/build_indices_<jobid>.out
```

Or run interactively on a GPU node:

```bash
salloc --partition=gpu --gres=gpu:1 --mem=64G --cpus-per-task=8 --time=2:00:00
conda activate akn_rlm_hpc
export EMBED_MODEL=BAAI/bge-m3
export RERANKER_MODEL=BAAI/bge-reranker-v2-m3
cd akn_rlm
python scripts/build_indices.py --force --index bm25
python scripts/build_indices.py --force --index dense
```

The script prints `dense.faiss vectors=8998 dim=1024` on success (the F5
baseline was `dim=384`). If you see `dim=384` after build, the
`EMBED_MODEL` env var didn't take — check the env activation order.

---

## 4 — Run the full benchmark

```bash
# SLURM (recommended)
sbatch hpc/run_full.sbatch

# Interactive
salloc --partition=gpu --gres=gpu:1 --mem=64G --cpus-per-task=8 --time=4:00:00
conda activate akn_rlm_hpc
export EMBED_MODEL=BAAI/bge-m3
export RERANKER_MODEL=BAAI/bge-reranker-v2-m3
export AKN_CEILING_BREAKERS=1
cd akn_rlm
python scripts/run_dispatcher.py --ceiling-breakers --run-id rlm_dispatched_full_ceiling
```

Stratified smoke first (40 q, ~3-5 min — sanity-check on ALB v3.0
strata before the 244-q run):

```bash
python scripts/run_dispatcher.py --ceiling-breakers --stratified 5 \
    --run-id rlm_dispatched_strat5_ceiling
```

After the full run, build the comparison table:

```bash
python scripts/compare_baselines.py \
    --runs "baseline_bm25_full,baseline_dense_full,baseline_hybrid_full,baseline_hybrid_rerank_full,baseline_kg_full,baseline_kg_hybrid_full,rlm_dispatched_full,rlm_dispatched_full_ceiling" \
    --out eval_results/comparison_ceiling_full.md --no-stdout
```

---

## 5 — Files to send back for analysis

After the full run finishes, copy these from HPC and share them so we
can analyse together:

```
eval_results/rlm_dispatched_full_ceiling/metrics.json        # overall + per-stratum metrics
eval_results/rlm_dispatched_full_ceiling/metrics.md          # human-readable report
eval_results/rlm_dispatched_full_ceiling/predictions.jsonl   # per-question telemetry (sub_call_count, calls_by_model, dispatched_handler, supervisor_used)
eval_results/comparison_ceiling_full.md                      # 8-pipeline thesis table side-by-side with F5 baseline
logs/run_full_<jobid>.out                                    # any warnings (esp. NLI model load, BGE-m3 dim, reranker fallback)
```

Optional but useful:

```
eval_results/rlm_dispatched_strat5_ceiling/metrics.json      # if smoke ran
data/indices/dense.faiss.model.txt                           # records which embedding model built the index ("BAAI/bge-m3" expected)
```

A copy-paste tarball:

```bash
cd akn_rlm
tar czf ~/akn_ceiling_results.tgz \
    eval_results/rlm_dispatched_full_ceiling/ \
    eval_results/rlm_dispatched_strat5_ceiling/ \
    eval_results/comparison_ceiling_full.md \
    data/indices/dense.faiss.model.txt
ls -lh ~/akn_ceiling_results.tgz
# scp this back to your laptop
```

---

## 6 — What we'll be looking for in the results

1. **Overall Cite F1**: F5 = 0.301. Target with ceiling-breakers: **≥ 0.40**
   (the analytic max given R@10 art ceiling lifted by BGE-m3).
2. **MRR art**: F5 = 0.269. Expected lift mostly from BGE-m3 + reranker
   (steps 1+2). Target: **≥ 0.35**.
3. **R@10 art**: F5 = 0.216 → if it doesn't move, BGE-m3 isn't actually
   loading. Sanity: check `dense.faiss.model.txt` says `BAAI/bge-m3` and
   FAISS dim is 1024.
4. **Per-handler `calls_by_model` in predictions.jsonl**: should now show
   gpt-oss-120b calls from the doc-router LLM tie-breaker (Step 4) — F5
   had zero gpt-oss-120b calls because the supervisor never fired.
5. **HCR (hallucination)**: must stay at 0.0. If it rises, the NLI
   verifier (Step 3) is admitting marginal candidates the LLM verifier
   would have rejected — we'd revert it for handlers where it hurts.
6. **Per-handler regression list**: any stratum where ceiling > F5 by
   ≥0.02 is a real lift; any stratum where ceiling < F5 by ≥0.02 is a
   regression to investigate.

If overall Cite F1 lands in the 0.32–0.40 range we know which step to
ablate next. If it lands ≥ 0.40 the thesis chapter gets a fresh
"HPC-grade" results section alongside the F5 16 GB-RAM result.

---

## 7 — Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `BGE-m3 not found` on first encode | HF_HOME unwritable / no internet on compute node | `export HF_HOME=$SCRATCH/hf_cache; mkdir -p $HF_HOME`. Pre-download on a login node: `python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"` |
| `dense.faiss vectors=8998 dim=384` after rebuild | `EMBED_MODEL` env not set when `build_indices.py` ran | Re-export and re-run `python scripts/build_indices.py --force --index dense` |
| Reranker silently degrades to RRF order | bge-reranker-v2-m3 download failed | check logs/build_indices_*.err for HuggingFace 401/403; some clusters block outbound HTTPS — pre-download on login node |
| `dispatch_build_error` for temporal_factual | KG TTL not found | `ls akn_rlm/data/kg/*.ttl` — corpus loader will look in `new_dataset/data/rdf/` automatically; otherwise set `AKN_RLM_KG_PATH=/path/to/algerian_legal_kg.ttl` |
| `AI_GRID_API_KEY missing` on first LLM dispatch | `.env` not loaded | put it in `akn_rlm/.env` OR `export` before `sbatch` |
| OOM during BGE-m3 encode | batch_size 16 default too high for older 16 GB GPUs | edit `scripts/build_indices.py` — pass `batch_size=8` to `DenseIndex.build` |

---

## 8 — Reverting to the F5 baseline

If the ceiling-breakers regress on the full run, you can A/B against F5
from the same env without re-installing:

```bash
unset AKN_CEILING_BREAKERS
unset EMBED_MODEL  # falls back to intfloat/multilingual-e5-small
unset RERANKER_MODEL  # falls back to cross-encoder/mmarco-mMiniLMv2-L12-H384-v1

# Rebuild the e5-small dense index (BM25 unchanged)
cd akn_rlm
python scripts/build_indices.py --force --index dense

# Re-run dispatcher without --ceiling-breakers
python scripts/run_dispatcher.py --run-id rlm_dispatched_full_f5_repro
```

The two `metrics.json` files (`rlm_dispatched_full_ceiling` vs
`rlm_dispatched_full_f5_repro`) give a clean head-to-head on the same
HPC environment.
