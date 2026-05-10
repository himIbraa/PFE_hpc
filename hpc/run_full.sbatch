#!/bin/bash
#SBATCH --job-name=akn_full_244
#SBATCH --output=logs/run_full_%j.out
#SBATCH --error=logs/run_full_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# ---------------------------------------------------------------------------
# Run the RLM dispatcher on the full 244-q ALB v3.0 benchmark with all 5
# ceiling-breaker upgrades enabled:
#   1. BGE-m3 dense retriever        (env: EMBED_MODEL=BAAI/bge-m3)
#   2. BGE-reranker-v2-m3            (env: RERANKER_MODEL=BAAI/bge-reranker-v2-m3)
#   3. Per-citation NLI verifier     (--ceiling-breakers in run_dispatcher.py)
#   4. Doc-router LLM tie-breaker    (--ceiling-breakers)
#   5. Concept->amendment SPARQL     (--ceiling-breakers; helper exposed)
# ---------------------------------------------------------------------------
# Hardware requirements:
#   - 1 GPU with >= 16 GB VRAM (BGE-m3 + BGE-reranker-v2 in fp16; rdflib in RAM)
#   - 64 GB system RAM (758k-triple KG + 8998-chunk corpus + dense index + reranker)
#   - 30 GB scratch
# Wall-clock: ~25-40 min on A100 (244 q × ~5-8 s/q with reranker on GPU).
# ---------------------------------------------------------------------------

set -euo pipefail

mkdir -p logs

source ~/miniconda3/etc/profile.d/conda.sh
conda activate akn_rlm_hpc

export HF_HOME=${HF_HOME:-$SCRATCH/hf_cache}
export EMBED_MODEL="BAAI/bge-m3"
export RERANKER_MODEL="BAAI/bge-reranker-v2-m3"
export AKN_CEILING_BREAKERS=1

# AI-Grid LLM API credentials. Either export AI_GRID_API_KEY before sbatch,
# or place it in akn_rlm/.env. The repo's .gitignore excludes .env.
if [ -z "${AI_GRID_API_KEY:-}" ]; then
  echo "[run_full] WARN: AI_GRID_API_KEY not set in env; will rely on akn_rlm/.env"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/akn_rlm"

RUN_ID="rlm_dispatched_full_ceiling_${SLURM_JOB_ID:-manual}"

echo "[run_full] Repo root: $REPO_ROOT"
echo "[run_full] EMBED_MODEL=$EMBED_MODEL"
echo "[run_full] RERANKER_MODEL=$RERANKER_MODEL"
echo "[run_full] AKN_CEILING_BREAKERS=$AKN_CEILING_BREAKERS"
echo "[run_full] run_id=$RUN_ID"
echo "[run_full] $(date)  starting full 244-q dispatcher run"

python scripts/run_dispatcher.py \
  --ceiling-breakers \
  --run-id "$RUN_ID"

echo "[run_full] $(date)  dispatcher run done"

# Build comparison table side-by-side with the F5 baseline shipped in repo.
python scripts/compare_baselines.py \
  --runs "baseline_bm25_full,baseline_dense_full,baseline_hybrid_full,baseline_hybrid_rerank_full,baseline_kg_full,baseline_kg_hybrid_full,rlm_dispatched_full,${RUN_ID}" \
  --out "eval_results/comparison_ceiling_full.md" --no-stdout

echo "[run_full] DONE — results under eval_results/${RUN_ID}/ and eval_results/comparison_ceiling_full.md"
