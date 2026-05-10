#!/bin/bash
#SBATCH --job-name=akn_build_idx
#SBATCH --output=logs/build_indices_%j.out
#SBATCH --error=logs/build_indices_%j.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# ---------------------------------------------------------------------------
# Build BM25 + BGE-m3 dense FAISS indices over the 8998-chunk corpus.
# ---------------------------------------------------------------------------
# Hardware requirements:
#   - 1 GPU with >= 16 GB VRAM (A100 40 GB / V100 32 GB / RTX 6000 ideal)
#   - 64 GB system RAM (BGE-m3 at fp16 + corpus + FAISS build peak)
#   - 30 GB scratch (corpus + indices + HF cache)
# Wall-clock: ~15-25 min on A100 (BGE-m3 fp16 encode of 8998 chunks).
# ---------------------------------------------------------------------------

set -euo pipefail

mkdir -p logs

# 1. Conda env
source ~/miniconda3/etc/profile.d/conda.sh
conda activate akn_rlm_hpc

# 2. Cache + model selection
export HF_HOME=${HF_HOME:-$SCRATCH/hf_cache}
mkdir -p "$HF_HOME"

export EMBED_MODEL="BAAI/bge-m3"
export RERANKER_MODEL="BAAI/bge-reranker-v2-m3"

# 3. Repo root (this file lives in <repo>/hpc/)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT/akn_rlm"

# 4. Build indices. The dense.py module is model-agnostic; setting
#    EMBED_MODEL=BAAI/bge-m3 makes it embed the corpus with 1024-dim BGE-m3
#    vectors instead of 384-dim e5-small.
echo "[build_indices] Repo root: $REPO_ROOT"
echo "[build_indices] EMBED_MODEL=$EMBED_MODEL"
echo "[build_indices] RERANKER_MODEL=$RERANKER_MODEL"
echo "[build_indices] HF_HOME=$HF_HOME"

python scripts/build_indices.py --force --index bm25
python scripts/build_indices.py --force --index dense

# 5. Sanity check the new dense index dim
python - <<'PY'
import faiss, pathlib
p = pathlib.Path("data/indices/dense.faiss")
idx = faiss.read_index(str(p))
print(f"[build_indices] dense.faiss vectors={idx.ntotal} dim={idx.d}")
print(f"[build_indices] expected dim=1024 for BGE-m3, dim=384 for e5-small")
PY

echo "[build_indices] DONE"
