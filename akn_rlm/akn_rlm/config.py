"""
Central configuration for AKN-RLM.
All model names, paths, and feature flags live here.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Repository layout
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR       = PROJECT_ROOT / "data"
AKN_XML_DIR    = DATA_DIR / "akn_xml"
KG_DIR         = DATA_DIR / "kg"
BENCHMARK_DIR  = DATA_DIR / "benchmark"
INDICES_DIR    = DATA_DIR / "indices"

# Paths to built index artifacts
BM25_INDEX_PATH            = INDICES_DIR / "bm25.pkl"
SPLADE_INDEX_PATH          = INDICES_DIR / "splade.pkl"
DENSE_FAISS_PATH           = INDICES_DIR / "dense.faiss"
DENSE_META_PATH            = INDICES_DIR / "dense_meta.parquet"
COLBERT_INDEX_DIR          = INDICES_DIR / "colbert"
ARTICLE_REGISTRY_PATH      = INDICES_DIR / "article_registry.json"
TEMPORAL_INDEX_PATH        = INDICES_DIR / "temporal_index.json"

# Benchmark
BENCHMARK_PATH = BENCHMARK_DIR / "AlgerianLegalBench_v3.0_final.json"

# KG
KG_TTL_PATH = KG_DIR / "algerian_legal_kg.ttl"

# ---------------------------------------------------------------------------
# Source data fallback (use existing corpus if data/akn_xml is empty)
# ---------------------------------------------------------------------------

DATASET_ROOT_ENV = "AKN_RLM_DATASET_ROOT"
AKN_DIR_ENV = "AKN_RLM_AKN_DIR"
KG_PATH_ENV = "AKN_RLM_KG_PATH"
BENCHMARK_PATH_ENV = "AKN_RLM_BENCHMARK_PATH"


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    deduped: list[Path] = []
    for path in paths:
        key = str(path.resolve(strict=False)).lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def dataset_root_candidates() -> list[Path]:
    """Candidate dataset roots, most explicit first."""
    env_root = os.getenv(DATASET_ROOT_ENV, "").strip()
    candidates = []
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend([
        PROJECT_ROOT.parent / "new_dataset",
        PROJECT_ROOT / "new_dataset",
        PROJECT_ROOT.parent / "code" / "Dataset",
    ])
    return _dedupe_paths(candidates)


def get_dataset_root() -> Path | None:
    """Return the first detected external dataset root, if any."""
    for root in dataset_root_candidates():
        if (root / "data" / "akn").exists() or (root / "AlgerianLegalBench_v3.0_final.json").exists():
            return root
    return None


def _akn_dir_candidates() -> list[Path]:
    candidates = []
    env_dir = os.getenv(AKN_DIR_ENV, "").strip()
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.append(AKN_XML_DIR)
    for root in dataset_root_candidates():
        candidates.extend([
            root / "data" / "akn",
            root / "akn",
        ])
    return _dedupe_paths(candidates)


def _kg_path_candidates() -> list[Path]:
    candidates = []
    env_path = os.getenv(KG_PATH_ENV, "").strip()
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(KG_TTL_PATH)
    for root in dataset_root_candidates():
        candidates.extend([
            root / "data" / "rdf" / "algerian_legal_kg.ttl",
            root / "data" / "kg" / "algerian_legal_kg.ttl",
            root / "algerian_legal_kg.ttl",
        ])
    return _dedupe_paths(candidates)


def _benchmark_path_candidates() -> list[Path]:
    candidates = []
    env_path = os.getenv(BENCHMARK_PATH_ENV, "").strip()
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(BENCHMARK_PATH)
    for root in dataset_root_candidates():
        candidates.extend([
            root / "AlgerianLegalBench_v3.0_final.json",
            root / "benchmark" / "AlgerianLegalBench_v3.0_final.json",
        ])
    return _dedupe_paths(candidates)


def get_akn_dir() -> Path:
    for candidate in _akn_dir_candidates():
        if candidate.exists() and list(candidate.glob("*.xml")):
            return candidate
    searched = ", ".join(str(p) for p in _akn_dir_candidates())
    raise FileNotFoundError(f"No AKN XML files found in: {searched}")


def get_kg_path() -> Path:
    for candidate in _kg_path_candidates():
        if candidate.exists():
            return candidate
    searched = ", ".join(str(p) for p in _kg_path_candidates())
    raise FileNotFoundError(f"KG TTL not found in: {searched}")


def get_benchmark_path() -> Path:
    for candidate in _benchmark_path_candidates():
        if candidate.exists():
            return candidate
    searched = ", ".join(str(p) for p in _benchmark_path_candidates())
    raise FileNotFoundError(f"Benchmark not found in: {searched}")


# ---------------------------------------------------------------------------
# Model names
# ---------------------------------------------------------------------------

AI_GRID_API_KEY  = os.getenv("AI_GRID_API_KEY", "")
AI_GRID_BASE_URL = os.getenv("AI_GRID_BASE_URL", "http://app.ai-grid.io:4000/v1")
QWEN_API_KEY     = os.getenv("QWEN_API_KEY", AI_GRID_API_KEY)   # Qwen3-30B-A3B-Thinking
GEMMA_API_KEY    = os.getenv("GEMMA_API_KEY", AI_GRID_API_KEY)  # google/gemma-4-31B

ROOT_LLM_MODEL = os.getenv("ROOT_LLM_MODEL", "gpt-oss-120b")
SUB_LLM_MODEL  = os.getenv("SUB_LLM_MODEL",  "Qwen3-30B-A3B-Thinking")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL",  "Qwen3-30B-A3B-Thinking")

EMBED_MODEL    = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
COLBERT_MODEL  = "BAAI/bge-m3"   # multi-vector head (disabled — OOM on low-RAM)
SPLADE_MODEL   = "BAAI/bge-m3"   # sparse head (disabled — OOM on low-RAM)

VERTEX_PROJECT  = os.getenv("VERTEX_PROJECT", "pfe-api-494023")
VERTEX_LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")

# Vertex AI service-account key — set GOOGLE_APPLICATION_CREDENTIALS so the
# google-auth SDK picks it up automatically when Vertex clients are instantiated.
_SA_KEY = os.getenv(
    "GOOGLE_APPLICATION_CREDENTIALS",
    str(Path(__file__).resolve().parent.parent.parent / "pfe-api-494023-b2aa0c91575c.json"),
)
if Path(_SA_KEY).exists() and not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = _SA_KEY

# ---------------------------------------------------------------------------
# RLM recursion limits
# ---------------------------------------------------------------------------

MAX_RECURSION_DEPTH       = 1          # hard cap; only multi_hop may use depth 2
MULTI_HOP_MAX_DEPTH       = 2
MAX_TOKENS_PER_ROOT_CALL  = 32_000
MAX_TOKENS_PER_SUB_CALL   = 8_000
MAX_CORRECTIVE_RETRIES    = 3

# ---------------------------------------------------------------------------
# Retrieval settings
# ---------------------------------------------------------------------------

DEFAULT_RETRIEVAL_K = 20
RERANK_TOP_K        = 10
RRF_K               = 60              # RRF rank fusion constant

# ---------------------------------------------------------------------------
# Feature flags
# ---------------------------------------------------------------------------

ENABLE_DENSE    = True    # re-enabled: multilingual-e5-small (471 MB) fits in RAM
ENABLE_SPLADE   = False   # disabled: bge-m3 multi-vector encoder causes C-level OOM
ENABLE_COLBERT  = False   # disabled: same bge-m3 OOM issue
ENABLE_GRAPHRAG = True
ENABLE_TEMPORAL = True

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
