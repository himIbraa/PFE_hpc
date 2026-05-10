"""Shared BGEM3FlagModel loader with process-level cache.

Both the sparse and colbert indexers import get_bge_m3() so the model
is loaded only once per process even when building all indices in sequence.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

log = logging.getLogger(__name__)

_CACHE: dict[str, object] = {}


def get_bge_m3(model_name: str = "BAAI/bge-m3", use_fp16: bool = True):
    """Return a cached BGEM3FlagModel instance.

    use_fp16 is attempted first; if FlagEmbedding/transformers version mismatch
    causes a TypeError (dtype kwarg bug), falls back to fp32 automatically.
    """
    if model_name not in _CACHE:
        os.environ.setdefault("SAFETENSORS_FAST_GPU", "1")
        from FlagEmbedding import BGEM3FlagModel  # type: ignore
        try:
            log.info("Loading BGEM3FlagModel: %s  (fp16=%s)", model_name, use_fp16)
            _CACHE[model_name] = BGEM3FlagModel(model_name, use_fp16=use_fp16)
        except TypeError:
            log.warning("fp16 load failed (FlagEmbedding/transformers mismatch) — retrying fp32")
            _CACHE[model_name] = BGEM3FlagModel(model_name, use_fp16=False)
    return _CACHE[model_name]
