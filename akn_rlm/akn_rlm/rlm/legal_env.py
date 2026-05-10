"""LegalEnv — the Python REPL environment exposed to the root RLM.

Every public method is callable from root-LM-generated code.
All return values are JSON-serialisable.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any

from akn_rlm.config import DEFAULT_RETRIEVAL_K, RERANK_TOP_K
from akn_rlm.gates import jurisdiction as jurisdiction_gate
from akn_rlm.normalizers import normalize_arabic
from akn_rlm.retrievers import hybrid_fusion
from akn_rlm.rlm.recursion_budget import RecursionBudget, RecursionBudgetExceeded

log = logging.getLogger(__name__)


class LegalEnv:
    """The REPL environment the root RLM operates over.

    Every method here is callable from generated Python code.
    All return values are JSON-serialisable.
    """

    def __init__(
        self,
        registry,           # ArticleRegistry
        kg,                 # rdflib.Graph
        indices: dict,      # {"bm25": BM25Index, "dense": DenseIndex, ...}
        llm_pool,           # LLMPool
        budget: RecursionBudget,
        temporal_index: dict | None = None,
    ) -> None:
        self.registry = registry
        self.kg = kg
        self.indices = indices
        self.llm_pool = llm_pool
        self.budget = budget
        self.temporal_index = temporal_index or {}
        self.cache: dict[str, Any] = {}

    def begin_query(self, *, clear_cache: bool = True) -> None:
        """Reset per-query mutable state before a new top-level pipeline run."""
        self.budget.reset()
        self._degraded_retrievers: list[str] = []
        if clear_cache:
            self.cache.clear()

    # ------------------------------------------------------------------
    # Internal cache helpers
    # ------------------------------------------------------------------

    def _cache_key(self, method: str, *args) -> str:
        raw = f"{method}:{args!r}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()[:16]

    def _cached(self, key: str, fn):
        if key not in self.cache:
            self.cache[key] = fn()
        return self.cache[key]

    # ------------------------------------------------------------------
    # Retrieval primitives
    # ------------------------------------------------------------------

    def search_dense(self, query: str, k: int = DEFAULT_RETRIEVAL_K) -> list[dict]:
        """Dense vector search (bge-m3 CLS embeddings + FAISS)."""
        key = self._cache_key("dense", query, k)
        def _fn():
            idx = self.indices.get("dense")
            if idx is None:
                return []
            hits = idx.search(query, k=k)
            return [self._hit_to_dict(h, "dense") for h in hits]
        return self._cached(key, _fn)

    def search_bm25(self, query: str, k: int = DEFAULT_RETRIEVAL_K) -> list[dict]:
        """BM25 lexical search with Arabic prefix-stripping."""
        key = self._cache_key("bm25", query, k)
        def _fn():
            idx = self.indices.get("bm25")
            if idx is None:
                return []
            hits = idx.search(query, k=k)
            return [self._hit_to_dict(h, "bm25") for h in hits]
        return self._cached(key, _fn)

    def search_splade(self, query: str, k: int = DEFAULT_RETRIEVAL_K) -> list[dict]:
        """SPLADE sparse retrieval."""
        key = self._cache_key("splade", query, k)
        def _fn():
            idx = self.indices.get("splade")
            if idx is None:
                return []
            hits = idx.search(query, k=k)
            return [self._hit_to_dict(h, "splade") for h in hits]
        return self._cached(key, _fn)

    def search_colbert(self, query: str, k: int = DEFAULT_RETRIEVAL_K) -> list[dict]:
        """ColBERT late-interaction retrieval (AraBERT)."""
        key = self._cache_key("colbert", query, k)
        def _fn():
            idx = self.indices.get("colbert")
            if idx is None:
                return []
            hits = idx.search(query, k=k)
            return [self._hit_to_dict(h, "colbert") for h in hits]
        return self._cached(key, _fn)

    def search_hybrid(
        self,
        query: str,
        k: int = DEFAULT_RETRIEVAL_K,
        weights: dict | None = None,
        rerank: bool = True,
    ) -> list[dict]:
        """RRF fusion over all available retrievers, then bge-reranker reranking.

        Fuses BM25 + dense + sparse + colbert top-(k*5) via RRF,
        then reranks the fused top-50 with bge-reranker-v2-m3 down to k.

        weights: {"bm25": float, "dense": float, "splade": float, "colbert": float}
        rerank:  set False to skip reranking (faster, lower quality).
        """
        names = ["bm25", "dense", "splade", "colbert"]
        w_map = weights or {}
        result_lists, w_list = [], []
        degraded: list[str] = []
        for name in names:
            if self.indices.get(name) is None:
                degraded.append(name)
                log.debug("search_hybrid: %s index not loaded — arm missing", name)
                continue
            try:
                hits = getattr(self, f"search_{name}")(query, k=k * 5)
                result_lists.append(hits)
                w_list.append(float(w_map.get(name, 1.0)))
            except Exception as exc:
                degraded.append(name)
                log.warning("search_hybrid: %s retriever failed (%s) — skipping", name, exc)
        if degraded:
            log.info("search_hybrid: degraded arms=%s (active=%d/4)", degraded, len(result_lists))
            # Persist per-query degradation list for telemetry
            existing = getattr(self, "_degraded_retrievers", [])
            self._degraded_retrievers = sorted(set(existing) | set(degraded))
        if not result_lists:
            return []
        fused = hybrid_fusion.rrf_fuse(result_lists, weights=w_list)
        candidates = fused[:50]   # reranker input pool
        if rerank and candidates:
            try:
                reranked = self.rerank(query, candidates, k=k)
                return reranked
            except Exception as exc:
                log.warning("search_hybrid: reranker failed (%s) — using RRF order", exc)
        return candidates[:k]

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        k: int = RERANK_TOP_K,
    ) -> list[dict]:
        """Cross-encoder reranking using BGE-reranker-v2-m3."""
        from akn_rlm import reranker
        return reranker.rerank(query, candidates, k=k)

    # ------------------------------------------------------------------
    # KG primitives
    # ------------------------------------------------------------------

    def kg_sparql(self, query: str) -> list[dict]:
        """Execute a SPARQL SELECT against the in-memory KG."""
        if self.kg is None:
            log.debug("kg_sparql: KG not loaded — returning []")
            return []
        from akn_rlm.retrievers import graphrag
        return graphrag.sparql_query(self.kg, query)

    def kg_entity_lookup(self, surface: str) -> list[dict]:
        """Find KG entities matching a surface form (label substring)."""
        if self.kg is None:
            log.debug("kg_entity_lookup: KG not loaded — returning []")
            return []
        from akn_rlm.retrievers import graphrag
        return graphrag.entity_lookup(self.kg, surface)

    def kg_amendment_chain(self, article_id: str) -> list[dict]:
        """Return all historical versions of an article, ordered by date.

        article_id: either "doc_id#eid" or a full URI.
        """
        if self.kg is None:
            log.debug("kg_amendment_chain: KG not loaded — returning []")
            return []
        from akn_rlm.retrievers import graphrag
        if not article_id.startswith("http"):
            article_id = f"http://legislation.dz/id/{article_id}"
        return graphrag.amendment_chain(self.kg, article_id)

    def kg_get_article_at_date(
        self,
        article_id: str,
        date: str,
    ) -> dict | None:
        """Return article text as it stood on date (YYYY-MM-DD).

        article_id: "doc_id/eid" format (e.g. "84-11_1984-06-09/art_54").
        """
        from akn_rlm.retrievers.temporal import get_article_at_date
        if "/" in article_id:
            doc_id, eid = article_id.split("/", 1)
        elif "#" in article_id:
            doc_id, eid = article_id.split("#", 1)
        else:
            return None
        return get_article_at_date(self.temporal_index, doc_id, eid, date)

    # ------------------------------------------------------------------
    # Article-level operations
    # ------------------------------------------------------------------

    def get_article(self, doc_id: str, article_ref: str) -> dict | None:
        """Retrieve a single article by doc_id + article_ref.

        Returns None if the article doesn't exist in the registry.
        """
        entry = self.registry.get_doc(doc_id)
        if entry is None:
            # Try alias resolution
            canonical = self.registry.resolve_alias(doc_id)
            if canonical:
                entry = self.registry.get_doc(canonical)
                doc_id = canonical
        if entry is None:
            return None
        if not self.registry.has_article(doc_id, article_ref):
            return None
        # Pull text from the index (BM25 meta has truncated text; prefer dense)
        from akn_rlm.normalizers import ref_to_eid
        eid = ref_to_eid(article_ref)
        chunk_id = f"{doc_id}#{eid}"
        # Try BM25 meta first (always available)
        bm25_idx = self.indices.get("bm25")
        if bm25_idx:
            for m in bm25_idx._meta:
                if m["chunk_id"] == chunk_id:
                    return {
                        "doc_id":     doc_id,
                        "article_ref":article_ref,
                        "eid":        eid,
                        "text":       m["text"],
                        "doc_title":  entry.doc_title if hasattr(entry, "doc_title") else "",
                    }
        return {
            "doc_id":     doc_id,
            "article_ref":article_ref,
            "eid":        eid,
            "text":       "",
            "doc_title":  getattr(entry, "doc_title", ""),
        }

    def list_documents(self) -> list[dict]:
        """Return registry contents as a list of doc metadata dicts."""
        docs = []
        for doc_id, entry in self.registry._docs.items():
            docs.append({
                "doc_id":       doc_id,
                "doc_title":    getattr(entry, "doc_title", ""),
                "doc_date":     getattr(entry, "doc_date", ""),
                "article_count":len(getattr(entry, "article_eids", set())),
            })
        return sorted(docs, key=lambda d: d["doc_id"])

    def has_article(self, doc_id: str, article_ref: str) -> bool:
        """Hard gate: True iff article exists in registry."""
        return self.registry.has_article(doc_id, article_ref)

    # ------------------------------------------------------------------
    # Recursive sub-LM call (THE RLM PRIMITIVE)
    # ------------------------------------------------------------------

    def llm_call(
        self,
        prompt: str,
        purpose: str = "verify",
        context_articles: list[dict] | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
    ) -> str:
        """Spawn a depth+1 LM call with bounded context.

        purpose: "verify" | "summarize" | "decompose" | "extract_adu"
        Raises RecursionBudgetExceeded if depth or token budget is exhausted.
        """
        from akn_rlm.config import SUB_LLM_MODEL
        model = model or SUB_LLM_MODEL
        self.budget.charge(prompt, max_tokens)
        with self.budget.descend():
            system = self._system_for_purpose(purpose)
            return self.llm_pool.call(
                prompt=prompt,
                model=model,
                max_tokens=max_tokens,
                context_articles=context_articles,
                system=system,
            )

    @staticmethod
    def _system_for_purpose(purpose: str) -> str:
        _SYSTEMS = {
            "verify":      "You are a legal fact-checker. Verify whether the cited Algerian law article supports the claim. Reply with YES or NO and a brief reason.",
            "summarize":   "You are a legal summarizer. Summarize the provided Algerian law articles in 2-3 sentences, in the same language as the query.",
            "decompose":   "You are a legal analyst. Decompose the legal question into sub-questions that can each be answered by a single article.",
            "extract_adu": "You are an argumentation analyst. Extract the claim, premise, and warrant from the provided legal article text.",
        }
        return _SYSTEMS.get(purpose, "You are a helpful legal assistant for Algerian law.")

    # ------------------------------------------------------------------
    # Verification primitives (non-LM, cheap)
    # ------------------------------------------------------------------

    def verify_in_corpus(self, citation: dict) -> bool:
        """Hard gate: returns True iff citation passes has_article()."""
        return self.has_article(
            citation.get("doc_id", ""),
            citation.get("article_ref", ""),
        )

    def verify_article(
        self,
        sub_question: str,
        article: dict,
        model: str | None = None,
    ) -> dict:
        """Structured sub-LM verification of (sub_question, article).

        Returns strict-JSON dict:
          {"relevant": bool, "supporting_span": str|None,
           "contradicting_span": str|None, "confidence": float}

        Charged to the recursion budget. Use inside env.llm_call budget.
        """
        from akn_rlm.config import SUB_LLM_MODEL
        from akn_rlm.rlm.sub_worker import call_verifier
        _model = model or SUB_LLM_MODEL
        self.budget.charge(sub_question + str(article), 512)
        with self.budget.descend():
            return call_verifier(self.llm_pool, sub_question, article, _model)

    def detect_infection_signals(self, text: str) -> list[str]:
        """Scan text for foreign-law canary signals.

        Returns a list of signal IDs like ["fr:isf", "us:at_will"].
        Empty list = no detected signals.
        """
        return jurisdiction_gate.detect(text)

    # ------------------------------------------------------------------
    # ADU primitive
    # ------------------------------------------------------------------

    def extract_adu(self, article_text: str) -> dict:
        """Extract full Toulmin argumentation structure from article text.

        Returns dict with keys: claim, ground, warrant, backing, rebuttal.
        All values are strings; any key may be empty string when not applicable.
        """
        from akn_rlm.adu import extract as _adu_extract
        from akn_rlm.config import SUB_LLM_MODEL
        try:
            self.budget.charge(article_text, 512)
            with self.budget.descend():
                return _adu_extract(
                    article_text,
                    self.llm_pool,
                    model=SUB_LLM_MODEL,
                    max_tokens=512,
                )
        except RecursionBudgetExceeded as exc:
            log.warning("extract_adu: budget exceeded — %s", exc)
        except Exception as exc:
            log.warning("extract_adu failed: %s", exc)
        return {"claim": "", "ground": "", "warrant": "", "backing": "", "rebuttal": ""}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _hit_to_dict(hit, retriever: str) -> dict:
        return {
            "chunk_id":   getattr(hit, "chunk_id", ""),
            "doc_id":     getattr(hit, "doc_id", ""),
            "article_ref":getattr(hit, "article_ref", ""),
            "score":      float(getattr(hit, "score", 0.0)),
            "text":       getattr(hit, "text", ""),
            "retriever":  retriever,
        }
