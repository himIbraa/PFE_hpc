"""Ceiling-breaker integrations for the HPC run.

The five HPC-grade upgrades documented in HANDOFF §"What would actually
break the ceiling":

  1. BGE-m3 dense retriever  — handled at index level
     (set ``EMBED_MODEL=BAAI/bge-m3`` in env; ``akn_rlm.indexers.dense`` is
     model-agnostic and rebuilds the FAISS index with 1024-dim vectors).
  2. BGE-reranker-v2-m3       — handled at config level
     (set ``RERANKER_MODEL=BAAI/bge-reranker-v2-m3``; the existing
     ``akn_rlm.reranker.rerank`` already loads via sentence-transformers
     ``CrossEncoder`` so swapping is a one-line env change).
  3. Per-citation NLI as live verifier — :func:`make_nli_verifier_fn`
  4. Doc-router LLM tie-breaker         — :func:`make_llm_doc_router_call`
  5. Concept→article SPARQL via amendment chain
                                         — :func:`make_concept_amendment_search`

All factories degrade gracefully on failure: NLI returns the F5 default
(LLM verifier path) if the model can't load; the LLM tie-breaker returns
[] on any error so the deterministic doc-router still works; the SPARQL
helper returns an empty set if the KG is missing or a query throws.

Activation: ``AKN_CEILING_BREAKERS=1`` env flag. The R7 dispatcher honours
this flag automatically when constructed via :func:`build_ceiling_dispatcher`
(``run_dispatcher.py --ceiling-breakers``).
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any, Callable, Optional

from akn_rlm.gates.faithfulness_nli import (
    CLAIM_THRESHOLD,
    entailment_score,
)
from akn_rlm.normalizers import canonical_article_ref

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Activation flag
# ---------------------------------------------------------------------------

CEILING_BREAKERS_ENV: str = "AKN_CEILING_BREAKERS"


def is_enabled() -> bool:
    return os.getenv(CEILING_BREAKERS_ENV, "0").strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# 3. NLI verifier
# ---------------------------------------------------------------------------

VerifierFn = Callable[[Any, str, dict, str], dict]


def make_nli_verifier_fn(
    threshold: float = CLAIM_THRESHOLD,
    *,
    fall_back_to_llm: bool = True,
) -> VerifierFn:
    """Return a verifier_fn matching :func:`call_verifier`'s signature.

    Per-citation NLI: scores entailment between the article text (premise)
    and the sub-question (hypothesis). Score >= threshold => relevant.

    When ``fall_back_to_llm=True`` the verifier delegates to the standard
    ``call_verifier`` if the NLI model can't load, so an HPC node without
    the NLI model deps still works (just falls back to F5 behaviour).
    """
    from akn_rlm.rlm.sub_worker import call_verifier as _llm_verifier

    def _verify(
        llm_pool: Any,
        sub_question: str,
        article: dict,
        model: str,
    ) -> dict:
        article_text = article.get("text", "") or article.get("supporting_span", "")
        if not article_text or not sub_question:
            return {
                "relevant": False,
                "supporting_span": None,
                "contradicting_span": None,
                "confidence": 0.0,
            }
        score = entailment_score(article_text, sub_question)
        if score == 0.5 and fall_back_to_llm:
            # Sentinel from faithfulness_nli — model unavailable / threw.
            log.debug("NLI returned neutral 0.5 — falling back to LLM verifier")
            try:
                return _llm_verifier(llm_pool, sub_question, article, model)
            except Exception as exc:
                log.warning("LLM-fallback verifier raised: %s", exc)
        # Build verdict in the call_verifier shape. supporting_span is the
        # article text head when relevant — the handlers later truncate to
        # 280 chars for the citation field.
        relevant = bool(score >= threshold)
        return {
            "relevant": relevant,
            "supporting_span": article_text[:280] if relevant else None,
            "contradicting_span": None,
            "confidence": float(score),
        }

    return _verify


# ---------------------------------------------------------------------------
# 4. LLM doc-router tie-breaker
# ---------------------------------------------------------------------------

LLMDocRouterCall = Callable[[str, list[str]], list[str]]


_LLM_ROUTER_PROMPT = (
    "You are a legal-document classifier for the Algerian legal corpus.\n"
    "Given a question (Arabic / French / English) and a list of candidate "
    "document IDs, return the 1-3 IDs the question is most likely about. "
    "Each candidate ID maps to a single law / order / decree / "
    "constitution. Output ONLY the IDs, one per line, no explanation, no "
    "numbering, no surrounding text.\n\n"
    "QUESTION:\n{query}\n\n"
    "CANDIDATES (id per line):\n{candidates}\n\n"
    "ANSWER (1-3 IDs, one per line):"
)


def make_llm_doc_router_call(
    llm_pool: Any,
    *,
    model: str = "gpt-oss-120b",
    max_picks: int = 3,
) -> LLMDocRouterCall:
    """Return an llm_call(query, candidate_ids) for :class:`DocRouter`.

    The LLM is asked to pick 1-3 candidates from the deterministic top-N.
    Picks that aren't in the candidate set are silently dropped (defends
    against hallucinated doc_ids). Any exception returns [] so the
    router still ranks via alias + numeric + BM25 channels.
    """

    def _call(query: str, candidate_ids: list[str]) -> list[str]:
        if not candidate_ids:
            return []
        prompt = _LLM_ROUTER_PROMPT.format(
            query=(query or "")[:1200],
            candidates="\n".join(candidate_ids),
        )
        try:
            raw = llm_pool.call(
                prompt, model=model, max_tokens=200, temperature=0.0,
            )
        except Exception as exc:
            log.debug("LLM doc-router call failed: %s", exc)
            return []

        if not isinstance(raw, str):
            return []

        candidate_set = set(candidate_ids)
        picks: list[str] = []
        for raw_line in raw.splitlines():
            tok = raw_line.strip().lstrip("-•*").strip().strip("`'\"")
            if tok and tok in candidate_set and tok not in picks:
                picks.append(tok)
            if len(picks) >= max_picks:
                break
        return picks

    return _call


# ---------------------------------------------------------------------------
# 5. Concept → article SPARQL via amendment chain
# ---------------------------------------------------------------------------

ConceptAmendmentSearch = Callable[[list[str]], set[tuple[str, str]]]


# Resolves articles whose ANY version (original or amendment) text contains
# the concept phrase. This catches the §R4 ceiling case where the gold
# definition lives inside the amending decree's article (e.g. lab_cd_q01:
# definition of art_114 actually appears inside 96-21#art_17 — the
# amendment).
_AMENDMENT_QUERY = """\
PREFIX dzdoc: <https://legal.dz/ontology/document#>

SELECT DISTINCT ?article ?text WHERE {
  ?article a dzdoc:Article .
  {
    ?article dzdoc:hasVersion ?version .
    ?version dzdoc:versionText ?text .
  } UNION {
    ?article dzdoc:text ?text .
  }
  FILTER(CONTAINS(?text, ?phrase))
}
LIMIT 100
"""

# URI shape produced by the AKN-RLM corpus loader:
#   https://legal.dz/resource/<category>/<date>/<num>#art_<ref>
_URI_RE = re.compile(
    r"resource/([^/]+)/(\d{4}-\d{2}-\d{2})/([^/#]+)(?:#art_(.+))?",
)


def _uri_to_doc_ref(uri: str) -> Optional[tuple[str, str]]:
    """Parse an article URI into (canonical doc_id, canonical article_ref)."""
    m = _URI_RE.search(str(uri))
    if not m:
        return None
    _cat, date, num, ref = m.groups()
    if not ref:
        return None
    doc_id = f"{num}_{date}"
    return doc_id, canonical_article_ref(ref)


def make_concept_amendment_search(kg: Any) -> ConceptAmendmentSearch:
    """Return a helper that maps concept phrases to (doc_id, article_ref) sets.

    The helper queries the rdflib graph (the same one the `temporal_factual`
    and `conceptual_definitional` handlers use). For each phrase, finds
    articles whose original or amendment text contains the literal phrase.
    Empty / no-KG / SPARQL errors silently return empty.
    """
    if kg is None:
        def _noop(_: list[str]) -> set[tuple[str, str]]:
            return set()
        return _noop

    try:
        from rdflib import Literal  # type: ignore
    except Exception as exc:
        log.warning("rdflib unavailable, concept-amendment search disabled: %s", exc)

        def _noop2(_: list[str]) -> set[tuple[str, str]]:
            return set()
        return _noop2

    def _search(phrases: list[str]) -> set[tuple[str, str]]:
        results: set[tuple[str, str]] = set()
        for phrase in phrases or []:
            phrase = (phrase or "").strip()
            if len(phrase) < 4:
                continue
            try:
                rows = kg.query(
                    _AMENDMENT_QUERY,
                    initBindings={"phrase": Literal(phrase)},
                )
            except Exception as exc:
                log.debug("concept-amendment SPARQL failed for %r: %s", phrase, exc)
                continue
            for row in rows:
                try:
                    article_uri = str(row[0])
                except Exception:
                    continue
                parsed = _uri_to_doc_ref(article_uri)
                if parsed is not None:
                    results.add(parsed)
        return results

    return _search
