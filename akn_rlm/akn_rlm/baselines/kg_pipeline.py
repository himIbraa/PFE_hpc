"""KG-only baseline pipeline (B5).

Deterministic SPARQL retrieval over the Algerian legal KG
(``data/kg/algerian_legal_kg.ttl``). For each significant token in the
Arabic query, a single UNION-style SPARQL searches paragraph text,
``dznorm:provisionText`` and ``dznorm:conditionText`` for substring matches
and returns the article URI of the matching node. Article URIs are scored
by the number of distinct query tokens they cover, sorted, and resolved
back to ``(doc_id, article_ref)`` via the article registry's alias map.
The top-``top_k`` (default 5) articles are emitted as a deterministic
Arabic template answer matching B1-B4.

If an optional fallback pipeline is provided (e.g. hybrid baseline) and
the KG returns zero hits, the fallback is consulted and its answer is
returned with telemetry baseline rewritten to ``"kg+fallback"``. With no
fallback the pipeline abstains. No LLM call → HCR/JIR are zero by
construction.

Output shape matches ``akn_rlm.rlm.pipeline.build_pipeline().run(query)``
so :func:`akn_rlm.eval.runner._answer_to_result` can consume it unchanged.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Callable, Optional

from akn_rlm.corpus.article_registry import ArticleRegistry
from akn_rlm.normalizers import canonical_article_ref, normalize_arabic

log = logging.getLogger(__name__)

DEFAULT_TOP_K: int = 5
SUPPORT_SPAN_LEN: int = 280
MAX_TOKENS_PER_QUERY: int = 8
MIN_TOKEN_LEN: int = 3
SPARQL_LIMIT: int = 50

# Article URIs in the Algerian legal KG follow:
#   https://legal.dz/resource/<type>/<date>/<num>#art_<X>[_<sub>...]
# where <num> may already include <_date> for a few documents.
_URI_RE = re.compile(
    r"https://legal\.dz/resource/[^/]+/(?P<date>[0-9-]+)/(?P<num>[A-Za-z0-9_-]+)#"
    r"(?P<frag>art_[A-Za-z0-9_]+)"
)
# Suffixes appended to ``art_<X>`` for sub-nodes (paragraph, right, …).
_ART_SUFFIX_RE = re.compile(
    r"_(para|right|obligation|condition|permission|prohibition|action|role)"
    r"_[A-Za-z0-9_]+$"
)

# Single-token concept search. Three UNIONs: paragraph text,
# provision text, condition text. ``article`` is always the parent
# article URI (not the sub-node).
_SPARQL_CONCEPT_SEARCH = """
PREFIX dzdoc:  <https://legal.dz/ontology/document#>
PREFIX dznorm: <https://legal.dz/ontology/norm#>

SELECT DISTINCT ?article ?text WHERE {{
    {{
        ?node dzdoc:directlyContainedIn ?article ;
              dzdoc:text                ?text .
        FILTER(CONTAINS(?text, "{token}"))
    }} UNION {{
        ?node dznorm:expressedIn    ?article ;
              dznorm:provisionText  ?text .
        FILTER(CONTAINS(?text, "{token}"))
    }} UNION {{
        ?node dznorm:expressedIn    ?article ;
              dznorm:conditionText  ?text .
        FILTER(CONTAINS(?text, "{token}"))
    }}
}}
LIMIT {limit}
"""

# Conservative Arabic + Latin stopword list. Anything that hits this set
# after Arabic normalisation is dropped from the SPARQL token list.
_STOPWORDS: set[str] = {
    # Arabic question words / connectives
    "ما", "هي", "هو", "هل", "ماذا", "لماذا", "كيف", "أين", "متى", "من",
    "ومن", "إلى", "عن", "على", "في", "مع", "كل", "بعض", "هذا", "هذه",
    "ذلك", "تلك", "الذي", "التي", "الذين", "اللاتي", "كان", "كانت",
    "يكون", "ليس", "ليست", "قد", "لقد", "أن", "إن", "أو", "أم", "ثم",
    "بين", "حتى", "لكن", "غير", "بدون", "ضد", "حسب", "وفق", "وفقا",
    "وفقًا", "بموجب", "هنا", "هناك", "أيضا", "أيضًا", "نفس",
    "ينبغي", "يجب", "يمكن", "يحق", "حق", "حقوق",
    # Latin
    "the", "a", "an", "and", "or", "of", "in", "to", "is", "are", "was",
    "were", "be", "been", "for", "on", "with", "as", "at", "by", "this",
    "that", "these", "those", "it", "its", "from", "what", "which",
    "who", "how", "why", "when", "where",
}

# Splits on anything that isn't an Arabic letter or alphanumeric.
_TOKEN_SPLIT_RE = re.compile(r"[^\w؀-ۿ]+", flags=re.UNICODE)


class KGBaselinePipeline:
    """SPARQL-driven KG retrieval + deterministic Arabic template answer."""

    def __init__(
        self,
        kg: Any,                                       # rdflib.Graph — passed to sparql_fn
        registry: ArticleRegistry,
        top_k: int = DEFAULT_TOP_K,
        fallback: Any = None,                          # optional pipeline with .run(query)
        sparql_fn: Optional[Callable[[Any, str], list[dict[str, Any]]]] = None,
    ) -> None:
        self._kg = kg
        self._registry = registry
        self._top_k = top_k
        self._fallback = fallback
        if sparql_fn is None:
            from akn_rlm.retrievers import graphrag as _gr
            sparql_fn = _gr.sparql_query
        self._sparql_fn = sparql_fn

    # ------------------------------------------------------------------
    def run(self, query: str) -> dict[str, Any]:
        if not query or not query.strip():
            return self._abstain("empty_query")

        tokens = self._tokenize(query)
        if not tokens:
            return self._abstain("no_tokens")

        # token → article URIs that mention it; URI → first matched span
        scores: dict[str, float] = {}
        spans:  dict[str, str]   = {}
        for tok in tokens:
            rows = self._sparql_for_token(tok)
            seen_uri_for_tok: set[str] = set()
            for row in rows:
                uri = (row.get("article") or "").strip()
                if not uri or uri in seen_uri_for_tok:
                    continue
                seen_uri_for_tok.add(uri)
                scores[uri] = scores.get(uri, 0.0) + 1.0
                if uri not in spans:
                    spans[uri] = (row.get("text") or "").strip()

        if not scores:
            return self._fallback_or_abstain(query, reason="no_kg_hits")

        citations = self._scores_to_citations(scores, spans)
        if not citations:
            return self._fallback_or_abstain(query, reason="no_kg_hits")

        return self._answer(citations, baseline="kg")

    # ------------------------------------------------------------------
    # SPARQL plumbing
    # ------------------------------------------------------------------
    def _sparql_for_token(self, token: str) -> list[dict[str, Any]]:
        # Escape double quotes to keep the SPARQL literal well-formed.
        safe = token.replace('"', "'").replace("\\", "")
        q = _SPARQL_CONCEPT_SEARCH.format(token=safe, limit=SPARQL_LIMIT)
        try:
            return self._sparql_fn(self._kg, q) or []
        except Exception as exc:                                    # noqa: BLE001
            log.warning("KG SPARQL failed for token %r: %s", token, exc)
            return []

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------
    @staticmethod
    def _tokenize(query: str) -> list[str]:
        """Pick up to ``MAX_TOKENS_PER_QUERY`` distinctive tokens from query.

        Drops stopwords and tokens shorter than ``MIN_TOKEN_LEN``. Keeps
        the surface form (no Arabic normalisation) because the KG text
        literals are stored in their original surface form. The ``ال``
        definite-article prefix is stripped so ``القانون`` and ``قانون``
        collapse.
        """
        out: list[str] = []
        seen: set[str] = set()
        for raw in _TOKEN_SPLIT_RE.split(query):
            tok = raw.strip()
            if not tok or len(tok) < MIN_TOKEN_LEN:
                continue
            # Strip the Arabic definite article prefix to widen substring match.
            if tok.startswith("ال") and len(tok) > 4:
                tok_stripped = tok[2:]
            else:
                tok_stripped = tok
            # Drop stopwords (raw and normalised forms).
            if (
                tok_stripped in _STOPWORDS
                or normalize_arabic(tok_stripped) in _STOPWORDS
            ):
                continue
            if tok_stripped in seen:
                continue
            seen.add(tok_stripped)
            out.append(tok_stripped)
            if len(out) >= MAX_TOKENS_PER_QUERY:
                break
        return out

    # ------------------------------------------------------------------
    # URI → (doc_id, article_ref) resolution
    # ------------------------------------------------------------------
    def _uri_to_doc_ref(self, uri: str) -> tuple[Optional[str], Optional[str]]:
        m = _URI_RE.match(uri)
        if not m:
            return (None, None)
        date     = m.group("date")
        num      = m.group("num")
        frag     = m.group("frag")               # "art_X" possibly with sub-suffix
        # Strip sub-node suffix to keep just ``art_<X>``.
        art_eid = _ART_SUFFIX_RE.sub("", frag)
        if not art_eid.startswith("art_"):
            return (None, None)
        article_ref = art_eid[len("art_"):]
        # Try registry alias resolution: ``<num>`` itself, then ``<num>_<date>``.
        canonical = (
            self._registry.resolve_alias(num)
            or self._registry.resolve_alias(f"{num}_{date}")
            or num
        )
        return (canonical, article_ref)

    # ------------------------------------------------------------------
    # Scoring & citation building
    # ------------------------------------------------------------------
    def _scores_to_citations(
        self,
        scores: dict[str, float],
        spans:  dict[str, str],
    ) -> list[dict[str, Any]]:
        # Highest score first, URI string for tie-break (deterministic).
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        out:  list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for uri, score in ranked:
            doc_id, art_ref = self._uri_to_doc_ref(uri)
            if not doc_id or not art_ref:
                continue
            canon_ref = canonical_article_ref(art_ref) or art_ref
            key = (doc_id, canon_ref)
            if key in seen:
                continue
            seen.add(key)
            text = spans.get(uri, "") or ""
            out.append({
                "doc_id":          doc_id,
                "article_ref":     canon_ref,
                "doc_title":       self._doc_title(doc_id),
                "supporting_span": text[:SUPPORT_SPAN_LEN],
                "text":            text,
                "confidence":      float(score),
            })
            if len(out) >= self._top_k:
                break
        return out

    def _doc_title(self, doc_id: str) -> str:
        entry = self._registry.get_doc(doc_id)
        return entry.doc_title if entry and entry.doc_title else doc_id

    # ------------------------------------------------------------------
    # Answer assembly
    # ------------------------------------------------------------------
    @staticmethod
    def _template_answer(citations: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for c in citations:
            doc_title = c.get("doc_title") or c.get("doc_id", "")
            ref       = c.get("article_ref", "")
            text      = c.get("supporting_span") or c.get("text", "")
            parts.append(f"وفقًا لـ {doc_title}، المادة {ref}: {text}")
        return "\n\n".join(parts)

    def _answer(
        self, citations: list[dict[str, Any]], *, baseline: str,
    ) -> dict[str, Any]:
        return {
            "answer_text":       self._template_answer(citations),
            "abstention":        False,
            "abstention_reason": None,
            "citations":         citations,
            "reasoning_chain":   [],
            "trajectory":        [],
            "tokens_used":       0,
            "depth_max_reached": 0,
            "_telemetry": {
                "retry_count":  0,
                "gate_results": {},
                "baseline":     baseline,
            },
        }

    @staticmethod
    def _abstain(reason: str, *, baseline: str = "kg") -> dict[str, Any]:
        return {
            "answer_text":       "",
            "abstention":        True,
            "abstention_reason": reason,
            "citations":         [],
            "reasoning_chain":   [],
            "trajectory":        [],
            "tokens_used":       0,
            "depth_max_reached": 0,
            "_telemetry": {
                "retry_count":  0,
                "gate_results": {},
                "baseline":     baseline,
            },
        }

    # ------------------------------------------------------------------
    # Fallback
    # ------------------------------------------------------------------
    def _fallback_or_abstain(self, query: str, *, reason: str) -> dict[str, Any]:
        if self._fallback is None:
            return self._abstain(reason)
        try:
            answer = self._fallback.run(query)
        except Exception as exc:                                    # noqa: BLE001
            log.warning("KG fallback failed: %s", exc)
            return self._abstain(reason)
        # Rewrite telemetry baseline so downstream metrics know this came
        # from the fallback path. Keep all other answer keys intact.
        if isinstance(answer, dict):
            tel = answer.setdefault("_telemetry", {})
            tel["baseline"] = "kg+fallback"
            tel.setdefault("retry_count", 0)
            tel.setdefault("gate_results", {})
        return answer


def build_kg_pipeline(
    kg: Any,
    registry: ArticleRegistry,
    top_k: int = DEFAULT_TOP_K,
    fallback: Any = None,
    sparql_fn: Optional[Callable[[Any, str], list[dict[str, Any]]]] = None,
) -> KGBaselinePipeline:
    """Factory mirroring :func:`akn_rlm.rlm.pipeline.build_pipeline`."""
    return KGBaselinePipeline(
        kg=kg,
        registry=registry,
        top_k=top_k,
        fallback=fallback,
        sparql_fn=sparql_fn,
    )
