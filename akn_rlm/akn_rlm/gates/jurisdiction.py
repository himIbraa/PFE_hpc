"""Jurisdiction infection detector + gate.

Two-stage check:
  1. Fast regex canary scan (always runs).
  2. Optional LLM classifier (runs only when llm_pool provided and signals found).

Gate semantics:
  - signals found + query_type == "unanswerable"  → PASSED  (correct abstention)
  - signals found + query_type != "unanswerable"  → FAILED  (foreign-law contamination)
  - no signals                                     → PASSED
"""
from __future__ import annotations

import logging
import re
from typing import Sequence

from akn_rlm.gates.base import GateResult

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canary table: (regex_pattern, signal_id, jurisdiction_tag)
# ---------------------------------------------------------------------------

_CANARIES: list[tuple[re.Pattern, str, str]] = []


def _add(pattern: str, signal_id: str) -> None:
    _CANARIES.append((re.compile(pattern, re.IGNORECASE | re.UNICODE), signal_id, signal_id.split(":")[0]))


# -- French law --
_add(r"\bISF\b",                             "fr:isf")
_add(r"\bIFI\b",                             "fr:ifi")
_add(r"\bEURL\b",                            "fr:eurl")
_add(r"\bSAS\b(?!\s+\d)",                   "fr:sas")
_add(r"rupture conventionnelle",             "fr:rupture_conventionnelle")
_add(r"garde.{0,3}vue",                      "fr:garde_a_vue")
_add(r"cr.dit d.imp.t",                      "fr:credit_impot")
_add(r"\bimpr.vision\b",                     "fr:imprevision")
_add(r"action de groupe",                    "fr:action_de_groupe")
_add(r"liquidation judiciaire",              "fr:liquidation_judiciaire")
_add(r"redressement judiciaire",             "fr:redressement_judiciaire")
_add(r"conge? parental",                     "fr:conge_parental")
_add(r"travailleur d.tach.",                 "fr:travailleur_detache")
_add(r"\bPACS\b",                            "fr:pacs")
_add(r"prestation compensatoire",            "fr:prestation_compensatoire")
_add(r"ordonnance de protection",            "fr:ordonnance_protection")
_add(r"code civil fran.ais",                 "fr:civil_code_fr")
_add(r"article L\.\d+",                      "fr:french_law_article")

# -- US law --
_add(r"\bat.will\b",                         "us:at_will")
_add(r"\bat will employ",                    "us:at_will")
_add(r"punitive damages",                    "us:punitive_damages")
_add(r"\bfair use\b",                        "us:fair_use")
_add(r"class action",                        "us:class_action")
_add(r"plea bargain",                        "us:plea_bargaining")
_add(r"\bjury trial\b",                      "us:jury")
_add(r"\bjury\b(?! d.lib)(?! n.est)",        "us:jury")
_add(r"stare decisis",                       "us:stare_decisis")
_add(r"pre.?trial discovery",               "us:discovery")
_add(r"Second Amendment",                    "us:second_amendment")
_add(r"\bLWOP\b",                            "us:lwop")
_add(r"life without parole",                 "us:lwop")
_add(r"Miranda",                             "us:miranda")
_add(r"\bAmendment\b",                       "us:amendment")
_add(r"US Constitution",                     "us:us_constitution")
_add(r"First Amendment",                     "us:first_amendment")
_add(r"due process",                         "us:due_process")
_add(r"\b501\(c\)",                          "us:501c")
_add(r"Chapter 11",                          "us:chapter11")
_add(r"Chapter 7",                           "us:chapter7")

# -- Egyptian law --
_add(r"ال?وصية\s+ال?واجبة",                  "eg:wasiya_wajiba")
_add(r"مأذون",                               "eg:madhoun")
_add(r"article 224.*r.duction",             "eg:art224_reduction")
_add(r"cod. p.nal .gyptien",               "eg:egyptian_penal")
_add(r"loi .gyptienne",                     "eg:egyptian_law")

# -- Tunisian law --
_add(r"code du statut personnel",           "tn:csp")
_add(r"\bCSP\b",                             "tn:csp")
_add(r"chapitre.{0,5}18.{0,20}polygam",    "tn:polygamy_ban")
_add(r"adoption pl.ni.re",                  "tn:adoption_pleniere")
_add(r"tribunal.{0,10}commerce.{0,20}tunisien", "tn:commercial_courts")
_add(r"droit tunisien",                     "tn:tunisian_law")
_add(r"loi tunisienne",                     "tn:tunisian_law")

# -- UK law --
_add(r"\bconsideration\b(?!.*prix|.*contrepartie)", "uk:consideration")
_add(r"duty of care",                        "uk:duty_of_care")
_add(r"civil jury",                          "uk:civil_jury")
_add(r"common law precedent",               "uk:common_law")
_add(r"tort law",                            "uk:tort")
_add(r"\bcontract law\b",                   "uk:contract_law")
_add(r"Companies Act",                       "uk:companies_act")

# -- Gulf law --
_add(r"\bkafeel\b",                          "gulf:kafala")
_add(r"\bkafala\b",                          "gulf:kafala")
_add(r"كفالة\b(?!.*مال|.*قرض)",            "gulf:kafala")
_add(r"exit permit",                         "gulf:exit_permit")
_add(r"تصريح الخروج",                        "gulf:exit_permit")

# -- DZ-absence --
_add(r"auto.?saisine.*conseil constitu",    "dz_absent:auto_saisine")
_add(r"taxe munic",                          "dz_absent:municipal_tax")
_add(r"peine de mort.*corruption",          "dz_absent:death_corruption")
_add(r"quota.*lectoral.*genre",            "dz_absent:electoral_quota")
_add(r"cour constitutionnelle.*auto",      "dz_absent:auto_saisine")

# -- International non-DZ --
_add(r"\bICSID\b",                           "intl:icsid")
_add(r"\bICJ\b",                             "intl:icj")
_add(r"\bpeine de mort.*d.lits .conomiques\b", "asia:economic_death_penalty")

# Catch-all foreign law references
_add(r"droit fran.ais",                     "fr:french_law")
_add(r"loi fran.aise",                      "fr:french_law")
_add(r"droit am.ricain",                    "us:us_law")
_add(r"US law",                              "us:us_law")
_add(r"American law",                        "us:us_law")


# ---------------------------------------------------------------------------
# Regex detector (fast, no LLM)
# ---------------------------------------------------------------------------

def detect(text: str) -> list[str]:
    """Return list of signal IDs found in text."""
    found: list[str] = []
    seen: set[str] = set()
    for pattern, signal_id, _ in _CANARIES:
        if signal_id not in seen and pattern.search(text):
            found.append(signal_id)
            seen.add(signal_id)
    return found


def is_infected(text: str) -> bool:
    """True if any foreign-jurisdiction signal detected."""
    return bool(detect(text))


def jurisdiction_summary(signals: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for s in signals:
        prefix = s.split(":")[0]
        groups.setdefault(prefix, []).append(s)
    return groups


# ---------------------------------------------------------------------------
# LLM classifier (second-pass, optional)
# ---------------------------------------------------------------------------

_LLM_PROMPT = """\
You are a jurisdiction auditor for Algerian law.

The following answer may contain references to foreign-law concepts that do not \
exist under Algerian law.  Suspected signals: {signals}.

Answer text:
{text}

Question: Does this answer actually rely on foreign-law concepts that do NOT \
apply in Algeria?  Reply with exactly one word: YES or NO."""


def _llm_classify(
    answer_text: str,
    signals: list[str],
    llm_pool,
    model: str,
) -> bool:
    """Return True if the LLM confirms foreign-law contamination."""
    prompt = _LLM_PROMPT.format(
        signals=", ".join(signals),
        text=answer_text[:2000],   # truncate to keep prompt short
    )
    try:
        raw = llm_pool.call(prompt, model=model, max_tokens=16, temperature=0.0)
        return "yes" in raw.lower()
    except Exception as exc:
        log.warning("Jurisdiction LLM classifier failed (%s); falling back to regex-only", exc)
        return True   # conservative: trust regex signals when LLM fails


# ---------------------------------------------------------------------------
# Gate function
# ---------------------------------------------------------------------------

def run_gate(
    answer_text: str,
    query_type: str,
    *,
    llm_pool=None,
    model: str = "Qwen3-30B-A3B-Thinking",
) -> GateResult:
    """Run the jurisdiction gate on an answer string.

    Args:
        answer_text : the model's answer text to audit.
        query_type  : the query type (unanswerable → correct abstention is OK).
        llm_pool    : optional LLMPool; if provided, adds LLM second-pass.
        model       : model to use for LLM classifier.
    """
    signals = detect(answer_text)

    if not signals:
        return GateResult(passed=True, score=1.0)

    # Optionally verify signals with LLM second-pass
    if llm_pool is not None:
        confirmed = _llm_classify(answer_text, signals, llm_pool, model)
        if not confirmed:
            return GateResult(passed=True, score=1.0, details=[{
                "signals": signals,
                "note": "signals_detected_but_llm_cleared",
            }])

    if query_type == "unanswerable":
        # Signals in a correct abstention answer are expected and OK
        return GateResult(passed=True, score=1.0, details=[{
            "signals": signals,
            "note": "correct_abstention_foreign_jurisdiction",
        }])

    # Signals in an answerable question → contamination
    log.warning("Jurisdiction gate failed: signals=%s query_type=%s", signals, query_type)
    return GateResult(passed=False, score=0.0, details=[{
        "signals": signals,
        "reason": "foreign_jurisdiction_contamination",
    }])
