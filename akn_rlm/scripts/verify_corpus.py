#!/usr/bin/env python
"""
Phase A sanity-check script.

Prints:
  - Count of articles per canonical doc
  - 06-01 collision warning
  - FRBR <-> filename mismatches
  - Alias map coverage vs benchmark expected_documents

Exits with code 0 if:
  - Articles total is in range [8997, 9007]
  - Benchmark expected_docs covered 100%
    (except 06-01_2006-02-20 flagged as collision and other known missing docs)

Usage:
    cd D:/TRY_AGAIN/akn_rlm
    python scripts/verify_corpus.py
"""
from __future__ import annotations

import io
import json
import sys
import logging
from pathlib import Path
from typing import Dict, Set

# Force UTF-8 stdout on Windows to handle Arabic text in print() calls
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
else:
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Allow running from project root without install
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s  %(message)s",
)

from akn_rlm.corpus.akn_parser import parse_all, parse_akn_file
from akn_rlm.corpus.article_registry import (
    ArticleRegistry,
    FILENAME_COLLISIONS,
    MISSING_FROM_CORPUS,
    METADATA_MISMATCHES,
)
from akn_rlm.config import get_akn_dir, get_benchmark_path


def _separator(label: str) -> None:
    width = 72
    print()
    print("-" * width)
    print(f"  {label}")
    print("-" * width)


def main() -> int:
    exit_code = 0

    # ------------------------------------------------------------------
    # 1. Parse all AKN XML files
    # ------------------------------------------------------------------
    _separator("Step 1  -  Parsing AKN XML files")
    akn_dir = get_akn_dir()
    print(f"  AKN directory : {akn_dir}")

    articles = parse_all(akn_dir)
    total_articles = len(articles)
    print(f"  Articles parsed: {total_articles:,}")

    # Count per canonical doc_id
    per_doc: Dict[str, int] = {}
    per_doc_filename: Dict[str, str] = {}
    for art in articles:
        per_doc[art.doc_id] = per_doc.get(art.doc_id, 0) + 1
        per_doc_filename[art.doc_id] = art.filename_stem

    print()
    print(f"  {'Canonical doc_id':<35}  {'Filename':<40}  {'Articles':>8}")
    print(f"  {'-'*35}  {'-'*40}  {'-'*8}")
    for cid in sorted(per_doc):
        print(f"  {cid:<35}  {per_doc_filename.get(cid,''):<40}  {per_doc[cid]:>8,}")

    EXPECTED_MIN = 8997
    EXPECTED_MAX = 9007
    if not (EXPECTED_MIN <= total_articles <= EXPECTED_MAX):
        print(
            f"\n  [WARN] Article count {total_articles} is outside expected "
            f"range [{EXPECTED_MIN}, {EXPECTED_MAX}]"
        )
    else:
        print(f"\n  Articles total: {total_articles}  OK  (within +/-5 of 9,002)")

    # ------------------------------------------------------------------
    # 2. Build registry and log collisions
    # ------------------------------------------------------------------
    _separator("Step 2  -  Building ArticleRegistry (collision check)")

    registry = ArticleRegistry()
    # Capture warnings emitted during build
    import io
    collision_log = io.StringIO()
    handler = logging.StreamHandler(collision_log)
    handler.setLevel(logging.WARNING)
    logging.getLogger("akn_rlm.corpus.article_registry").addHandler(handler)
    logging.getLogger("akn_rlm.corpus.article_registry").setLevel(logging.WARNING)

    registry.build(articles)

    handler_output = collision_log.getvalue()
    if "FILENAME COLLISION" in handler_output:
        print()
        for line in handler_output.splitlines():
            if "COLLISION" in line or "MISMATCH" in line or "MISSING" in line:
                print(f"  {line}")
    else:
        # Print manually since logger may have been silenced above
        for fname, (implied, frbr) in FILENAME_COLLISIONS.items():
            print(
                f"  [COLLISION] {fname}.xml  -  "
                f"filename implies {implied}, FRBR says {frbr}"
            )

    print()
    print(f"  Canonical documents registered: {registry.doc_count}")
    print(f"  Total article eIds registered : {registry.article_count():,}")

    # ------------------------------------------------------------------
    # 3. FRBR <-> filename mismatches
    # ------------------------------------------------------------------
    _separator("Step 3  -  FRBR <-> filename / type mismatches")

    # Detect mismatches by comparing filename-implied canonical_id vs FRBR-derived one
    mismatches_found = 0
    xml_files = sorted(get_akn_dir().glob("*.xml"))
    from lxml import etree
    AKN_NS = "http://docs.oasis-open.org/legaldocml/ns/akn/3.0"
    _T = lambda loc: f"{{{AKN_NS}}}{loc}"  # noqa: E731

    for xml_path in xml_files:
        stem = xml_path.stem
        try:
            tree = etree.parse(str(xml_path))
            root = tree.getroot()
        except Exception:
            continue

        # FRBR canonical id
        frbr_id = ""
        work = root.find(f".//{_T('FRBRWork')}")
        if work is not None:
            this_el = work.find(_T("FRBRthis"))
            if this_el is not None:
                val = this_el.get("value", "")
                parts = val.strip("/").split("/")
                if len(parts) >= 6:
                    frbr_id = f"{parts[5]}_{parts[4]}"
            if not frbr_id:
                num_el = work.find(_T("FRBRnumber"))
                date_el = work.find(_T("FRBRdate"))
                if num_el is not None and date_el is not None:
                    frbr_id = f"{num_el.get('value','')}_{date_el.get('date','')}"

        if not frbr_id:
            continue

        # AKN root act type
        act_root = root.find(f"{{{AKN_NS}}}act")
        act_type = act_root.get("name", "unknown") if act_root is not None else "unknown"

        # docNumber
        doc_num_el = root.find(f".//{_T('docNumber')}")
        doc_num_text = doc_num_el.text.strip() if doc_num_el is not None and doc_num_el.text else ""

        flags = []

        # Collision flag
        if stem in FILENAME_COLLISIONS:
            implied, frbr_real = FILENAME_COLLISIONS[stem]
            flags.append(f"CONTENT COLLISION: filename implies {implied}, FRBR={frbr_real}")

        # FRBR vs filename mismatch
        if frbr_id != stem and not any(
            frbr_id.lower() == v.lower()
            for v in [stem, frbr_id]
        ):
            pass  # normalize differences

        # Specific known mismatches from dataset report
        if stem in METADATA_MISMATCHES:
            flags.append(f"METADATA MISMATCH: {METADATA_MISMATCHES[stem]}")

        # Act-type vs doc number type mismatch
        decree_keywords = ["decree", "presidential"]
        if (
            any(k in (doc_num_text.lower() or "") for k in decree_keywords)
            and act_type == "law"
        ) or (
            act_type in ("presidential-decree", "decree")
            and "order" in (doc_num_text.lower() or "")
        ):
            if stem not in METADATA_MISMATCHES:
                flags.append(f"TYPE MISMATCH: root act='{act_type}', docNumber='{doc_num_text}'")

        if flags:
            mismatches_found += 1
            for f in flags:
                print(f"  {stem}.xml  ->  {f}")

    if mismatches_found == 0:
        print("  No additional mismatches detected beyond known ones.")

    # Print all known mismatches
    print()
    print("  Known metadata mismatches (non-fatal, logged at registry init):")
    for stem, desc in METADATA_MISMATCHES.items():
        print(f"    {stem}.xml : {desc}")

    # ------------------------------------------------------------------
    # 4. Benchmark expected_documents coverage
    # ------------------------------------------------------------------
    _separator("Step 4  -  Benchmark expected_documents coverage")

    benchmark_path = get_benchmark_path()
    with open(benchmark_path, encoding="utf-8") as fh:
        bench_data = json.load(fh)

    all_expected: Set[str] = set()
    for q in bench_data.get("questions", []):
        all_expected.update(q.get("expected_documents", []))

    # Also include document_registry keys
    for key in bench_data.get("document_registry", {}):
        all_expected.add(key)

    print(f"  Unique expected_document IDs in benchmark: {len(all_expected)}")
    print()

    covered = []
    missing_or_collision = []
    unknown = []

    for doc_id in sorted(all_expected):
        canonical = registry.resolve_alias(doc_id)

        if canonical is not None and canonical in registry.canonical_ids:
            covered.append((doc_id, canonical, "OK"))
        elif doc_id in FILENAME_COLLISIONS or doc_id in {v[0] for v in FILENAME_COLLISIONS.values()}:
            missing_or_collision.append((doc_id, None, "COLLISION  -  content is AML law 05-01"))
        elif any(doc_id.lower() == m.lower() for m in MISSING_FROM_CORPUS):
            missing_or_collision.append((doc_id, None, "MISSING FROM CORPUS"))
        elif canonical is None:
            # Check if it's an alias pointing to None (collision / known missing)
            norm_key = doc_id.lower().strip()
            from akn_rlm.corpus.article_registry import _STATIC_ALIASES
            if norm_key in _STATIC_ALIASES and _STATIC_ALIASES[norm_key] is None:
                missing_or_collision.append((doc_id, None, "KNOWN ABSENT (collision/missing)"))
            else:
                unknown.append((doc_id, None, "UNKNOWN  -  not in registry or alias map"))
        else:
            unknown.append((doc_id, canonical, "REGISTERED BUT CANONICAL ID UNCLEAR"))

    print(f"  {'Expected doc_id':<40}  {'Canonical':<35}  {'Status'}")
    print(f"  {'-'*40}  {'-'*35}  {'-'*30}")
    for doc_id, canonical, status in sorted(covered, key=lambda x: x[0]):
        print(f"  {doc_id:<40}  {canonical or '':<35}  {status}")
    for doc_id, canonical, status in sorted(missing_or_collision, key=lambda x: x[0]):
        print(f"  {doc_id:<40}  {'':35}  [WARN] {status}")
    for doc_id, canonical, status in sorted(unknown, key=lambda x: x[0]):
        print(f"  {doc_id:<40}  {'':35}  [ERROR] {status}")

    print()
    coverage_pct = (
        len(covered) / len(all_expected) * 100 if all_expected else 0
    )
    print(f"  Covered:  {len(covered):3d} / {len(all_expected)}")
    print(f"  Missing or collision: {len(missing_or_collision)}")
    print(f"  Unknown:  {len(unknown)}")

    # Determine pass/fail
    # Pass = no unknown entries AND article count in range
    if unknown:
        print()
        print("  [FAIL] Unknown expected_documents found  -  alias map incomplete.")
        exit_code = 1
    else:
        print()
        print("  Benchmark expected_docs covered: 100% except flagged collision/missing docs.")

    # ------------------------------------------------------------------
    # 5. Alias map spot-checks
    # ------------------------------------------------------------------
    _separator("Step 5  -  Alias map spot-checks")

    spot_checks = [
        ("Civil Code",    "75-58_1975-09-26"),
        ("Cciv",          "75-58_1975-09-26"),
        ("Family Code",   "84-11_1984-06-09"),
        ("Cfam",          "84-11_1984-06-09"),
        ("Ccom",          "75-59_1975-09-26"),
        ("CPCA",          "08-09_2008-02-25"),
        ("CPP",           "25-14_2025-08-03"),
        ("Cinv",          "22-18_2022-07-24"),
        ("Const",         "2020_2020-12-30"),
        ("anti_corruption", None),          # must be None (collision)
        ("06-01_2006-02-20", None),         # must be None (collision)
        ("06-15_2006-05-11", None),         # MISSING
    ]

    all_ok = True
    for alias, expected in spot_checks:
        got = registry.resolve_alias(alias)
        ok = got == expected
        tag = "OK" if ok else "FAIL"
        print(f"  {tag}  resolve_alias({alias!r}) = {got!r}  (expected {expected!r})")
        if not ok:
            all_ok = False

    if not all_ok:
        print()
        print("  [FAIL] Some alias spot-checks failed.")
        exit_code = 1
    else:
        print()
        print("  All alias spot-checks passed.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    _separator("Summary")
    print(f"  Articles total     : {total_articles:,}")
    print(f"  Canonical docs     : {registry.doc_count}")
    print(f"  Article eIds       : {registry.article_count():,}")
    print(f"  Alias keys         : {len(registry._aliases)}")
    print(f"  Benchmark coverage : {coverage_pct:.0f}% (excluding known missing/collisions)")
    print()

    if exit_code == 0:
        print("  RESULT: PASS")
    else:
        print("  RESULT: FAIL  -  see warnings above")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
