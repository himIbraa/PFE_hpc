"""Temporal (version-aware) retrieval wrapper."""
from __future__ import annotations

import logging
import re
from datetime import date, datetime

log = logging.getLogger(__name__)


def normalize_date(date_str: str) -> str | None:
    """Parse common date formats to YYYY-MM-DD string.

    Accepts: "2005-02-06", "2005/02/06", "06/02/2005", "February 2005", etc.
    Returns None if unparseable.
    """
    date_str = date_str.strip()
    # ISO format
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", date_str)
    if m:
        return date_str
    # DD/MM/YYYY
    m = re.match(r"^(\d{1,2})/(\d{1,2})/(\d{4})$", date_str)
    if m:
        d, mo, y = m.groups()
        return f"{y}-{int(mo):02d}-{int(d):02d}"
    # YYYY/MM/DD
    m = re.match(r"^(\d{4})/(\d{2})/(\d{2})$", date_str)
    if m:
        return date_str.replace("/", "-")
    # Year only
    m = re.match(r"^(\d{4})$", date_str)
    if m:
        return f"{m.group(1)}-01-01"
    try:
        for fmt in ("%B %Y", "%b %Y", "%d %B %Y", "%d %b %Y"):
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                pass
    except Exception:
        pass
    return None


def get_article_at_date(
    temporal_index: dict,
    doc_id: str,
    article_ref: str,
    query_date: str,
) -> dict | None:
    """Return the article text version in effect on query_date.

    temporal_index keys: "{doc_id}/{eid}" (built by kg_loader.build_temporal_index).
    query_date: "YYYY-MM-DD".
    Returns dict with keys: text, version_date, doc_id, article_ref.
    Returns None if no entry found.
    """
    from akn_rlm.normalizers import ref_to_eid

    eid = ref_to_eid(article_ref)
    key = f"{doc_id}/{eid}"
    versions = temporal_index.get(key, [])
    if not versions:
        return None

    # versions is a list of {"date": "YYYY-MM-DD", "text": "..."}
    # Return the latest version with date <= query_date
    try:
        qd = date.fromisoformat(normalize_date(query_date) or query_date)
    except ValueError:
        log.warning("Cannot parse query date: %s", query_date)
        return None

    best = None
    for v in versions:
        try:
            vd = date.fromisoformat(v.get("date", "1900-01-01"))
        except ValueError:
            continue
        if vd <= qd:
            if best is None or vd > date.fromisoformat(best["date"]):
                best = v

    if best is None:
        return None
    return {
        "doc_id":       doc_id,
        "article_ref":  article_ref,
        "version_date": best["date"],
        "text":         best.get("text", ""),
    }
