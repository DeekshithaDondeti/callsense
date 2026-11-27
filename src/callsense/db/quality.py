"""Data-quality checks on the loaded database. Free: reads the database and local files.

Usage:
    uv run python -m callsense.db.quality     # exits 1 if any check fails
"""

import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date, timedelta

import psycopg
from pydantic import ValidationError

from callsense.db.connect import connect
from callsense.db.load import source_rows
from callsense.redact import redact
from callsense.schema import CallAnalysis
from callsense.synth import store

# A day is unusual if its volume is outside this range of the median for weekdays (or weekends).
VOLUME_LOW, VOLUME_HIGH = 0.6, 1.4


@dataclass
class Check:
    name: str
    status: str  # "pass", "warn" or "fail"
    detail: str


def _result(name: str, problems: list, ok: str, status: str = "fail") -> Check:
    if not problems:
        return Check(name, "pass", ok)
    shown = ", ".join(map(str, problems[:5]))
    return Check(name, status, f"{len(problems)} problem(s): {shown}" + (" ..." if len(problems) > 5 else ""))


def _stored_text(conn: psycopg.Connection) -> list[tuple[str, str]]:
    """Every free-text value in the database, as (call_id and column, text)."""
    out = []
    query = (
        "SELECT c.call_id, c.transcript, a.summary, a.action_items "
        "FROM calls c LEFT JOIN call_analysis a USING (call_id) ORDER BY c.call_id"
    )
    for cid, transcript, summary, actions in conn.execute(query):
        out.append((f"{cid} transcript", transcript))
        if summary is not None:
            out.append((f"{cid} summary", summary))
        out += [(f"{cid} action_items", item) for item in actions or []]
    return out


def check_pii_patterns(conn: psycopg.Connection) -> Check:
    hits = [f"{where}: {r.kind}" for where, text in _stored_text(conn) for r in redact(text).redactions]
    return _result("no personal-data patterns in stored text", hits, "regex finds nothing in any stored text")


def check_planted_values(conn: psycopg.Connection) -> Check:
    """Synthetic data only: the planted values are known, so also look for them directly (covers spoken forms)."""
    name = "no planted personal data in stored text"
    plan = {r["call_id"]: r for r in store.read_jsonl(store.PLAN_PATH)}
    if not plan:
        return Check(name, "warn", "skipped: no plan file")
    hits = []
    for where, text in _stored_text(conn):
        cid = where.split()[0]
        hits += [f"{where}: {p['kind']}" for p in plan.get(cid, {}).get("pii", []) if p["value"].lower() in text.lower()]
    return _result(name, hits, "no planted value appears in any stored text")


def check_analysis_schema(conn: psycopg.Connection) -> Check:
    fields = list(CallAnalysis.model_fields)
    query = f"SELECT c.call_id, a.call_id IS NOT NULL, {', '.join('a.' + f for f in fields)} FROM calls c LEFT JOIN call_analysis a USING (call_id)"
    problems, total = [], 0
    for cid, has_analysis, *values in conn.execute(query):
        total += 1
        if not has_analysis:
            problems.append(f"{cid}: no analysis")
            continue
        try:
            CallAnalysis.model_validate(dict(zip(fields, values)))
        except ValidationError as e:
            problems.append(f"{cid}: {e.error_count()} schema error(s)")
    return _result("every call has an analysis that passes the schema", problems, f"{total} rows valid")


def check_row_counts(conn: psycopg.Connection) -> Check:
    counts = {
        "generated": len(store.read_jsonl(store.CALLS_PATH)),
        "redacted": len(store.read_jsonl(store.REDACTED_PATH)),
        "analysed": len(store.read_jsonl(store.ANALYZED_PATH)),
        "calls table": conn.execute("SELECT count(*) FROM calls").fetchone()[0],
        "call_analysis table": conn.execute("SELECT count(*) FROM call_analysis").fetchone()[0],
    }
    summary = ", ".join(f"{k} {v}" for k, v in counts.items())
    same = len(set(counts.values())) == 1 and counts["generated"] > 0
    return Check("row counts reconcile between stages", "pass" if same else "fail", summary)


def check_stored_matches_source(conn: psycopg.Connection) -> Check:
    source = {r["call_id"]: r["content_hash"] for r in source_rows()[0]}
    stored = dict(conn.execute("SELECT call_id, content_hash FROM calls"))
    problems = [f"{cid}: missing from database" for cid in sorted(source.keys() - stored.keys())]
    problems += [f"{cid}: not in source files" for cid in sorted(stored.keys() - source.keys())]
    problems += [f"{cid}: differs from source" for cid in sorted(source.keys() & stored.keys()) if source[cid] != stored[cid]]
    return _result("stored rows match the source files", problems, f"{len(stored)} rows match")


def check_embeddings(conn: psycopg.Connection) -> Check:
    missing = [r[0] for r in conn.execute("SELECT call_id FROM calls WHERE embedding IS NULL ORDER BY call_id")]
    return _result("every call has an embedding", missing, "no missing embeddings")


def check_daily_volume(conn: psycopg.Connection) -> list[Check]:
    """Gaps and per-day mismatches fail; unusual volume only warns, because real spikes happen."""
    stored = dict(conn.execute("SELECT started_at::date, count(*) FROM calls GROUP BY 1"))
    source = Counter(date.fromisoformat(r["started_at"][:10]) for r in store.read_jsonl(store.CALLS_PATH))
    if not stored:
        return [Check("daily volume reconciles with the source", "fail", "no calls loaded")]
    first, last = min(stored), max(stored)
    days = [first + timedelta(days=i) for i in range((last - first).days + 1)]
    mismatched = [f"{d}: {stored.get(d, 0)} stored, {source.get(d, 0)} generated"
                  for d in sorted(set(days) | set(source)) if stored.get(d, 0) != source.get(d, 0) or d not in stored]
    unusual = []
    for weekend in (False, True):
        group = [d for d in days if (d.weekday() >= 5) == weekend]
        if not group:
            continue
        median = statistics.median(stored.get(d, 0) for d in group)
        unusual += [(d, stored.get(d, 0), median) for d in group
                    if not VOLUME_LOW * median <= stored.get(d, 0) <= VOLUME_HIGH * median]
    return [
        _result("daily volume reconciles with the source", mismatched, f"{len(days)} days, no gaps"),
        _result("daily volume is in the usual range",
                [f"{d}: {n} calls (median {m:.0f})" for d, n, m in sorted(unusual)],
                "no unusual days", status="warn"),
    ]


def run_checks(conn: psycopg.Connection) -> list[Check]:
    checks = [
        check_pii_patterns(conn),
        check_planted_values(conn),
        check_analysis_schema(conn),
        check_row_counts(conn),
        check_stored_matches_source(conn),
        check_embeddings(conn),
    ]
    return checks + check_daily_volume(conn)


def report(checks: list[Check], log=print) -> bool:
    """Print each check; return True if none failed."""
    for c in checks:
        log(f"  {c.status.upper():4}  {c.name}: {c.detail}")
    return not any(c.status == "fail" for c in checks)


def main() -> int:
    with connect() as conn:
        return 0 if report(run_checks(conn)) else 1


if __name__ == "__main__":
    sys.exit(main())
