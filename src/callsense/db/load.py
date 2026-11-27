"""Load redacted, analysed calls into Postgres and embed them. Free: no API calls.

Idempotent: rows are upserted by call_id, so re-running never duplicates them, and a call is only
re-embedded when its text or the embedding model changes. Each chunk is one transaction, so an
interrupted run carries on where it stopped.

Calls that haven't been redacted and analysed yet are not loaded; run callsense.synth.process
(paid, asks first) for those. It skips calls it has already done.
"""

import hashlib
from dataclasses import dataclass

import psycopg
from psycopg.types.json import Jsonb

from callsense.db import embed
from callsense.synth import store

CHUNK = 256


@dataclass
class LoadResult:
    source_calls: int  # calls in data/calls.jsonl
    loaded: int  # calls upserted (redacted and analysed)
    new: int
    embedded: int
    waiting: int  # calls not yet redacted and analysed


def source_rows() -> tuple[list[dict], int]:
    """Rows ready to load, and how many generated calls are still waiting for redaction or analysis."""
    calls = store.read_jsonl(store.CALLS_PATH)
    redacted = {r["call_id"]: r for r in store.read_jsonl(store.REDACTED_PATH)}
    analyzed = {r["call_id"]: r for r in store.read_jsonl(store.ANALYZED_PATH)}
    rows = []
    for call in calls:
        cid = call["call_id"]
        if cid not in redacted or cid not in analyzed:
            continue
        a = analyzed[cid]
        # Only the redacted transcript is taken from here on; call["transcript"] is never stored.
        transcript = redacted[cid]["transcript"]
        text = embed.text_to_embed(a["analysis"]["summary"], transcript)
        rows.append({
            "call_id": cid,
            "started_at": call["started_at"],
            "region": call["region"],
            "product": call["product"],
            "transcript": transcript,
            "redaction_count": len(redacted[cid]["redactions"]),
            "content_hash": hashlib.sha256(f"{embed.EMBEDDING_MODEL}\n{text}".encode()).hexdigest(),
            "embed_text": text,
            **a["analysis"],
            "action_items": Jsonb(a["analysis"]["action_items"]),
            "model": a["model"],
            "input_tokens": a["input_tokens"],
            "output_tokens": a["output_tokens"],
            "cost_usd": a["cost_usd"],
            "latency_s": a.get("latency_s"),
        })
    return rows, len(calls) - len(rows)


UPSERT_CALL = """
INSERT INTO calls (call_id, started_at, region, product, transcript, redaction_count, content_hash, embedding)
VALUES (%(call_id)s, %(started_at)s, %(region)s, %(product)s, %(transcript)s, %(redaction_count)s,
        %(content_hash)s, %(embedding)s)
ON CONFLICT (call_id) DO UPDATE SET
    started_at = EXCLUDED.started_at, region = EXCLUDED.region, product = EXCLUDED.product,
    transcript = EXCLUDED.transcript, redaction_count = EXCLUDED.redaction_count,
    content_hash = EXCLUDED.content_hash, embedding = COALESCE(EXCLUDED.embedding, calls.embedding)
"""

ANALYSIS_COLUMNS = (
    "call_id", "summary", "issue_category", "customer_sentiment", "resolved", "follow_up_required",
    "action_items", "pii_mentioned", "model", "input_tokens", "output_tokens", "cost_usd", "latency_s",
)
UPSERT_ANALYSIS = (
    f"INSERT INTO call_analysis ({', '.join(ANALYSIS_COLUMNS)}) "
    f"VALUES ({', '.join(f'%({c})s' for c in ANALYSIS_COLUMNS)}) "
    f"ON CONFLICT (call_id) DO UPDATE SET "
    + ", ".join(f"{c} = EXCLUDED.{c}" for c in ANALYSIS_COLUMNS[1:])
)


def load(conn: psycopg.Connection, log=print) -> LoadResult:
    rows, waiting = source_rows()
    existing = {
        cid: (digest, has_embedding)
        for cid, digest, has_embedding in conn.execute("SELECT call_id, content_hash, embedding IS NOT NULL FROM calls")
    }
    new = embedded = 0
    for start in range(0, len(rows), CHUNK):
        chunk = rows[start : start + CHUNK]
        stale = [r for r in chunk if existing.get(r["call_id"]) != (r["content_hash"], True)]
        vectors = dict(zip((r["call_id"] for r in stale), embed.embed_documents([r["embed_text"] for r in stale]) if stale else []))
        params = [{**r, "embedding": vectors.get(r["call_id"])} for r in chunk]
        with conn.transaction(), conn.cursor() as cur:
            cur.executemany(UPSERT_CALL, params)
            cur.executemany(UPSERT_ANALYSIS, params)
        new += sum(r["call_id"] not in existing for r in chunk)
        embedded += len(stale)
        log(f"  loaded {start + len(chunk)}/{len(rows)} ({embedded} embedded)")
    return LoadResult(len(rows) + waiting, len(rows), new, embedded, waiting)
