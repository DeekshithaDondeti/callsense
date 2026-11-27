"""Pipeline and data-quality checks against a real Postgres, with a fake embedding model.

Uses its own database (callsense_test), never the real one. Skipped when Postgres isn't running,
except in CI, where it must be.
"""

import os

import numpy as np
import psycopg
import pytest

from callsense.db import build, embed, load, quality
from callsense.db.connect import DEFAULT_URL, connect
from callsense.synth import store

from .conftest import make_analysis

ADMIN_URL = os.environ.get("CALLSENSE_TEST_ADMIN_URL", DEFAULT_URL)
TEST_URL = ADMIN_URL.rsplit("/", 1)[0] + "/callsense_test"


@pytest.fixture(scope="module")
def test_database():
    try:
        admin = psycopg.connect(ADMIN_URL, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError:
        if os.environ.get("CI"):
            raise
        pytest.skip("Postgres isn't running (docker compose up -d --wait)")
    with admin:
        admin.execute("DROP DATABASE IF EXISTS callsense_test WITH (FORCE)")
        admin.execute("CREATE DATABASE callsense_test")
        yield TEST_URL
        admin.execute("DROP DATABASE callsense_test WITH (FORCE)")


@pytest.fixture
def conn(test_database):
    with connect(test_database) as c:
        c.execute("TRUNCATE calls, call_analysis")
        yield c


@pytest.fixture
def embedded(monkeypatch):
    """A fake embedder that records what it was asked to embed."""
    texts = []

    def fake(batch):
        texts.extend(batch)
        return [np.full(embed.EMBEDDING_DIM, 0.1, dtype=np.float32) for _ in batch]

    monkeypatch.setattr(embed, "embed_documents", fake)
    return texts


@pytest.fixture
def dataset(tmp_path, monkeypatch):
    """Ten processed calls across five days. call-1 had a phone number that redaction removed."""
    for name in ("PLAN_PATH", "CALLS_PATH", "REDACTED_PATH", "ANALYZED_PATH"):
        monkeypatch.setattr(store, name, tmp_path / f"{name}.jsonl")
    calls, redacted, analyzed, plan = [], [], [], []
    for i in range(1, 11):
        cid = f"call-{i}"
        secret = "(415) 555-0123" if i == 1 else ""
        calls.append({"call_id": cid, "started_at": f"2026-06-0{(i - 1) // 2 + 1}T09:00:00", "region": "West",
                      "product": "Kestrel Cam", "transcript": f"Customer: call {i} {secret}"})
        redacted.append({"call_id": cid, "transcript": f"Customer: call {i} {'[PHONE]' if secret else ''}",
                         "redactions": [{"kind": "PHONE", "start": 0, "end": 1}] if secret else []})
        analyzed.append({"call_id": cid, "analysis": make_analysis(action_items=["Call back"]).model_dump(mode="json"),
                         "model": "claude-haiku-5-5", "input_tokens": 10, "output_tokens": 5, "cost_usd": 0.000001})
        plan.append({"call_id": cid, "pii": [{"kind": "PHONE", "value": secret}] if secret else []})
    for path, rows in ((store.CALLS_PATH, calls), (store.REDACTED_PATH, redacted),
                       (store.ANALYZED_PATH, analyzed), (store.PLAN_PATH, plan)):
        store.write_jsonl(path, rows)
    return {"redacted": redacted, "analyzed": analyzed}


def statuses(conn):
    return {c.name: c.status for c in quality.run_checks(conn)}


def count(conn, table):
    return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def quiet(*_):
    pass


def test_load_stores_redacted_text_only_and_passes_every_check(conn, dataset, embedded):
    result = load.load(conn, log=quiet)

    assert (result.loaded, result.new, result.embedded, result.waiting) == (10, 10, 10, 0)
    assert count(conn, "calls") == count(conn, "call_analysis") == 10
    stored = conn.execute("SELECT transcript FROM calls WHERE call_id = 'call-1'").fetchone()[0]
    assert "[PHONE]" in stored and "555" not in stored
    assert not any("555-0123" in text for text in embedded)
    assert set(statuses(conn).values()) == {"pass"}


def test_rerun_adds_nothing_and_embeds_nothing(conn, dataset, embedded):
    load.load(conn, log=quiet)
    embedded.clear()

    result = load.load(conn, log=quiet)

    assert (result.new, result.embedded) == (0, 0) and embedded == []
    assert count(conn, "calls") == count(conn, "call_analysis") == 10


def test_only_a_changed_call_is_embedded_again(conn, dataset, embedded):
    load.load(conn, log=quiet)
    embedded.clear()
    dataset["analyzed"][3]["analysis"]["summary"] = "A new summary."
    store.write_jsonl(store.ANALYZED_PATH, dataset["analyzed"])
    assert statuses(conn)["stored rows match the source files"] == "fail"

    result = load.load(conn, log=quiet)

    assert result.embedded == 1 and embedded[0].startswith("A new summary.")
    assert statuses(conn)["stored rows match the source files"] == "pass"


def test_interrupted_load_resumes(conn, dataset, embedded, monkeypatch):
    monkeypatch.setattr(load, "CHUNK", 4)
    working = embed.embed_documents

    def crash_on_second_chunk(batch):
        if embedded:
            raise KeyboardInterrupt
        return working(batch)

    monkeypatch.setattr(embed, "embed_documents", crash_on_second_chunk)
    with pytest.raises(KeyboardInterrupt):
        load.load(conn, log=quiet)
    assert count(conn, "calls") == count(conn, "call_analysis") == 4  # whole chunks only

    monkeypatch.setattr(embed, "embed_documents", working)
    result = load.load(conn, log=quiet)

    assert (result.new, result.embedded) == (6, 6) and count(conn, "calls") == 10


def test_unprocessed_calls_wait_and_fail_reconciliation(conn, dataset, embedded):
    store.write_jsonl(store.ANALYZED_PATH, dataset["analyzed"][:8])

    result = load.load(conn, log=quiet)

    assert (result.loaded, result.waiting) == (8, 2)
    assert statuses(conn)["row counts reconcile between stages"] == "fail"
    assert statuses(conn)["daily volume reconciles with the source"] == "fail"


@pytest.mark.parametrize(
    "sql, check",
    [
        ("UPDATE call_analysis SET summary = 'Call 212-555-0188' WHERE call_id = 'call-2'",
         "no personal-data patterns in stored text"),
        ("""UPDATE call_analysis SET action_items = '["Ring (415) 555-0123"]' WHERE call_id = 'call-1'""",
         "no planted personal data in stored text"),
        ("DELETE FROM call_analysis WHERE call_id = 'call-3'", "every call has an analysis that passes the schema"),
        ("UPDATE calls SET embedding = NULL WHERE call_id = 'call-4'", "every call has an embedding"),
        ("DELETE FROM calls WHERE call_id = 'call-5'", "daily volume reconciles with the source"),
    ],
)
def test_each_check_catches_its_problem(conn, dataset, embedded, sql, check):
    load.load(conn, log=quiet)
    conn.execute(sql)
    assert statuses(conn)[check] == "fail"


def test_database_rejects_values_outside_the_schema(conn, dataset, embedded):
    load.load(conn, log=quiet)
    with pytest.raises(psycopg.errors.CheckViolation):
        conn.execute("UPDATE call_analysis SET issue_category = 'refunds'")


def test_build_command_exit_code_follows_the_checks(conn, dataset, embedded, monkeypatch):
    monkeypatch.setenv("CALLSENSE_DB_URL", TEST_URL)
    assert build.main(["--no-docker", "--reset"]) == 0
    store.write_jsonl(store.ANALYZED_PATH, dataset["analyzed"][:8])
    assert build.main(["--no-docker"]) == 1
