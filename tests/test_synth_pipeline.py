"""Generate -> redact -> analyse -> check, end to end, against a fake Claude. Free."""

import json

import pytest

from callsense import extract, redact_llm
from callsense.synth import check, generate, plan, process, store

from .conftest import make_analysis, make_message


@pytest.fixture
def small_plan():
    """40 planned calls, including spoken and written personal data and a planted outage call."""
    full = plan.build_plan()
    with_pii = [c for c in full if c["pii"]]
    picked = with_pii[:25] + [c for c in full if c["pii"] and any(i["spoken"] for i in c["pii"])][:5]
    picked += [c for c in full if "payment_outage" in c["events"]][:5] + [c for c in full if not c["pii"]][:5]
    return list({c["call_id"]: c for c in picked}.values())


@pytest.fixture
def data_dir(tmp_path, monkeypatch, small_plan):
    paths = {
        "PLAN_PATH": tmp_path / "plan.jsonl",
        "CALLS_PATH": tmp_path / "calls.jsonl",
        "REDACTED_PATH": tmp_path / "redacted.jsonl",
        "ANALYZED_PATH": tmp_path / "analyzed.jsonl",
        "BATCH_STATE_PATH": tmp_path / "batches.json",
        "CHECK_PATH": tmp_path / "check.json",
    }
    for module in (store, generate, process, check):
        for name, path in paths.items():
            if hasattr(module, name):
                monkeypatch.setattr(module, name, path)
    store.write_jsonl(paths["PLAN_PATH"], small_plan)
    return paths


def planted_values(small_plan):
    return {item["value"]: item["kind"] for c in small_plan for item in c["pii"]}


def make_responder(small_plan, sent_to_analysis=None, bad_ids=()):
    values = planted_values(small_plan)

    def respond(params):
        content = params["messages"][0]["content"]
        if params["system"] == generate.SYSTEM_PROMPT:
            given = [line.split(": ", 1)[1] for line in content.splitlines() if line.startswith("- ")]
            if any(f"Agent: {name}" in content for name in bad_ids):
                return make_message("Here is your transcript!\nAgent: hi")
            lines = ["Agent: Thanks for calling Kestrel, how can I help?", "Customer: I have a problem with my order."]
            lines += [f"Customer: Sure, it's {v}." for v in given]
            lines += ["Agent: Thanks, all sorted.", "Customer: Great, thank you."]
            return make_message("\n".join(lines), input_tokens=600, output_tokens=300)
        if params["system"] == redact_llm.SYSTEM_PROMPT:  # a perfect second pass: list what regex left
            found = [{"text": v, "kind": k} for v, k in values.items() if v in content]
            return make_message(json.dumps({"findings": found}))
        if params["system"] == extract.SYSTEM_PROMPT:
            if sent_to_analysis is not None:
                sent_to_analysis.append(content)
            return make_message(make_analysis().model_dump_json())
        raise AssertionError("unexpected request")

    return respond


def test_generate_writes_transcripts_and_skips_them_on_rerun(fake_api, data_dir, small_plan):
    client = fake_api(make_responder(small_plan))

    assert generate.main(["--yes", "--sync"]) == 0
    rows = store.read_jsonl(data_dir["CALLS_PATH"])
    assert len(rows) == len(small_plan)
    assert all(all(r["pii_present"]) for r in rows)  # every planted value made it in verbatim
    sent = len(client.messages.calls)

    generate.main(["--yes", "--sync"])
    assert len(client.messages.calls) == sent  # nothing re-sent
    assert len(store.read_jsonl(data_dir["CALLS_PATH"])) == len(small_plan)


def test_generation_request_shape(small_plan):
    spec = next(c for c in small_plan if c["pii"])
    request = generate.generation_request(spec)
    assert request["thinking"] == {"type": "disabled"}
    brief = request["messages"][0]["content"]
    assert all(item["value"] in brief for item in spec["pii"])
    assert spec["situation"] in brief and spec["outcome"] in brief


def test_batch_generation_is_billed_at_half_price(fake_api, data_dir, small_plan, monkeypatch):
    monkeypatch.setattr("callsense.batch.time.sleep", lambda s: None)
    fake_api(make_responder(small_plan))

    generate.main(["--yes", "--limit", "3"])

    rows = store.read_jsonl(data_dir["CALLS_PATH"])
    assert len(rows) == 3
    full = extract.Usage("claude-haiku-5-5", 600, 300).cost_usd
    assert rows[0]["cost_usd"] == pytest.approx(full / 2, abs=1e-6)


def test_invalid_transcripts_are_not_saved(fake_api, data_dir, small_plan, capsys):
    bad = small_plan[0]["agent_name"]
    fake_api(make_responder(small_plan, bad_ids=[bad]))

    generate.main(["--yes", "--sync"])

    saved = {r["call_id"] for r in store.read_jsonl(data_dir["CALLS_PATH"])}
    assert small_plan[0]["call_id"] not in saved
    assert "not a valid transcript" in capsys.readouterr().out


def test_dry_run_sends_nothing(fake_api, data_dir, small_plan, capsys):
    client = fake_api(make_responder(small_plan))
    generate.main(["--dry-run"])
    assert client.messages.calls == [] and client.messages.batches.created == []
    assert "Estimated cost" in capsys.readouterr().out


def test_clean_transcript():
    assert generate.clean_transcript("**Agent:** Hi\n\nCustomer: hey\nAgent: ok\nCustomer: bye") == (
        "Agent: Hi\nCustomer: hey\nAgent: ok\nCustomer: bye"
    )
    assert generate.clean_transcript("Sure! Here it is:\nAgent: Hi\nCustomer: a\nAgent: b\nCustomer: c") is None
    assert generate.clean_transcript("Agent: a\nAgent: b\nAgent: c\nAgent: d") is None


def test_process_redacts_before_analysis_and_check_finds_no_leaks(fake_api, data_dir, small_plan, monkeypatch):
    monkeypatch.setattr("callsense.batch.time.sleep", lambda s: None)
    sent_to_analysis = []
    client = fake_api(make_responder(small_plan, sent_to_analysis))
    generate.main(["--yes", "--sync"])

    assert process.main(["--yes"]) == 0

    redacted = store.read_jsonl(data_dir["REDACTED_PATH"])
    analyzed = store.read_jsonl(data_dir["ANALYZED_PATH"])
    assert len(redacted) == len(analyzed) == len(small_plan)
    stored = data_dir["REDACTED_PATH"].read_text() + data_dir["ANALYZED_PATH"].read_text()
    for value in planted_values(small_plan):
        assert value not in stored
        assert not any(value in text for text in sent_to_analysis)
    assert set(redacted[0]["redactions"][0]) == {"kind", "start", "end"}  # no original values kept

    batches = len(client.messages.batches.created)
    process.main(["--yes"])
    assert len(client.messages.batches.created) == batches  # rerun sends nothing

    report = check.build_report()
    assert report["counts"] == {"planned": len(small_plan), "generated": len(small_plan),
                                "redacted": len(small_plan), "analyzed": len(small_plan)}
    assert report["pii"]["planted_values_leaked"] == 0 and report["pii"]["recall"] == 1.0
    assert report["pii"]["regex_hits_left_in_redacted_text"] == {}
    assert set(report["events"]) >= {"payment_outage", "midwest_low_resolution", "cancellations_flat"}
    assert 0 <= report["label_agreement"]["issue_category"]["agreement"] <= 1


def test_partial_redaction_counts_as_a_leak():
    raw = "Customer: four one five, five five five"
    start = raw.index("four")
    whole = [{"kind": "PHONE", "start": start, "end": len(raw)}]
    half = [{"kind": "PHONE", "start": start, "end": start + 8}]
    value = "four one five, five five five"
    assert check.fully_redacted(raw, value, whole)
    assert not check.fully_redacted(raw, value, half)
