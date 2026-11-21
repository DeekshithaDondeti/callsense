import json

from callsense.batch import run_requests

from .conftest import make_message


def requests(n=2):
    return {f"r{i}": {"model": "m", "messages": [{"role": "user", "content": str(i)}]} for i in range(n)}


def test_one_by_one_returns_every_reply(fake_api, tmp_path):
    client = fake_api(lambda params: make_message("ok " + params["messages"][0]["content"]))

    outcomes = list(run_requests(requests(3), stage="s", state_path=tmp_path / "b.json", use_batch=False))

    assert sorted(o.custom_id for o in outcomes) == ["r0", "r1", "r2"]
    assert all(o.message.content[0].text.startswith("ok") for o in outcomes)
    assert len(client.messages.calls) == 3
    assert not (tmp_path / "b.json").exists()


def test_batch_submits_once_polls_and_clears_state(fake_api, tmp_path):
    client = fake_api(lambda params: make_message("ok"))
    state = tmp_path / "b.json"

    outcomes = list(run_requests(requests(2), stage="s", state_path=state, poll_s=0, log=lambda _: None))

    assert [o.custom_id for o in outcomes] == ["r0", "r1"]
    assert len(client.messages.batches.created) == 1
    assert client.messages.batches.retrieves == 2  # polled until it ended
    assert json.loads(state.read_text()) == {}  # forgotten once all results were handed over


def test_interrupted_batch_is_resumed_not_resubmitted(fake_api, tmp_path):
    client = fake_api(lambda params: make_message("ok"))
    state = tmp_path / "b.json"

    stream = run_requests(requests(2), stage="s", state_path=state, poll_s=0, log=lambda _: None)
    next(stream)  # the run crashes after saving one result
    assert json.loads(state.read_text()) == {"s": "batch_1"}

    outcomes = list(run_requests(requests(2), stage="s", state_path=state, poll_s=0, log=lambda _: None))

    assert len(client.messages.batches.created) == 1  # nothing paid for twice
    assert [o.custom_id for o in outcomes] == ["r0", "r1"]
    assert json.loads(state.read_text()) == {}


def test_errored_results_are_reported(fake_api, tmp_path):
    fake_api(lambda params: "overloaded_error")

    outcomes = list(run_requests(requests(1), stage="s", state_path=tmp_path / "b.json", poll_s=0, log=lambda _: None))

    assert outcomes[0].message is None and "overloaded_error" in outcomes[0].error


def test_nothing_to_send_sends_nothing(fake_api, tmp_path):
    client = fake_api(lambda params: make_message("ok"))
    assert list(run_requests({}, stage="s", state_path=tmp_path / "b.json")) == []
    assert client.messages.batches.created == []
