"""Shared test fixtures. No test in this suite calls the real Anthropic API."""

from types import SimpleNamespace

import pytest

import callsense.extract as extract
import callsense.meta as meta
from callsense.schema import CallAnalysis


def make_analysis(**overrides) -> CallAnalysis:
    data = {
        "summary": "Customer was double charged; agent refunded one charge.",
        "issue_category": "billing",
        "customer_sentiment": "positive",
        "resolved": True,
        "follow_up_required": False,
        "action_items": [],
        "pii_mentioned": False,
    }
    data.update(overrides)
    return CallAnalysis.model_validate(data)


def make_response(parsed=None, stop_reason="end_turn", input_tokens=1000, output_tokens=200, stop_details=None):
    return SimpleNamespace(
        parsed_output=parsed,
        stop_reason=stop_reason,
        stop_details=stop_details,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


class FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        return self._responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)


@pytest.fixture
def fake_client(monkeypatch):
    """Install a fake Anthropic client. Call it with the responses the API should return, in order."""

    def install(*responses):
        client = FakeClient(responses)
        monkeypatch.setattr(extract, "_client", client)
        return client

    return install


@pytest.fixture(autouse=True)
def no_real_client(monkeypatch):
    """Fail loudly if any test tries to build a real Anthropic client."""

    def forbidden(*args, **kwargs):
        raise AssertionError("Tests must not create a real Anthropic client")

    monkeypatch.setattr(extract.anthropic, "Anthropic", forbidden)
    monkeypatch.setattr(extract, "_client", None)


@pytest.fixture(autouse=True)
def no_real_meta_client(monkeypatch):
    """Fail loudly if any test tries to build a real Meta Model API client (it would spend Meta credits)."""

    def forbidden(*args, **kwargs):
        raise AssertionError("Tests must not create a real Meta client")

    monkeypatch.setattr(meta, "OpenAI", forbidden)
    monkeypatch.setattr(meta, "_client", None)


def make_message(text: str, stop_reason: str = "end_turn", input_tokens: int = 1000, output_tokens: int = 200):
    """A plain Messages API reply (what messages.create and batch results return)."""
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason,
        stop_details=None,
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


class FakeBatches:
    def __init__(self, messages):
        self._messages = messages
        self.created = []  # the request lists of each batch
        self.retrieves = 0

    def create(self, requests):
        self.created.append(list(requests))
        return SimpleNamespace(id=f"batch_{len(self.created)}", processing_status="in_progress")

    def retrieve(self, batch_id):
        self.retrieves += 1
        counts = SimpleNamespace(processing=0, succeeded=0, errored=0)
        # First poll says it's still running, so the polling loop is exercised.
        status = "in_progress" if self.retrieves == 1 else "ended"
        return SimpleNamespace(id=batch_id, processing_status=status, request_counts=counts)

    def results(self, batch_id):
        index = int(batch_id.split("_")[1]) - 1
        for request in self._messages.batch_requests(index):
            reply = self._messages.respond(request["params"])
            if isinstance(reply, str):  # an error type
                result = SimpleNamespace(type="errored", error=reply)
            else:
                result = SimpleNamespace(type="succeeded", message=reply)
            yield SimpleNamespace(custom_id=request["custom_id"], result=result)


class FakeCreateMessages:
    """messages.create and messages.batches, answering every request with responder(params)."""

    def __init__(self, responder):
        self.respond = responder
        self.calls = []
        self.batches = FakeBatches(self)

    def create(self, **params):
        self.calls.append(params)
        return self.respond(params)

    def batch_requests(self, index):
        return self.batches.created[index]


@pytest.fixture
def fake_api(monkeypatch):
    """Install a fake client whose replies come from responder(params). Supports the Batch API."""

    def install(responder):
        client = SimpleNamespace(messages=FakeCreateMessages(responder))
        monkeypatch.setattr(extract, "_client", client)
        return client

    return install


def make_chat_completion(text: str, finish_reason: str = "stop", prompt_tokens: int = 600, completion_tokens: int = 400):
    """A Meta Model API (OpenAI-compatible) chat completion."""
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=finish_reason, message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
    )


@pytest.fixture
def fake_meta(monkeypatch):
    """Install a fake Meta client whose replies come from responder(params)."""
    import callsense.meta as meta

    def install(responder):
        calls = []

        def create(**params):
            calls.append(params)
            return responder(params)

        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)), calls=calls)
        monkeypatch.setattr(meta, "_client", client)
        return client

    return install
