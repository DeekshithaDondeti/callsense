"""Shared test fixtures. No test in this suite calls the real Anthropic API."""

from types import SimpleNamespace

import pytest

import callsense.extract as extract
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
