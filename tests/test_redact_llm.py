import pytest

from callsense.extract import ExtractionError
from callsense.redact_llm import PiiFinding, PiiFindings, redact_with_claude

from .conftest import make_response


def findings(*items):
    return PiiFindings(findings=[PiiFinding(text=t, kind=k) for t, k in items])


def test_claude_findings_are_redacted(fake_client):
    text = "Customer: My birthday is March 3rd, 1990."
    fake_client(make_response(parsed=findings(("March 3rd, 1990", "DATE_OF_BIRTH"))))

    result, usage, unlocated = redact_with_claude(text)

    assert result.text == "Customer: My birthday is [DATE_OF_BIRTH]."
    assert unlocated == []
    assert usage.api_calls == 1


def test_regex_runs_first_and_its_data_is_not_sent(fake_client):
    text = "Customer: Card 4111 1111 1111 1111, born 03/03/1990."
    client = fake_client(make_response(parsed=findings(("03/03/1990", "DATE_OF_BIRTH"))))

    result, _, _ = redact_with_claude(text)

    sent = client.messages.calls[0]["messages"][0]["content"]
    assert "4111" not in sent and "[CARD_NUMBER]" in sent
    assert result.text == "Customer: Card [CARD_NUMBER], born [DATE_OF_BIRTH]."
    # Offsets point at the original text even though Claude saw the masked version.
    assert [r.original for r in result.redactions] == ["4111 1111 1111 1111", "03/03/1990"]


def test_findings_not_in_the_text_are_ignored(fake_client):
    text = "Customer: My package is late."
    fake_client(make_response(parsed=findings(("123 Fake Street", "ADDRESS"))))

    result, _, unlocated = redact_with_claude(text)

    assert result.text == text
    assert unlocated == ["123 Fake Street"]


def test_findings_overlapping_a_placeholder_are_ignored(fake_client):
    text = "Customer: Call 555-0199 please."
    fake_client(make_response(parsed=findings(("[PHONE]", "PHONE"))))

    result, _, unlocated = redact_with_claude(text)

    assert result.text == "Customer: Call [PHONE] please."
    assert unlocated == ["[PHONE]"]


def test_repeated_value_redacts_each_occurrence_once_per_finding(fake_client):
    text = "Customer: oh one four two. Again, oh one four two."
    fake_client(make_response(parsed=findings(("oh one four two", "PHONE"), ("oh one four two", "PHONE"))))

    result, _, _ = redact_with_claude(text)

    assert result.text == "Customer: [PHONE]. Again, [PHONE]."


def test_no_findings_returns_regex_result(fake_client):
    text = "Customer: Call 555-0199."
    fake_client(make_response(parsed=findings()))

    result, _, _ = redact_with_claude(text)

    assert result.text == "Customer: Call [PHONE]."


def test_refusal_raises(fake_client):
    fake_client(make_response(parsed=None, stop_reason="refusal"))
    with pytest.raises(ExtractionError, match="declined"):
        redact_with_claude("Customer: hi")
