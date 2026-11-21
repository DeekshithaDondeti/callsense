import pytest

from callsense.extract import MODEL_PRICES, ExtractionError, Usage, analyze_call

from .conftest import make_analysis, make_response


def test_returns_parsed_analysis_and_usage(fake_client):
    expected = make_analysis()
    fake_client(make_response(parsed=expected, input_tokens=1200, output_tokens=300))

    analysis, usage = analyze_call("Customer: hi", model="claude-haiku-5-5")

    assert analysis == expected
    assert usage.model == "claude-haiku-5-5"
    assert (usage.input_tokens, usage.output_tokens, usage.api_calls) == (1200, 300, 1)
    assert usage.latency_s >= 0


def test_sends_transcript_model_and_schema(fake_client):
    client = fake_client(make_response(parsed=make_analysis()))

    analyze_call("Customer: my card was declined", model="claude-sonnet-5-5")

    call = client.messages.calls[0]
    assert call["model"] == "claude-sonnet-5-5"
    assert "my card was declined" in call["messages"][0]["content"]
    assert call["output_format"].__name__ == "CallAnalysis"
    assert "output_config" not in call  # no effort sent unless asked for


def test_effort_is_passed_through(fake_client):
    client = fake_client(make_response(parsed=make_analysis()))
    analyze_call("Customer: hi", effort="low")
    assert client.messages.calls[0]["output_config"] == {"effort": "low"}


@pytest.mark.parametrize(
    "stop_reason, message",
    [("refusal", "declined"), ("max_tokens", "max_tokens")],
)
def test_bad_stop_reasons_raise(fake_client, stop_reason, message):
    fake_client(make_response(parsed=None, stop_reason=stop_reason))
    with pytest.raises(ExtractionError, match=message):
        analyze_call("Customer: hi")


def test_missing_parsed_output_raises(fake_client):
    fake_client(make_response(parsed=None, stop_reason="end_turn"))
    with pytest.raises(ExtractionError, match="no parsed output"):
        analyze_call("Customer: hi")


def test_cost_uses_model_prices():
    input_price, output_price = MODEL_PRICES["claude-opus-5-5"]
    usage = Usage(model="claude-opus-5-5", input_tokens=1_000_000, output_tokens=1_000_000)
    assert usage.cost_usd == pytest.approx(input_price + output_price)


def test_unknown_model_costs_zero():
    assert Usage(model="some-other-model", input_tokens=500, output_tokens=500).cost_usd == 0


def test_analysis_request_matches_analyze_call_and_adds_the_schema(fake_client):
    from callsense.extract import analysis_request, json_format
    from callsense.schema import CallAnalysis

    client = fake_client(make_response(parsed=make_analysis()))
    analyze_call("Customer: hi", effort="low")
    sync = client.messages.calls[0]

    batch = analysis_request("Customer: hi", effort="low")

    assert {k: v for k, v in batch.items() if k != "output_config"} == {
        k: v for k, v in sync.items() if k not in ("output_config", "output_format")
    }
    assert batch["output_config"] == {"effort": "low", "format": json_format(CallAnalysis)}


def test_parse_analysis_validates_json():
    from callsense.extract import parse_analysis

    from .conftest import make_message

    assert parse_analysis(make_message(make_analysis().model_dump_json())) == make_analysis()
    with pytest.raises(ExtractionError, match="doesn't match"):
        parse_analysis(make_message('{"summary": "x"}'))
    with pytest.raises(ExtractionError, match="declined"):
        parse_analysis(make_message("", stop_reason="refusal"))


def test_batch_usage_costs_half():
    full = Usage(model="claude-haiku-5-5", input_tokens=1000, output_tokens=1000)
    half = Usage(model="claude-haiku-5-5", input_tokens=1000, output_tokens=1000, batch=True)
    assert half.cost_usd == pytest.approx(full.cost_usd / 2)
