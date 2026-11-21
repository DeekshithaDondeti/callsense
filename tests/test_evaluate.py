import pytest

import callsense.evaluate as evaluate_module
from callsense.evaluate import SCORED_FIELDS, SPLITS, evaluate, load_cases, summarize
from callsense.extract import Usage
from callsense.schema import CallAnalysis, IssueCategory, Sentiment

from .conftest import make_analysis


# --- The eval data itself ---------------------------------------------------

CASES = load_cases()


def test_case_ids_are_unique():
    ids = [c["id"] for c in CASES]
    assert len(ids) == len(set(ids))


def test_every_case_has_a_valid_split():
    assert {c["split"] for c in CASES} <= set(SPLITS)
    for split in SPLITS:
        assert any(c["split"] == split for c in CASES), f"no cases in {split}"


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_labels_are_complete_and_valid(case):
    expected = case["expected"]
    assert set(expected) == set(SCORED_FIELDS)
    assert expected["issue_category"] in {c.value for c in IssueCategory}
    assert expected["customer_sentiment"] in {s.value for s in Sentiment}
    for field in ("resolved", "follow_up_required", "pii_mentioned"):
        assert isinstance(expected[field], bool)
    assert case["transcript"].strip()


def test_load_cases_filters_by_split():
    dev = load_cases("dev")
    assert dev and all(c["split"] == "dev" for c in dev)


# --- Scoring logic -------------------------------------------------------------

def _case(case_id, split, **expected):
    labels = {
        "issue_category": "billing",
        "customer_sentiment": "positive",
        "resolved": True,
        "follow_up_required": False,
        "pii_mentioned": False,
    }
    labels.update(expected)
    return {"id": case_id, "split": split, "transcript": f"transcript {case_id}", "expected": labels}


@pytest.fixture
def scripted_model(monkeypatch):
    """Replace analyze_call with canned answers keyed by transcript."""

    def install(answers: dict[str, CallAnalysis]):
        def fake_analyze(transcript, model, effort):
            usage = Usage(model=model, input_tokens=1000, output_tokens=100, api_calls=1, latency_s=1.5)
            return answers[transcript], usage

        monkeypatch.setattr(evaluate_module, "analyze_call", fake_analyze)
        monkeypatch.setattr(evaluate_module, "get_client", lambda: None)

    return install


def test_evaluate_records_misses(scripted_model):
    cases = [_case("a", "dev"), _case("b", "test", resolved=False)]
    scripted_model({
        "transcript a": make_analysis(),                        # all correct
        "transcript b": make_analysis(customer_sentiment="neutral"),  # wrong sentiment and resolved
    })

    rows = evaluate(cases, model="claude-haiku-5-5", workers=1)

    assert rows[0]["misses"] == {}
    assert rows[1]["misses"] == {
        "customer_sentiment": ("positive", "neutral"),
        "resolved": (False, True),
    }


def test_summarize_counts_accuracy_cost_and_latency(scripted_model):
    cases = [_case("a", "dev"), _case("b", "dev"), _case("c", "test", pii_mentioned=True)]
    scripted_model({
        "transcript a": make_analysis(),
        "transcript b": make_analysis(),
        "transcript c": make_analysis(),  # misses pii_mentioned
    })

    summary = summarize(evaluate(cases, model="claude-haiku-5-5", workers=1))

    assert summary["exact"] == 2
    assert summary["splits"] == {"dev": {"exact": 2, "n": 2}, "test": {"exact": 0, "n": 1}}
    assert summary["fields"]["pii_mentioned"] == pytest.approx(2 / 3)
    assert summary["fields"]["issue_category"] == 1.0
    assert summary["input_tokens"] == 3000
    assert summary["latency_p50_s"] == 1.5
    # Haiku: 3,000 input tokens at $0.10/M plus 300 output tokens at $0.50/M
    assert summary["cost_usd"] == pytest.approx(3000 * 0.10 / 1e6 + 300 * 0.50 / 1e6)
