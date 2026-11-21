import pytest
from pydantic import ValidationError

from callsense.schema import CallAnalysis

from .conftest import make_analysis


def test_valid_analysis_parses():
    analysis = make_analysis(issue_category="shipping", customer_sentiment="negative")
    assert analysis.issue_category.value == "shipping"
    assert analysis.customer_sentiment.value == "negative"


def test_unknown_category_is_rejected():
    with pytest.raises(ValidationError):
        make_analysis(issue_category="refunds")


def test_unknown_sentiment_is_rejected():
    with pytest.raises(ValidationError):
        make_analysis(customer_sentiment="angry")


def test_action_items_default_to_empty():
    data = make_analysis().model_dump()
    del data["action_items"]
    assert CallAnalysis.model_validate(data).action_items == []


def test_every_field_has_a_description():
    # Field descriptions are the definitions the model reads, so none may be missing.
    properties = CallAnalysis.model_json_schema()["properties"]
    missing = [name for name, prop in properties.items() if not prop.get("description")]
    assert missing == []
