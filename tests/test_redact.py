import pytest

from callsense.evaluate_redaction import load_cases, score
from callsense.redact import _luhn_valid, redact


@pytest.mark.parametrize(
    "text, expected",
    [
        ("card 4111 1111 1111 1111.", "card [CARD_NUMBER]."),
        ("card 5500-0000-0000-0004.", "card [CARD_NUMBER]."),
        ("card 4111111111111111.", "card [CARD_NUMBER]."),
        ("expiry 09/28, CVV 123", "expiry [CARD_EXPIRY], CVV [CVV]"),
        ("security code: 4321", "security code: [CVV]"),
        ("It's 123-45-6789.", "It's [SSN]."),
        ("social 123 45 6789", "social [SSN]"),
        ("at 555-0199.", "at [PHONE]."),
        ("at (415) 555-0123.", "at [PHONE]."),
        ("at +1 415 555 0123.", "at [PHONE]."),
        ("at 415.555.0198.", "at [PHONE]."),
        ("email priya.n@example.com", "email [EMAIL]"),
        ("at 42 Maple Street, Springfield", "at [ADDRESS], Springfield"),
        ("at 7 Elm Ave.", "at [ADDRESS]"),
    ],
)
def test_redacts_written_personal_data(text, expected):
    assert redact(text).text == expected


@pytest.mark.parametrize(
    "text",
    [
        "order 1234 5678 9012 3456",  # fails the card checksum
        "ends in 4821",
        "j.m****@example.com",  # already masked
        "charged $29 on June 3rd, outage on 09/28",  # no 'expiry' context
        "version 4.2.1 on Android 14, ticket 48213",
        "tracking 1Z999AA10123456784, zip 97201",
        "SSN-like but invalid: 000-12-3456",
    ],
)
def test_leaves_safe_text_alone(text):
    assert redact(text).text == text


def test_luhn():
    assert _luhn_valid("4111111111111111")
    assert not _luhn_valid("4111111111111112")


def test_redactions_point_at_original_text():
    text = "Card 4111 1111 1111 1111, phone 555-0199."
    result = redact(text)
    assert [(r.kind, text[r.start:r.end]) for r in result.redactions] == [
        ("CARD_NUMBER", "4111 1111 1111 1111"),
        ("PHONE", "555-0199"),
    ]
    assert all(r.original == text[r.start:r.end] for r in result.redactions)


def test_card_digits_are_not_also_matched_as_a_phone():
    result = redact("4111 1111 1111 1111")
    assert [r.kind for r in result.redactions] == ["CARD_NUMBER"]


def test_no_personal_data_means_no_changes():
    text = "Customer: My package is late.\nAgent: It arrives Thursday."
    result = redact(text)
    assert result.text == text and result.redactions == []


# --- Regression guard on the labeled redaction eval ------------------------------

# Kinds regex is not expected to catch yet (they need context or a model).
KNOWN_GAPS = {"DATE_OF_BIRTH"}


def test_no_written_leaks_beyond_known_gaps():
    leaks = score(load_cases())["leaks"]
    cases = {c["id"]: c for c in load_cases()}
    written_leaks = [
        (case_id, kind, text)
        for case_id, kind, text in leaks
        if cases[case_id]["form"] == "written" and kind not in KNOWN_GAPS
    ]
    assert written_leaks == []


def test_no_safe_strings_removed():
    assert score(load_cases())["wrongly_removed"] == []
