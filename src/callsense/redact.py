"""Rule-based redaction of personal data in call transcripts.

Replaces card numbers, CVVs, card expiry dates, SSNs, phone numbers, email addresses and street
addresses with placeholders such as [CARD_NUMBER]. It runs locally: no API calls, no cost.

Regex only catches data written as digits or in standard formats. Spoken forms ("four one one
one...", "forty-two Maple Street") and free-form data such as dates of birth are not covered;
see evals/redaction_cases.json for what is and isn't caught.

Usage:
    echo "Call me at (415) 555-0123" | uv run python -m callsense.redact
"""

import re
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Redaction:
    kind: str
    start: int  # offsets into the original text
    end: int
    original: str


@dataclass(frozen=True)
class RedactionResult:
    text: str
    redactions: list[Redaction]


def _luhn_valid(digits: str) -> bool:
    """Card-number checksum. Filters out order numbers and other long digit strings."""
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _is_card(match: re.Match) -> bool:
    digits = re.sub(r"\D", "", match.group(0))
    return 13 <= len(digits) <= 19 and _luhn_valid(digits)


_STREET_SUFFIX = (
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl|"
    r"Terrace|Circle|Parkway|Pkwy|Highway|Hwy)"
)

# (kind, pattern, group to redact, extra check). Earlier rules win when matches overlap.
_RULES: list[tuple[str, re.Pattern, int, object]] = [
    ("CARD_NUMBER", re.compile(r"\b(?:\d[ -]?){12,18}\d\b"), 0, _is_card),
    ("CVV", re.compile(r"\b(?:CVV|CVC|CV2|security code)\s*(?:is\s*)?:?\s*(\d{3,4})\b", re.IGNORECASE), 1, None),
    (
        "CARD_EXPIRY",
        re.compile(r"\b(?:exp(?:iry|iration)?(?:\s+date)?|expires)\s*(?:is\s*)?:?\s*(\d{1,2}\s*/\s*\d{2,4})\b", re.IGNORECASE),
        1,
        None,
    ),
    ("SSN", re.compile(r"\b(?!000|666|9\d\d)\d{3}([- ])(?!00)\d{2}\1(?!0000)\d{4}\b"), 0, None),
    ("EMAIL", re.compile(r"(?<![\w.*+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), 0, None),
    (
        "PHONE",
        re.compile(r"(?:\+?1[ .-]?)?(?:\(\d{3}\)\s?|\b\d{3}[ .-])?\b\d{3}[ .-]\d{4}\b"),
        0,
        None,
    ),
    ("ADDRESS", re.compile(rf"\b\d{{1,6}}\s+(?:[A-Z][a-z]+\s+){{1,3}}{_STREET_SUFFIX}\b\.?"), 0, None),
]


def redact(text: str) -> RedactionResult:
    """Return the text with personal data replaced by [KIND] placeholders, plus what was replaced."""
    found: list[Redaction] = []
    for kind, pattern, group, check in _RULES:
        for match in pattern.finditer(text):
            if check is not None and not check(match):
                continue
            start, end = match.span(group)
            if any(start < r.end and r.start < end for r in found):
                continue  # already covered by a higher-priority rule
            found.append(Redaction(kind, start, end, text[start:end]))

    found.sort(key=lambda r: r.start)
    pieces, cursor = [], 0
    for r in found:
        pieces.append(text[cursor:r.start])
        pieces.append(f"[{r.kind}]")
        cursor = r.end
    pieces.append(text[cursor:])
    return RedactionResult(text="".join(pieces), redactions=found)


def main() -> None:
    print(redact(sys.stdin.read()).text, end="")


if __name__ == "__main__":
    main()
