"""Two-pass redaction: regex first, then Claude for what regex can't catch.

Pass 1 (free, local): regex masks written-format data such as card numbers and phone numbers.
Pass 2 (Claude): the already-masked transcript is sent to Claude, which lists any remaining
personal data that needs context, such as dates of birth and numbers spoken as words.

Claude only *points at* text to remove; it never rewrites the transcript. Each finding must
match the transcript exactly, so the model can't invent or alter content, and anything it
returns that isn't in the text is ignored. Because pass 1 runs first, the data regex can
catch is never sent to the API.
"""

import time
from enum import Enum

from pydantic import BaseModel, Field

from callsense.extract import DEFAULT_MODEL, MAX_TOKENS, ExtractionError, Usage, get_client
from callsense.redact import Redaction, RedactionResult, apply_redactions, redact


class PiiKind(str, Enum):
    CARD_NUMBER = "CARD_NUMBER"
    CVV = "CVV"
    CARD_EXPIRY = "CARD_EXPIRY"
    SSN = "SSN"
    PHONE = "PHONE"
    EMAIL = "EMAIL"
    ADDRESS = "ADDRESS"
    DATE_OF_BIRTH = "DATE_OF_BIRTH"
    BANK_ACCOUNT = "BANK_ACCOUNT"
    GOVERNMENT_ID = "GOVERNMENT_ID"


class PiiFinding(BaseModel):
    text: str = Field(description="The personal data, copied exactly, character for character, from the transcript.")
    kind: PiiKind


class PiiFindings(BaseModel):
    findings: list[PiiFinding] = Field(description="Every piece of personal data still in the transcript. Empty if none.")


SYSTEM_PROMPT = """You find personal data in customer support call transcripts so it can be redacted.

Some data is already masked with placeholders like [PHONE] or [CARD_NUMBER]; ignore those.
List any personal data that is still present, including:
- numbers spoken as words (card, phone, SSN, bank account numbers)
- dates of birth, in any format, when the context shows it is a birth date
- street addresses, including house numbers written as words
- email addresses spelled out ("name dot surname at example dot com")
- bank account numbers and government ID numbers

Do not list: masked or partial values (such as the last four digits of a card), prices, ordinary dates
that are not birth dates, order, ticket or tracking numbers, product versions, or a city or zip code on its own.

Copy each finding exactly as it appears in the transcript, including spaces and punctuation, and cover the
whole value (for example the full spoken card number, not one group of digits)."""


def _map_to_original(regex_result: RedactionResult, original: str) -> list[tuple[int, int, int]]:
    """Segments of unchanged text: (start in redacted text, end in redacted text, start in original)."""
    segments, red_pos, orig_pos = [], 0, 0
    for r in regex_result.redactions:
        length = r.start - orig_pos
        segments.append((red_pos, red_pos + length, orig_pos))
        red_pos += length + len(f"[{r.kind}]")
        orig_pos = r.end
    segments.append((red_pos, red_pos + len(original) - orig_pos, orig_pos))
    return segments


def _locate(finding_text: str, redacted: str, segments, taken: list[Redaction]) -> tuple[int, int] | None:
    """Find the finding in the redacted text and return its span in the original text, if it maps cleanly."""
    search_from = 0
    while (pos := redacted.find(finding_text, search_from)) != -1:
        end = pos + len(finding_text)
        for seg_start, seg_end, orig_start in segments:
            if seg_start <= pos and end <= seg_end:  # entirely inside unchanged text
                start = orig_start + (pos - seg_start)
                span = (start, start + len(finding_text))
                if not any(span[0] < r.end and r.start < span[1] for r in taken):
                    return span
        search_from = pos + 1
    return None


def redact_with_claude(text: str, model: str = DEFAULT_MODEL) -> tuple[RedactionResult, Usage, list[str]]:
    """Redact with regex, then Claude. Returns the result, API usage, and any findings that couldn't be located."""
    regex_result = redact(text)
    client = get_client()

    start = time.perf_counter()
    response = client.messages.parse(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Transcript:\n\n{regex_result.text}"}],
        output_format=PiiFindings,
    )
    usage = Usage(
        model=model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        api_calls=1,
        latency_s=time.perf_counter() - start,
    )
    if response.stop_reason == "refusal":
        raise ExtractionError(f"{model} declined the redaction request: {response.stop_details}")
    if response.stop_reason == "max_tokens":
        raise ExtractionError(f"{model} hit max_tokens ({MAX_TOKENS}) before finishing")
    if response.parsed_output is None:
        raise ExtractionError(f"{model} returned no parsed output (stop_reason={response.stop_reason})")

    segments = _map_to_original(regex_result, text)
    redactions = list(regex_result.redactions)
    unlocated = []
    for finding in response.parsed_output.findings:
        span = _locate(finding.text, regex_result.text, segments, redactions)
        if span is None:
            unlocated.append(finding.text)
            continue
        redactions.append(Redaction(finding.kind.value, span[0], span[1], text[span[0]:span[1]]))

    redactions.sort(key=lambda r: r.start)
    return RedactionResult(text=apply_redactions(text, redactions), redactions=redactions), usage, unlocated
