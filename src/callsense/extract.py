import time
from dataclasses import dataclass

import anthropic
from dotenv import load_dotenv

from callsense.schema import CallAnalysis

DEFAULT_MODEL = "claude-haiku-5-5"

# USD per million tokens (input, output): Anthropic first-party API list prices, prompts under 100K tokens.
# Check https://www.anthropic.com/pricing before relying on these.
MODEL_PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}

# Generous ceiling: thinking tokens count toward max_tokens on these models.
MAX_TOKENS = 16000

SYSTEM_PROMPT = (
    "You analyze customer support call transcripts. "
    "Fill in every field based only on what is stated in the transcript, following each field's definition exactly. "
    "Do not invent facts that are not in the transcript."
)


@dataclass
class Usage:
    """Token usage, latency, and cost for one transcript."""

    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    api_calls: int = 0
    latency_s: float = 0.0

    @property
    def cost_usd(self) -> float:
        input_price, output_price = MODEL_PRICES.get(self.model, (0.0, 0.0))
        return (self.input_tokens * input_price + self.output_tokens * output_price) / 1_000_000


class ExtractionError(RuntimeError):
    pass


_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        load_dotenv()
        _client = anthropic.Anthropic()
    return _client


def analyze_call(
    transcript: str,
    model: str = DEFAULT_MODEL,
    effort: str | None = None,
) -> tuple[CallAnalysis, Usage]:
    """Return the validated analysis and usage for one transcript.

    Uses structured outputs, so the model's response is constrained to CallAnalysis's JSON schema
    and parsed into a CallAnalysis instance by the SDK. Works the same way on Haiku, Sonnet, and Opus.
    """
    client = get_client()
    extra = {"output_config": {"effort": effort}} if effort else {}

    start = time.perf_counter()
    response = client.messages.parse(
        model=model,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Transcript:\n\n{transcript}"}],
        output_format=CallAnalysis,
        **extra,
    )
    latency = time.perf_counter() - start

    usage = Usage(
        model=model,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        api_calls=1,
        latency_s=latency,
    )

    if response.stop_reason == "refusal":
        raise ExtractionError(f"{model} declined to analyze the transcript: {response.stop_details}")
    if response.stop_reason == "max_tokens":
        raise ExtractionError(f"{model} hit max_tokens ({MAX_TOKENS}) before finishing")
    if response.parsed_output is None:
        raise ExtractionError(f"{model} returned no parsed output (stop_reason={response.stop_reason})")

    return response.parsed_output, usage
