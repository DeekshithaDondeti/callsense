import time
from dataclasses import dataclass

import anthropic
from anthropic import transform_schema
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

# The Message Batches API bills every token at half the list price.
BATCH_DISCOUNT = 0.5

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
    batch: bool = False  # sent through the Message Batches API, billed at half price

    @property
    def cost_usd(self) -> float:
        input_price, output_price = MODEL_PRICES.get(self.model, (0.0, 0.0))
        cost = (self.input_tokens * input_price + self.output_tokens * output_price) / 1_000_000
        return cost * BATCH_DISCOUNT if self.batch else cost


class ExtractionError(RuntimeError):
    pass


_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        load_dotenv()
        _client = anthropic.Anthropic()
    return _client


def json_format(model_class) -> dict:
    """The output_config.format value that constrains a reply to model_class's JSON schema.

    This is what messages.parse(output_format=...) sends; batch requests need it spelled out.
    """
    return {"type": "json_schema", "schema": transform_schema(model_class)}


def check_stop_reason(response, model: str) -> None:
    """Raise ExtractionError for a refusal or a truncated reply."""
    if response.stop_reason == "refusal":
        raise ExtractionError(f"{model} declined the request: {response.stop_details}")
    if response.stop_reason == "max_tokens":
        raise ExtractionError(f"{model} hit max_tokens before finishing")


def parse_json_message(message, model_class, model: str):
    """Validate a plain (non-parse) Message whose reply was constrained by json_format(model_class)."""
    check_stop_reason(message, model)
    text = next((b.text for b in message.content if b.type == "text"), None)
    if text is None:
        raise ExtractionError(f"{model} returned no text (stop_reason={message.stop_reason})")
    try:
        return model_class.model_validate_json(text)
    except ValueError as e:
        raise ExtractionError(f"{model} returned output that doesn't match {model_class.__name__}: {e}") from e


def _analysis_kwargs(transcript: str, model: str, effort: str | None) -> dict:
    kwargs = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": f"Transcript:\n\n{transcript}"}],
    }
    if effort:
        kwargs["output_config"] = {"effort": effort}
    return kwargs


def analysis_request(transcript: str, model: str = DEFAULT_MODEL, effort: str | None = None) -> dict:
    """Messages API params for analysing one transcript: the same request analyze_call sends.

    Use with the Message Batches API; read the reply with parse_analysis.
    """
    kwargs = _analysis_kwargs(transcript, model, effort)
    kwargs["output_config"] = {**kwargs.get("output_config", {}), "format": json_format(CallAnalysis)}
    return kwargs


def parse_analysis(message, model: str = DEFAULT_MODEL) -> CallAnalysis:
    """Validate the reply to an analysis_request."""
    return parse_json_message(message, CallAnalysis, model)


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

    start = time.perf_counter()
    response = client.messages.parse(**_analysis_kwargs(transcript, model, effort), output_format=CallAnalysis)
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
