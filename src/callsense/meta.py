"""Meta Model API client, used only to write synthetic transcripts.

Redaction and analysis stay on Claude: those are what CallSense measures. Using a different model
family to write the calls also means Claude never analyses its own writing.

The API is OpenAI-compatible (https://dev.meta.ai/docs/protocols/chat-completions), so it is called
through the openai SDK with Meta's base URL. The key is read from META_API_KEY in .env.
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

BASE_URL = "https://api.meta.ai/v1"
DEFAULT_META_MODEL = "muse-spark-1.3"

# USD per million tokens (input, output), standard tier, prompts under 200K tokens.
# From https://dev.meta.ai/docs/pricing-rate-limits; check it before relying on these.
META_PRICES = {
    "muse-spark-1.3": (1.25, 4.25),
    "muse-ember-1.0": (0.10, 0.50),
    "muse-wonder-1.0": (4.00, 20.00),
}

_client: OpenAI | None = None


def get_meta_client() -> OpenAI:
    global _client
    if _client is None:
        load_dotenv()
        key = os.environ.get("META_API_KEY")
        if not key:
            raise RuntimeError("META_API_KEY is not set. Add it to .env.")
        # The SDK retries 429s and 5xx errors with backoff, honouring Retry-After.
        _client = OpenAI(base_url=BASE_URL, api_key=key, max_retries=4)
    return _client


def meta_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    input_price, output_price = META_PRICES.get(model, (0.0, 0.0))
    return (input_tokens * input_price + output_tokens * output_price) / 1_000_000
