from dataclasses import dataclass

import anthropic
from dotenv import load_dotenv
from pydantic import ValidationError

from callsense.schema import CallAnalysis

MODEL = "claude-haiku-5-5"
TOOL_NAME = "record_call_analysis"
MAX_ATTEMPTS = 2

SYSTEM_PROMPT = (
    "You analyze customer support call transcripts. "
    "Fill in every field of the analysis tool based only on what is stated in the transcript. "
    "Do not invent facts that are not in the transcript."
)

@dataclass
class Usage:
    """Token usage summed across every API call (including retries) for one transcript."""

    input_tokens: int = 0
    output_tokens: int = 0
    api_calls: int = 0


_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        load_dotenv()
        _client = anthropic.Anthropic()
    return _client


def analyze_call(transcript: str) -> tuple[CallAnalysis, Usage]:
    """Return the validated analysis and total token usage for one transcript.

    Uses forced tool use so the model returns JSON matching CallAnalysis's schema.
    Retries once with the validation error if the output does not parse.
    """
    client = get_client()
    tool = {
        "name": TOOL_NAME,
        "description": "Record the structured analysis of the support call.",
        "input_schema": CallAnalysis.model_json_schema(),
    }
    messages = [{"role": "user", "content": f"Transcript:\n\n{transcript}"}]
    usage = Usage()

    for attempt in range(MAX_ATTEMPTS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=[tool],
            tool_choice={"type": "tool", "name": TOOL_NAME},
            messages=messages,
        )
        usage.input_tokens += response.usage.input_tokens
        usage.output_tokens += response.usage.output_tokens
        usage.api_calls += 1
        tool_use = next(b for b in response.content if b.type == "tool_use")
        try:
            return CallAnalysis.model_validate(tool_use.input), usage
        except ValidationError as err:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            messages.append({"role": "assistant", "content": response.content})
            messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_use.id,
                    "is_error": True,
                    "content": f"The output failed validation:\n{err}\nCall the tool again with corrected values.",
                }],
            })

    raise RuntimeError("unreachable")
