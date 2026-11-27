"""Write a transcript for every planned call with Claude, then add speech-to-text noise in code.

The model only writes dialogue. The plan (plan.py) already fixed what each call is about, how it ends
and how the customer feels, so the planted trends don't depend on the model.

By default the dialogue is written by Meta's Model API (see meta.py), so Claude later analyses calls
it didn't write. --provider anthropic uses Claude Haiku through the Batch API instead.

Re-running is safe: calls already in data/calls.jsonl are skipped, and an interrupted batch is
resumed rather than resubmitted.

Usage:
    uv run python -m callsense.synth.generate --dry-run      # free: show the cost estimate and one prompt
    uv run python -m callsense.synth.generate --limit 20     # a sample spread over all 12 weeks, on Meta
    uv run python -m callsense.synth.generate                # every remaining call, on Meta
"""

import argparse
import random
import re
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime

import openai

from callsense.batch import run_requests
from callsense.extract import BATCH_DISCOUNT, DEFAULT_MODEL, MODEL_PRICES, Usage
from callsense.meta import DEFAULT_META_MODEL, META_PRICES, get_meta_client, meta_cost
from callsense.synth.noise import add_noise
from callsense.synth.scenarios import COMPANY_BRIEF, MOODS
from callsense.synth.store import BATCH_STATE_PATH, CALLS_PATH, PLAN_PATH, append_jsonl, done_ids, read_jsonl

GENERATION_MAX_TOKENS = 4000
META_MAX_TOKENS = 8000  # Muse models always reason a little, and reasoning counts toward this limit
META_WORKERS = 8  # well under Meta's 3,000 requests a minute

# Per-call token counts for the estimate printed before anything is sent, from the 20-call sample
# (about 590 input tokens and 48 output tokens per requested line, since Haiku wrote more lines than asked).
EST_INPUT_TOKENS = 600
EST_OUTPUT_TOKENS_PER_LINE = 48

SYSTEM_PROMPT = f"""You write realistic, synthetic customer support phone call transcripts for testing software.

{COMPANY_BRIEF}

Rules:
- Output only the transcript: one turn per line, each line starting with "Agent:" or "Customer:". No title, notes, \
stage directions or blank lines.
- Follow the call spec exactly: the situation, how the call ends, and how the customer feels in their last turns.
- Write exactly the number of lines the spec asks for, no more. A short call is short: skip pleasantries to fit.
- The last lines must show how the call ends: the outcome is settled and the customer reacts to it.
- Use first names only. Do not include any personal data (phone numbers, email addresses, street addresses, card \
numbers, birth dates, account numbers) unless the spec lists it. When it does, write each value exactly as given, \
character for character.
- Make it sound like real speech: people hesitate, repeat themselves and correct themselves. In longer calls the \
agent may also check the account or put the customer on a short hold.
- Only mention refunds if the spec's situation or ending is about one.
- Never name the issue category or the customer's mood; show them through what people say."""


def mentions_refund(spec: dict) -> bool:
    """Only calls about a refund get the refund policy line; on other calls it made the model bring refunds up."""
    return "refund" in f"{spec['situation']} {spec['outcome']}".lower()


def call_brief(spec: dict) -> str:
    """The user message describing one call."""
    when = datetime.fromisoformat(spec["started_at"])
    lines = [
        "Call spec",
        f"Date: {when:%A, %B} {when.day}, {when.year}",
        f"Agent: {spec['agent_name']}. Customer: {spec['customer_name']}.",
        f"Product: {spec['product']}",
        f"Situation: {spec['situation']}",
        f"How the call ends: {spec['outcome']}",
        f"Customer's mood at the end: {MOODS[spec['intended']['customer_sentiment']]}",
        f"Length: exactly {spec['lines']} lines.",
    ]
    if mentions_refund(spec):
        lines.append(f"Refund policy on this date: refunds within {spec['refund_window_days']} days of purchase.")
    if spec["small_talk"]:
        lines.append(f"Include a brief off-topic exchange about {spec['small_talk']}.")
    if spec["pii"]:
        lines.append("Personal data the customer says (write exactly as shown):")
        lines += [f"- {p['purpose']}: {p['value']}" for p in spec["pii"]]
    return "\n".join(lines)


def meta_request(spec: dict, model: str = DEFAULT_META_MODEL) -> dict:
    """Chat Completions params for Meta's Model API. Muse models can't turn reasoning off; minimal is the lowest."""
    return {
        "model": model,
        "max_completion_tokens": META_MAX_TOKENS,
        "reasoning_effort": "minimal",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": call_brief(spec)},
        ],
    }


def generation_request(spec: dict, model: str = DEFAULT_MODEL) -> dict:
    return {
        "model": model,
        "max_tokens": GENERATION_MAX_TOKENS,
        "thinking": {"type": "disabled"},  # plain dialogue writing; thinking would only add output tokens
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": call_brief(spec)}],
    }


_LABEL = re.compile(r"^\**\s*(Agent|Customer)\s*:\**\s*")


def clean_transcript(text: str) -> str | None:
    """Normalize the speaker labels and return the transcript, or None if it isn't a valid transcript."""
    lines = []
    for raw in text.strip().splitlines():
        if not raw.strip():
            continue
        match = _LABEL.match(raw.strip())
        if not match:
            return None
        lines.append(f"{match.group(1)}: {raw.strip()[match.end():].strip()}")
    speakers = {line.split(":")[0] for line in lines}
    if len(lines) < 4 or speakers != {"Agent", "Customer"}:
        return None
    return "\n".join(lines)


def estimate_cost(specs: list[dict], provider: str, model: str, use_batch: bool) -> float:
    prices = META_PRICES if provider == "meta" else MODEL_PRICES
    input_price, output_price = prices.get(model, (0.0, 0.0))
    tokens_out = sum(EST_OUTPUT_TOKENS_PER_LINE * s["lines"] for s in specs)
    cost = (len(specs) * EST_INPUT_TOKENS * input_price + tokens_out * output_price) / 1_000_000
    return cost * BATCH_DISCOUNT if provider == "anthropic" and use_batch else cost


def pending_specs(limit: int | None = None) -> list[dict]:
    """Planned calls without a transcript yet, in a fixed shuffled order so a --limit sample covers every week."""
    done = done_ids(CALLS_PATH)
    pending = [s for s in read_jsonl(PLAN_PATH) if s["call_id"] not in done]
    random.Random("generation-order").shuffle(pending)
    return pending[:limit] if limit else pending


@dataclass
class Reply:
    """One model reply, the same shape whichever provider wrote it."""

    text: str | None  # None when the reply was cut off, refused or errored
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    error: str | None = None


def claude_replies(specs: list[dict], model: str, use_batch: bool) -> Iterator[tuple[str, Reply]]:
    requests = {s["call_id"]: generation_request(s, model) for s in specs}
    for outcome in run_requests(requests, stage="generate", state_path=BATCH_STATE_PATH, use_batch=use_batch):
        message = outcome.message
        if message is None:
            yield outcome.custom_id, Reply(None, error=outcome.error)
            continue
        usage = Usage(model, message.usage.input_tokens, message.usage.output_tokens, api_calls=1, batch=use_batch)
        text = next((b.text for b in message.content if b.type == "text"), "")
        ok = message.stop_reason == "end_turn"
        yield outcome.custom_id, Reply(
            text if ok else None, usage.input_tokens, usage.output_tokens, usage.cost_usd,
            None if ok else f"stop_reason={message.stop_reason}",
        )


def meta_replies(specs: list[dict], model: str, workers: int = META_WORKERS) -> Iterator[tuple[str, Reply]]:
    client = get_meta_client()

    def send(spec):
        try:
            response = client.chat.completions.create(**meta_request(spec, model))
        except openai.APIError as e:
            return spec["call_id"], Reply(None, error=f"{type(e).__name__}: {e}")
        choice = response.choices[0]
        tokens_in, tokens_out = response.usage.prompt_tokens, response.usage.completion_tokens
        ok = choice.finish_reason == "stop"
        return spec["call_id"], Reply(
            choice.message.content if ok else None, tokens_in, tokens_out, meta_cost(model, tokens_in, tokens_out),
            None if ok else f"finish_reason={choice.finish_reason}",
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(send, specs)


def record_for(spec: dict, reply: Reply, provider: str, model: str) -> dict | None:
    """The calls.jsonl row for one reply, or None if the reply isn't a usable transcript."""
    transcript = clean_transcript(reply.text) if reply.text else None
    if transcript is None:
        return None
    values = [p["value"] for p in spec["pii"]]
    transcript, noisy = add_noise(transcript, spec["call_id"], protected=values)
    return {
        "call_id": spec["call_id"],
        "started_at": spec["started_at"],
        "week": spec["week"],
        "region": spec["region"],
        "product": spec["product"],
        "transcript": transcript,
        "stt_noise": noisy,
        "pii_present": [v in transcript for v in values],  # did the planted value make it in verbatim
        "provider": provider,
        "model": model,
        "input_tokens": reply.input_tokens,
        "output_tokens": reply.output_tokens,
        "cost_usd": round(reply.cost_usd, 6),
    }


def confirm(question: str) -> bool:
    try:
        return input(f"{question} [y/N] ").strip().lower() == "y"
    except EOFError:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="only generate this many calls")
    parser.add_argument("--provider", choices=("meta", "anthropic"), default="meta",
                        help="who writes the dialogue (default: meta, which uses no Anthropic credits)")
    parser.add_argument("--model", help=f"default: {DEFAULT_META_MODEL} on meta, {DEFAULT_MODEL} on anthropic")
    parser.add_argument("--sync", action="store_true", help="anthropic only: one request at a time, full price")
    parser.add_argument("--dry-run", action="store_true", help="print the estimate and one prompt; send nothing")
    parser.add_argument("--yes", action="store_true", help="don't ask before spending")
    args = parser.parse_args(argv)
    model = args.model or (DEFAULT_META_MODEL if args.provider == "meta" else DEFAULT_MODEL)

    if not PLAN_PATH.exists():
        print("No plan yet. Run: uv run python -m callsense.synth.plan", file=sys.stderr)
        return 1
    specs = pending_specs(args.limit)
    use_batch = args.provider == "anthropic" and not args.sync
    how = "Meta Model API" if args.provider == "meta" else ("Batch API, half price" if use_batch else "one at a time")
    print(f"{len(specs)} calls to generate with {model} ({how}).")
    print(f"Estimated cost: ${estimate_cost(specs, args.provider, model, use_batch):.2f}"
          + (" of Meta credits" if args.provider == "meta" else " of Anthropic credits"))
    if args.dry_run:
        if specs:
            print("\nExample prompt:\n" + call_brief(specs[0]))
        return 0
    if not specs:
        return 0
    if not args.yes and not confirm("Send these requests?"):
        print("Nothing sent.")
        return 0

    by_id = {s["call_id"]: s for s in specs}
    replies = meta_replies(specs, model) if args.provider == "meta" else claude_replies(specs, model, use_batch)
    written, failed, cost = 0, [], 0.0
    for call_id, reply in replies:
        spec = by_id.get(call_id)
        if spec is None:
            continue  # from a resumed batch, already saved
        cost += reply.cost_usd
        record = record_for(spec, reply, args.provider, model)
        if record is None:
            failed.append(f"{call_id}: {reply.error or 'not a valid transcript'}")
            continue
        append_jsonl(CALLS_PATH, record)
        written += 1

    print(f"Wrote {written} transcripts to data/calls.jsonl. Actual cost: ${cost:.4f}")
    if failed:
        print(f"{len(failed)} failed (re-run to retry them):")
        for line in failed[:10]:
            print(f"  {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
