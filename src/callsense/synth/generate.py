"""Write a transcript for every planned call with Claude, then add speech-to-text noise in code.

Claude only writes dialogue. The plan (plan.py) already fixed what each call is about, how it ends
and how the customer feels, so the planted trends don't depend on the model.

Re-running is safe: calls already in data/calls.jsonl are skipped, and an interrupted batch is
resumed rather than resubmitted.

Usage:
    uv run python -m callsense.synth.generate --dry-run      # free: show the cost estimate and one prompt
    uv run python -m callsense.synth.generate --limit 20     # a sample spread over all 12 weeks
    uv run python -m callsense.synth.generate                # every remaining call, via the Batch API
"""

import argparse
import random
import re
import sys
from datetime import datetime

from callsense.batch import run_requests
from callsense.extract import BATCH_DISCOUNT, DEFAULT_MODEL, MODEL_PRICES, Usage
from callsense.synth.noise import add_noise
from callsense.synth.scenarios import COMPANY_BRIEF, MOODS
from callsense.synth.store import BATCH_STATE_PATH, CALLS_PATH, PLAN_PATH, append_jsonl, done_ids, read_jsonl

GENERATION_MAX_TOKENS = 4000

# Rough per-call token counts for the estimate printed before anything is sent.
EST_INPUT_TOKENS = 650
EST_OUTPUT_TOKENS_PER_LINE = 30

SYSTEM_PROMPT = f"""You write realistic, synthetic customer support phone call transcripts for testing software.

{COMPANY_BRIEF}

Rules:
- Output only the transcript: one turn per line, each line starting with "Agent:" or "Customer:". No title, notes, \
stage directions or blank lines.
- Follow the call spec exactly: the situation, how the call ends, how the customer feels in their last turns, and \
the number of lines.
- Use first names only. Do not include any personal data (phone numbers, email addresses, street addresses, card \
numbers, birth dates, account numbers) unless the spec lists it. When it does, write each value exactly as given, \
character for character.
- Make it sound like real speech: people hesitate, repeat themselves and correct themselves. Agents greet the \
customer, check the account, sometimes put them on a short hold, and close the call.
- Never name the issue category or the customer's mood; show them through what people say."""


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
        f"Length: about {spec['lines']} lines.",
    ]
    if spec["intended"]["issue_category"] in ("billing", "cancellation"):
        lines.append(f"Refund policy on this date: refunds within {spec['refund_window_days']} days of purchase.")
    if spec["small_talk"]:
        lines.append(f"Include a brief off-topic exchange about {spec['small_talk']}.")
    if spec["pii"]:
        lines.append("Personal data the customer says (write exactly as shown):")
        lines += [f"- {p['purpose']}: {p['value']}" for p in spec["pii"]]
    return "\n".join(lines)


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


def estimate_cost(specs: list[dict], model: str, use_batch: bool) -> float:
    input_price, output_price = MODEL_PRICES.get(model, (0.0, 0.0))
    tokens_out = sum(EST_OUTPUT_TOKENS_PER_LINE * s["lines"] for s in specs)
    cost = (len(specs) * EST_INPUT_TOKENS * input_price + tokens_out * output_price) / 1_000_000
    return cost * BATCH_DISCOUNT if use_batch else cost


def pending_specs(limit: int | None = None) -> list[dict]:
    """Planned calls without a transcript yet, in a fixed shuffled order so a --limit sample covers every week."""
    done = done_ids(CALLS_PATH)
    pending = [s for s in read_jsonl(PLAN_PATH) if s["call_id"] not in done]
    random.Random("generation-order").shuffle(pending)
    return pending[:limit] if limit else pending


def record_for(spec: dict, message, model: str, use_batch: bool) -> dict | None:
    """The calls.jsonl row for one reply, or None if the reply isn't a usable transcript."""
    if message.stop_reason != "end_turn":
        return None
    text = next((b.text for b in message.content if b.type == "text"), "")
    transcript = clean_transcript(text)
    if transcript is None:
        return None
    values = [p["value"] for p in spec["pii"]]
    transcript, noisy = add_noise(transcript, spec["call_id"], protected=values)
    usage = Usage(model, message.usage.input_tokens, message.usage.output_tokens, api_calls=1, batch=use_batch)
    return {
        "call_id": spec["call_id"],
        "started_at": spec["started_at"],
        "week": spec["week"],
        "region": spec["region"],
        "product": spec["product"],
        "transcript": transcript,
        "stt_noise": noisy,
        "pii_present": [v in transcript for v in values],  # did the planted value make it in verbatim
        "model": model,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost_usd": round(usage.cost_usd, 6),
    }


def confirm(question: str) -> bool:
    try:
        return input(f"{question} [y/N] ").strip().lower() == "y"
    except EOFError:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="only generate this many calls")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sync", action="store_true", help="send requests one at a time (full price, no waiting for a batch)")
    parser.add_argument("--dry-run", action="store_true", help="print the estimate and one prompt; send nothing")
    parser.add_argument("--yes", action="store_true", help="don't ask before spending")
    args = parser.parse_args(argv)

    if not PLAN_PATH.exists():
        print("No plan yet. Run: uv run python -m callsense.synth.plan", file=sys.stderr)
        return 1
    specs = pending_specs(args.limit)
    use_batch = not args.sync
    estimate = estimate_cost(specs, args.model, use_batch)
    print(f"{len(specs)} calls to generate with {args.model} ({'Batch API, half price' if use_batch else 'one at a time'}).")
    print(f"Estimated cost: ${estimate:.2f}")
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
    requests = {cid: generation_request(s, args.model) for cid, s in by_id.items()}
    written, failed, cost = 0, [], 0.0
    for outcome in run_requests(requests, stage="generate", state_path=BATCH_STATE_PATH, use_batch=use_batch):
        spec = by_id.get(outcome.custom_id)
        if spec is None:
            continue  # from a resumed batch, already saved
        record = record_for(spec, outcome.message, args.model, use_batch) if outcome.message else None
        if record is None:
            failed.append(f"{outcome.custom_id}: {outcome.error or 'not a valid transcript'}")
            continue
        append_jsonl(CALLS_PATH, record)
        written += 1
        cost += record["cost_usd"]

    print(f"Wrote {written} transcripts to data/calls.jsonl. Actual cost: ${cost:.4f}")
    if failed:
        print(f"{len(failed)} failed (re-run to retry them):")
        for line in failed[:10]:
            print(f"  {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
