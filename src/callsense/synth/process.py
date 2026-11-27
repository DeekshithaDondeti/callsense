"""Redact every generated call, then analyse the redacted transcript. Uses the Batch API by default.

Stage 1, redaction: regex locally (free), then Claude lists what regex missed (redact_llm.py).
Stage 2, analysis: analyze_call's exact request on the *redacted* transcript, so raw personal
data never reaches the analysis step.

Re-running is safe: calls already in data/redacted.jsonl or data/analyzed.jsonl are skipped.
Neither output file stores the personal data that was removed, only its kind and position.

Usage:
    uv run python -m callsense.synth.process --dry-run   # free: show what would be sent and the estimate
    uv run python -m callsense.synth.process             # both stages, via the Batch API
"""

import argparse
import sys

from callsense.batch import run_requests
from callsense.extract import (
    BATCH_DISCOUNT,
    DEFAULT_MODEL,
    MODEL_PRICES,
    ExtractionError,
    Usage,
    analysis_request,
    parse_analysis,
)
from callsense.redact import Redaction, apply_redactions, redact
from callsense.redact_llm import apply_findings, mask_repeats, parse_findings, redaction_request
from callsense.synth.generate import confirm
from callsense.synth.store import (
    ANALYZED_PATH,
    BATCH_STATE_PATH,
    CALLS_PATH,
    REDACTED_PATH,
    append_jsonl,
    done_ids,
    read_jsonl,
    write_jsonl,
)

# Per-call (input, output) tokens for the estimate, measured on the 20-call sample. Input includes the
# system prompt and schema.
EST_TOKENS = {"redact": (1650, 90), "analyze": (1840, 450)}


def _usage_fields(message, model: str, use_batch: bool) -> dict:
    usage = Usage(model, message.usage.input_tokens, message.usage.output_tokens, api_calls=1, batch=use_batch)
    return {
        "model": model,
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cost_usd": round(usage.cost_usd, 6),
    }


def estimate_cost(stage: str, count: int, model: str, use_batch: bool) -> float:
    input_price, output_price = MODEL_PRICES.get(model, (0.0, 0.0))
    tokens_in, tokens_out = EST_TOKENS[stage]
    cost = count * (tokens_in * input_price + tokens_out * output_price) / 1_000_000
    return cost * BATCH_DISCOUNT if use_batch else cost


def redact_stage(calls: list[dict], model: str, use_batch: bool) -> tuple[int, list[str], float]:
    regex_results = {c["call_id"]: redact(c["transcript"]) for c in calls}
    by_id = {c["call_id"]: c for c in calls}
    requests = {cid: redaction_request(r, model) for cid, r in regex_results.items()}
    written, failed, cost = 0, [], 0.0
    for outcome in run_requests(requests, stage="redact", state_path=BATCH_STATE_PATH, use_batch=use_batch):
        call = by_id.get(outcome.custom_id)
        if call is None:
            continue
        if outcome.message is None:
            failed.append(f"{outcome.custom_id}: {outcome.error}")
            continue
        try:
            findings = parse_findings(outcome.message, model)
        except ExtractionError as e:
            failed.append(f"{outcome.custom_id}: {e}")
            continue
        result, unlocated = apply_findings(call["transcript"], regex_results[outcome.custom_id], findings)
        row = {
            "call_id": outcome.custom_id,
            "transcript": result.text,
            # Kinds and offsets into the raw transcript; the removed values themselves are not kept.
            "redactions": [{"kind": r.kind, "start": r.start, "end": r.end} for r in result.redactions],
            "regex_redactions": len(regex_results[outcome.custom_id].redactions),
            "unlocated_findings": len(unlocated),
            **_usage_fields(outcome.message, model, use_batch),
        }
        append_jsonl(REDACTED_PATH, row)
        written += 1
        cost += row["cost_usd"]
    return written, failed, cost


def analyze_stage(redacted: list[dict], model: str, use_batch: bool) -> tuple[int, list[str], float]:
    requests = {r["call_id"]: analysis_request(r["transcript"], model) for r in redacted}
    wanted = set(requests)
    written, failed, cost = 0, [], 0.0
    for outcome in run_requests(requests, stage="analyze", state_path=BATCH_STATE_PATH, use_batch=use_batch):
        if outcome.custom_id not in wanted:
            continue
        if outcome.message is None:
            failed.append(f"{outcome.custom_id}: {outcome.error}")
            continue
        try:
            analysis = parse_analysis(outcome.message, model)
        except ExtractionError as e:
            failed.append(f"{outcome.custom_id}: {e}")
            continue
        row = {"call_id": outcome.custom_id, "analysis": analysis.model_dump(mode="json"), **_usage_fields(outcome.message, model, use_batch)}
        append_jsonl(ANALYZED_PATH, row)
        written += 1
        cost += row["cost_usd"]
    return written, failed, cost


def repair_redactions() -> tuple[int, int]:
    """Re-apply stored redactions with mask_repeats, so read-back values are masked too. Free: no API calls.

    Rebuilds each call's redactions from the stored offsets and the raw transcript, so Claude's findings
    don't need to be requested again. Returns (calls changed, redactions added).
    """
    calls = {c["call_id"]: c for c in read_jsonl(CALLS_PATH)}
    rows = read_jsonl(REDACTED_PATH)
    changed = added = 0
    for row in rows:
        raw = calls[row["call_id"]]["transcript"]
        old = [Redaction(r["kind"], r["start"], r["end"], raw[r["start"]:r["end"]]) for r in row["redactions"]]
        if apply_redactions(raw, old) != row["transcript"]:
            raise RuntimeError(f"{row['call_id']}: stored offsets don't reproduce the stored text; not repairing")
        new = mask_repeats(raw, old)
        if len(new) != len(old):
            changed += 1
            added += len(new) - len(old)
            row["transcript"] = apply_redactions(raw, new)
            row["redactions"] = [{"kind": r.kind, "start": r.start, "end": r.end} for r in new]
    if changed:
        backup = REDACTED_PATH.with_name("redacted.before-repeat-fix.jsonl")
        if not backup.exists():
            backup.write_text(REDACTED_PATH.read_text())
        write_jsonl(REDACTED_PATH, rows)
    return changed, added


def _report(stage: str, written: int, failed: list[str], cost: float, path) -> None:
    print(f"[{stage}] wrote {written} rows to data/{path.name}. Actual cost: ${cost:.4f}")
    if failed:
        print(f"[{stage}] {len(failed)} failed (re-run to retry them):")
        for line in failed[:10]:
            print(f"  {line}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--limit", type=int, help="only process this many calls")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--sync", action="store_true", help="send requests one at a time (full price)")
    parser.add_argument("--dry-run", action="store_true", help="print the estimate; send nothing")
    parser.add_argument("--yes", action="store_true", help="don't ask before spending")
    parser.add_argument("--repair-redactions", action="store_true",
                        help="free: mask read-back repeats in data/redacted.jsonl, then exit")
    args = parser.parse_args(argv)
    use_batch = not args.sync

    if args.repair_redactions:
        changed, added = repair_redactions()
        print(f"Masked {added} repeated values in {changed} calls (backup: data/redacted.before-repeat-fix.jsonl).")
        return 0

    redacted_ids = done_ids(REDACTED_PATH)
    to_redact = [c for c in read_jsonl(CALLS_PATH) if c["call_id"] not in redacted_ids][: args.limit]
    analyzed_ids = done_ids(ANALYZED_PATH)
    already_redacted = [r for r in read_jsonl(REDACTED_PATH) if r["call_id"] not in analyzed_ids]
    to_analyze_count = min(len(already_redacted) + len(to_redact), args.limit or sys.maxsize)

    estimate = estimate_cost("redact", len(to_redact), args.model, use_batch) + estimate_cost(
        "analyze", to_analyze_count, args.model, use_batch
    )
    print(f"To redact: {len(to_redact)}. To analyse: up to {to_analyze_count}. Model: {args.model}"
          f" ({'Batch API, half price' if use_batch else 'one at a time'}).")
    print(f"Estimated cost: ${estimate:.2f}")
    if args.dry_run or (not to_redact and not already_redacted):
        return 0
    if not args.yes and not confirm("Send these requests?"):
        print("Nothing sent.")
        return 0

    if to_redact:
        _report("redact", *redact_stage(to_redact, args.model, use_batch), REDACTED_PATH)

    analyzed_ids = done_ids(ANALYZED_PATH)
    to_analyze = [r for r in read_jsonl(REDACTED_PATH) if r["call_id"] not in analyzed_ids][: args.limit]
    if to_analyze:
        _report("analyze", *analyze_stage(to_analyze, args.model, use_batch), ANALYZED_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
