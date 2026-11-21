"""Measure the regex redactor against labeled cases. Runs locally: no API calls, no cost.

Two numbers matter:
  recall              - share of personal data items fully removed (a miss is a privacy leak)
  false positives     - "must keep" strings that were wrongly removed (over-redaction)

Results are split by form: "written" (digits and standard formats) and "spoken" (digits as words,
as speech-to-text often produces), and by kind of data.

Usage:
    uv run python -m callsense.evaluate_redaction                 # regex only, free
    uv run python -m callsense.evaluate_redaction --with-claude   # regex + Claude, about $0.005 on Haiku
"""

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from callsense.redact import RedactionResult, redact

CASES_PATH = Path(__file__).resolve().parents[2] / "evals" / "redaction_cases.json"


def load_cases() -> list[dict]:
    return json.loads(CASES_PATH.read_text())


def score(cases: list[dict], results: list[RedactionResult] | None = None) -> dict:
    """Return per-item results: which personal data leaked and which safe strings were removed.

    results defaults to the regex redactor's output for each case.
    """
    if results is None:
        results = [redact(c["text"]) for c in cases]
    leaks, wrongly_removed = [], []
    caught = defaultdict(lambda: [0, 0])  # (form, kind) -> [caught, total]
    keeps = [0, 0]  # [kept, total]

    for case, result in zip(cases, results):
        output = result.text
        for item in case["must_redact"]:
            key = (case["form"], item["kind"])
            caught[key][1] += 1
            # Caught only if one redaction covers the whole item; a partial mask still leaks data.
            start = case["text"].index(item["text"])
            end = start + len(item["text"])
            if any(r.start <= start and r.end >= end for r in result.redactions):
                caught[key][0] += 1
            else:
                leaks.append((case["id"], item["kind"], item["text"]))
        for text in case["must_keep"]:
            keeps[1] += 1
            if text in output:
                keeps[0] += 1
            else:
                wrongly_removed.append((case["id"], text))

    return {"caught": dict(caught), "keeps": keeps, "leaks": leaks, "wrongly_removed": wrongly_removed}


def recall(caught: dict, form: str | None = None) -> tuple[int, int]:
    rows = [v for (f, _), v in caught.items() if form is None or f == form]
    return sum(c for c, _ in rows), sum(t for _, t in rows)


def print_report(title: str, result: dict) -> None:
    caught = result["caught"]
    print(f"=== {title} ===")
    print("Recall by kind (items fully removed / items):")
    for form in ("written", "spoken"):
        for (f, kind), (c, t) in sorted(caught.items()):
            if f == form:
                print(f"  {form:8s} {kind:14s} {c}/{t}")

    print()
    for form in ("written", "spoken", None):
        c, t = recall(caught, form)
        print(f"Recall, {form or 'all':8s} {c}/{t}  ({c / t:.0%})")

    kept, total_keep = result["keeps"]
    print(f"Safe strings kept         {kept}/{total_keep}  ({kept / total_keep:.0%})")

    if result["leaks"]:
        print("\nLeaked (not redacted):")
        for case_id, kind, text in result["leaks"]:
            print(f"  {case_id}: {kind} {text!r}")
    if result["wrongly_removed"]:
        print("\nWrongly removed:")
        for case_id, text in result["wrongly_removed"]:
            print(f"  {case_id}: {text!r}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure redaction recall and over-redaction.")
    parser.add_argument("--with-claude", action="store_true", help="Also run the regex + Claude two-pass redactor.")
    parser.add_argument("--model", default=None, help="Model for the Claude pass (default: Haiku).")
    args = parser.parse_args()

    cases = load_cases()
    print_report("Regex only", score(cases))

    if args.with_claude:
        from callsense.extract import DEFAULT_MODEL, get_client
        from callsense.redact_llm import redact_with_claude

        model = args.model or DEFAULT_MODEL
        get_client()
        with ThreadPoolExecutor(max_workers=4) as pool:
            outputs = list(pool.map(lambda c: redact_with_claude(c["text"], model=model), cases))
        print_report(f"Regex + Claude ({model})", score(cases, [o[0] for o in outputs]))

        unlocated = [(c["id"], u) for c, o in zip(cases, outputs) for u in o[2]]
        if unlocated:
            print("Claude findings not found verbatim in the text (ignored):")
            for case_id, text in unlocated:
                print(f"  {case_id}: {text!r}")
        cost = sum(o[1].cost_usd for o in outputs)
        tokens_in = sum(o[1].input_tokens for o in outputs)
        tokens_out = sum(o[1].output_tokens for o in outputs)
        print(f"Claude pass: {tokens_in} input, {tokens_out} output tokens, cost ${cost:.4f}")


if __name__ == "__main__":
    main()
