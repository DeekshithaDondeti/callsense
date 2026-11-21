"""Run the call-analysis extractor over the labeled eval set and report per-field accuracy.

The eval set has two splits:
  dev   - cases used while writing prompts and field definitions. Tune against these.
  test  - held-out cases. Only look at these to measure; never change definitions to fix a test miss.

Usage:
    uv run python -m callsense.evaluate                # all cases
    uv run python -m callsense.evaluate --split dev
    uv run python -m callsense.evaluate --split test
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from callsense.extract import analyze_call, get_client

CASES_PATH = Path(__file__).resolve().parents[2] / "evals" / "cases.json"
SCORED_FIELDS = ["issue_category", "customer_sentiment", "resolved", "follow_up_required", "pii_mentioned"]
SPLITS = ["dev", "test"]


def _print_summary(name: str, rows: list[dict]) -> None:
    n = len(rows)
    if n == 0:
        return
    print(f"\n{name} ({n} cases)")
    for f in SCORED_FIELDS:
        correct = sum(1 for r in rows if f not in r["misses"])
        print(f"  {f:20s} {correct}/{n}  ({correct / n:.0%})")
    exact = sum(1 for r in rows if not r["misses"])
    print(f"  {'every field correct':20s} {exact}/{n}  ({exact / n:.0%})")


def run(split: str = "all", workers: int = 4) -> None:
    cases = json.loads(CASES_PATH.read_text())
    if split != "all":
        cases = [c for c in cases if c["split"] == split]

    get_client()  # create the shared client once, before worker threads start
    with ThreadPoolExecutor(max_workers=workers) as pool:
        outputs = list(pool.map(lambda c: analyze_call(c["transcript"]), cases))

    rows = []
    total_input = total_output = total_calls = 0
    for case, (analysis, usage) in zip(cases, outputs):
        predicted = analysis.model_dump(mode="json")
        total_input += usage.input_tokens
        total_output += usage.output_tokens
        total_calls += usage.api_calls

        misses = {
            f: (case["expected"][f], predicted[f])
            for f in SCORED_FIELDS
            if predicted[f] != case["expected"][f]
        }
        rows.append({"case": case, "misses": misses})

        status = "PASS" if not misses else "MISS"
        print(f"[{status}] ({case['split']}) {case['id']}")
        for f, (want, got) in misses.items():
            print(f"        {f}: expected {want!r}, got {got!r}")

    for s in SPLITS:
        _print_summary(s.upper(), [r for r in rows if r["case"]["split"] == s])
    if split == "all":
        _print_summary("ALL", rows)

    print(f"\nTokens used: {total_input} input, {total_output} output across {total_calls} API calls")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate call analysis against labeled cases.")
    parser.add_argument("--split", choices=["all", *SPLITS], default="all")
    parser.add_argument("--workers", type=int, default=4, help="Parallel API calls.")
    args = parser.parse_args()
    run(split=args.split, workers=args.workers)


if __name__ == "__main__":
    main()
