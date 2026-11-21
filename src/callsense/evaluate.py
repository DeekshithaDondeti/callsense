"""Run the call-analysis extractor over the labeled eval set and report per-field accuracy.

The eval set has two splits:
  dev   - cases used while writing prompts and field definitions. Tune against these.
  test  - held-out cases. Only look at these to measure; never change definitions to fix a test miss.

Usage:
    uv run python -m callsense.evaluate                          # all cases, default model
    uv run python -m callsense.evaluate --split test
    uv run python -m callsense.evaluate --model claude-sonnet-5-5 --effort low
"""

import argparse
import json
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from callsense.extract import DEFAULT_MODEL, analyze_call, get_client

CASES_PATH = Path(__file__).resolve().parents[2] / "evals" / "cases.json"
SCORED_FIELDS = ["issue_category", "customer_sentiment", "resolved", "follow_up_required", "pii_mentioned"]
SPLITS = ["dev", "test"]


def load_cases(split: str = "all") -> list[dict]:
    cases = json.loads(CASES_PATH.read_text())
    return cases if split == "all" else [c for c in cases if c["split"] == split]


def evaluate(
    cases: list[dict],
    model: str = DEFAULT_MODEL,
    effort: str | None = None,
    workers: int = 4,
) -> list[dict]:
    """Analyze every case and return one row per case with misses and usage."""
    get_client()  # create the shared client once, before worker threads start
    with ThreadPoolExecutor(max_workers=workers) as pool:
        outputs = list(pool.map(lambda c: analyze_call(c["transcript"], model=model, effort=effort), cases))

    rows = []
    for case, (analysis, usage) in zip(cases, outputs):
        predicted = analysis.model_dump(mode="json")
        misses = {
            f: (case["expected"][f], predicted[f])
            for f in SCORED_FIELDS
            if predicted[f] != case["expected"][f]
        }
        rows.append({"case": case, "predicted": predicted, "misses": misses, "usage": usage})
    return rows


def summarize(rows: list[dict]) -> dict:
    """Accuracy per field and split, plus cost and latency."""
    summary = {"n": len(rows), "fields": {}, "splits": {}}
    for f in SCORED_FIELDS:
        summary["fields"][f] = sum(1 for r in rows if f not in r["misses"]) / len(rows)
    for s in SPLITS:
        split_rows = [r for r in rows if r["case"]["split"] == s]
        if split_rows:
            exact = sum(1 for r in split_rows if not r["misses"])
            summary["splits"][s] = {"exact": exact, "n": len(split_rows)}
    summary["exact"] = sum(1 for r in rows if not r["misses"])

    usages = [r["usage"] for r in rows]
    latencies = sorted(u.latency_s for u in usages)
    summary["cost_usd"] = sum(u.cost_usd for u in usages)
    summary["input_tokens"] = sum(u.input_tokens for u in usages)
    summary["output_tokens"] = sum(u.output_tokens for u in usages)
    summary["latency_p50_s"] = statistics.median(latencies)
    summary["latency_p95_s"] = latencies[min(len(latencies) - 1, round(0.95 * (len(latencies) - 1)))]
    return summary


def print_report(rows: list[dict]) -> None:
    for r in rows:
        status = "PASS" if not r["misses"] else "MISS"
        print(f"[{status}] ({r['case']['split']}) {r['case']['id']}")
        for f, (want, got) in r["misses"].items():
            print(f"        {f}: expected {want!r}, got {got!r}")

    for name, subset in [(s.upper(), [r for r in rows if r["case"]["split"] == s]) for s in SPLITS] + [("ALL", rows)]:
        if not subset:
            continue
        n = len(subset)
        print(f"\n{name} ({n} cases)")
        for f in SCORED_FIELDS:
            correct = sum(1 for r in subset if f not in r["misses"])
            print(f"  {f:20s} {correct}/{n}  ({correct / n:.0%})")
        exact = sum(1 for r in subset if not r["misses"])
        print(f"  {'every field correct':20s} {exact}/{n}  ({exact / n:.0%})")

    s = summarize(rows)
    print(
        f"\nTokens: {s['input_tokens']} input, {s['output_tokens']} output | "
        f"cost ${s['cost_usd']:.4f} | latency p50 {s['latency_p50_s']:.1f}s, p95 {s['latency_p95_s']:.1f}s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate call analysis against labeled cases.")
    parser.add_argument("--split", choices=["all", *SPLITS], default="all")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], default=None,
                        help="Defaults to the model's own default effort.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel API calls.")
    args = parser.parse_args()
    print(f"Model: {args.model}  effort: {args.effort or 'model default'}\n")
    print_report(evaluate(load_cases(args.split), model=args.model, effort=args.effort, workers=args.workers))


if __name__ == "__main__":
    main()
