"""Compare models on the eval set: accuracy, cost, and latency.

Each model runs the full eval several times, because model output varies between runs.
Results are printed as a table and saved to evals/results/ as JSON.

Usage:
    uv run python -m callsense.compare
    uv run python -m callsense.compare --runs 3 --models claude-haiku-5-5 claude-sonnet-5-5
"""

import argparse
import json
import statistics
from datetime import date
from pathlib import Path

from callsense.evaluate import SCORED_FIELDS, SPLITS, evaluate, load_cases, summarize
from callsense.extract import MODEL_PRICES

RESULTS_DIR = Path(__file__).resolve().parents[2] / "evals" / "results"


def _mean(values: list[float]) -> float:
    return statistics.mean(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare models on the call-analysis eval.")
    parser.add_argument("--models", nargs="+", default=list(MODEL_PRICES))
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--effort", default=None, help="Defaults to each model's own default effort.")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    cases = load_cases()
    results = []
    for model in args.models:
        runs = []
        for i in range(args.runs):
            print(f"{model}: run {i + 1}/{args.runs}...", flush=True)
            rows = evaluate(cases, model=model, effort=args.effort, workers=args.workers)
            runs.append(summarize(rows))
        n = runs[0]["n"]
        results.append({
            "model": model,
            "effort": args.effort or "model default",
            "runs": args.runs,
            "cases": n,
            "exact_per_run": [r["exact"] for r in runs],
            "split_exact_per_run": {s: [r["splits"][s]["exact"] for r in runs] for s in SPLITS},
            "split_sizes": {s: runs[0]["splits"][s]["n"] for s in SPLITS},
            "field_accuracy": {f: _mean([r["fields"][f] for r in runs]) for f in SCORED_FIELDS},
            "cost_per_call_usd": _mean([r["cost_usd"] for r in runs]) / n,
            "cost_per_1000_calls_usd": _mean([r["cost_usd"] for r in runs]) / n * 1000,
            "output_tokens_per_call": _mean([r["output_tokens"] for r in runs]) / n,
            "latency_p50_s": _mean([r["latency_p50_s"] for r in runs]),
            "latency_p95_s": _mean([r["latency_p95_s"] for r in runs]),
        })

    print("\n| Model | Every field correct (per run) | Dev | Test | Cost / 1,000 calls | Output tokens / call | Latency p50 | p95 |")
    print("|---|---|---|---|---|---|---|---|")
    for r in results:
        dev = "/".join(str(x) for x in r["split_exact_per_run"]["dev"])
        test = "/".join(str(x) for x in r["split_exact_per_run"]["test"])
        exact = ", ".join(f"{x}/{r['cases']}" for x in r["exact_per_run"])
        print(
            f"| {r['model']} | {exact} | {dev} of {r['split_sizes']['dev']} | {test} of {r['split_sizes']['test']} "
            f"| ${r['cost_per_1000_calls_usd']:.2f} | {r['output_tokens_per_call']:.0f} "
            f"| {r['latency_p50_s']:.1f}s | {r['latency_p95_s']:.1f}s |"
        )

    print("\nPer-field accuracy (mean across runs):")
    print("| Field | " + " | ".join(r["model"] for r in results) + " |")
    print("|---|" + "---|" * len(results))
    for f in SCORED_FIELDS:
        print(f"| {f} | " + " | ".join(f"{r['field_accuracy'][f]:.0%}" for r in results) + " |")

    total = sum(r["cost_per_call_usd"] * r["cases"] * r["runs"] for r in results)
    print(f"\nTotal spent on this comparison: ${total:.2f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"comparison-{date.today().isoformat()}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"Saved {out.relative_to(RESULTS_DIR.parents[1])}")


if __name__ == "__main__":
    main()
