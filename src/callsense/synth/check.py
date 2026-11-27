"""Check the processed dataset against the plan. Free: reads local files only.

Three questions:
  1. Did redaction remove every planted piece of personal data? (a miss is a leak)
  2. How often does the analysis agree with what each call was written to be?
  3. Are the planted trends visible in the *analysed* data, which is what an analyst agent will query?

Usage:
    uv run python -m callsense.synth.check    # prints a summary, writes evals/results/dataset-check.json
"""

import json
import re
from collections import Counter, defaultdict
from datetime import date

from callsense.redact import redact
from callsense.synth import plan as p
from callsense.synth.scenarios import REGIONS
from callsense.synth.store import ANALYZED_PATH, CALLS_PATH, CHECK_PATH, PLAN_PATH, REDACTED_PATH, read_jsonl

LOGIN_WORDS = re.compile(r"\blog(?:ging)? ?in\b|\blogin\b|\bsign(?:ing)? ?in\b", re.IGNORECASE)


def _mean(xs) -> float | None:
    xs = list(xs)
    return round(sum(xs) / len(xs), 2) if xs else None


def _share(rows, test) -> float | None:
    return round(sum(1 for r in rows if test(r)) / len(rows), 3) if rows else None


def _weekly(rows, test) -> list[int]:
    counts = Counter(r["week"] for r in rows if test(r))
    return [counts.get(w, 0) for w in range(1, p.WEEKS + 1)]


def fully_redacted(raw: str, value: str, redactions: list[dict]) -> bool:
    """True if every occurrence of value in the raw transcript is entirely covered by redaction spans.

    Partial redaction counts as a leak: masking half a phone number still leaks the other half.
    """
    covered = set()
    for r in redactions:
        covered.update(range(r["start"], r["end"]))
    start = raw.find(value)
    while start != -1:
        if not covered.issuperset(range(start, start + len(value))):
            return False
        start = raw.find(value, start + 1)
    return True


def pii_leaks(plan: dict, calls: dict, redacted: dict) -> dict:
    """Planted values that survived redaction, by kind and form, plus regex hits left in redacted text."""
    tally = defaultdict(lambda: {"present": 0, "leaked": 0})
    leaked_calls = []
    for cid, row in redacted.items():
        for item, present in zip(plan[cid]["pii"], calls[cid]["pii_present"]):
            if not present:
                continue
            key = f"{item['kind']} ({'spoken' if item['spoken'] else 'written'})"
            tally[key]["present"] += 1
            if not fully_redacted(calls[cid]["transcript"], item["value"], row["redactions"]):
                tally[key]["leaked"] += 1
                leaked_calls.append(cid)
    leftover = {cid: [r.kind for r in redact(row["transcript"]).redactions] for cid, row in redacted.items()}
    present = sum(t["present"] for t in tally.values())
    leaked = sum(t["leaked"] for t in tally.values())
    return {
        "planted_values_present": present,
        "planted_values_leaked": leaked,
        "recall": round(1 - leaked / present, 4) if present else None,
        "by_kind": dict(sorted(tally.items())),
        "leaked_call_ids": sorted(set(leaked_calls)),
        "regex_hits_left_in_redacted_text": {cid: kinds for cid, kinds in leftover.items() if kinds},
    }


def label_agreement(plan: dict, analyzed: dict) -> dict:
    out = {}
    for field in ("issue_category", "customer_sentiment", "resolved"):
        pairs = [(plan[cid]["intended"][field], row["analysis"][field]) for cid, row in analyzed.items()]
        out[field] = {
            "agreement": _share(pairs, lambda x: x[0] == x[1]),
            "disagreements": dict(Counter(f"{a} -> {b}" for a, b in pairs if a != b).most_common(8)),
        }
    return out


def event_checks(rows: list[dict]) -> dict:
    """Each planted event measured on analysed rows (call metadata + CallAnalysis), with a pass/fail."""
    a = lambda r, f: r["analysis"][f]  # noqa: E731
    by_day = defaultdict(list)
    for r in rows:
        by_day[r["started_at"][:10]].append(r)
    outage = [d.isoformat() for d in p.OUTAGE_DAYS]
    other_days = [
        d for d in by_day
        if d not in outage and date.fromisoformat(d).weekday() < 5 and 6 <= (date.fromisoformat(d) - p.START).days // 7 + 1 <= 8
    ]
    billing = lambda r: a(r, "issue_category") == "billing"  # noqa: E731
    outage_mean = _mean(sum(map(billing, by_day[d])) for d in outage)
    other_mean = _mean(sum(map(billing, by_day[d])) for d in other_days)

    policy = p.REFUND_POLICY_DATE.isoformat()
    refund = [r for r in rows if billing(r) and "refund" in a(r, "summary").lower()]
    negative = lambda r: a(r, "customer_sentiment") == "negative"  # noqa: E731
    neg_before = _share([r for r in refund if r["started_at"][:10] < policy], negative)
    neg_after = _share([r for r in refund if r["started_at"][:10] >= policy], negative)

    login = _weekly(rows, lambda r: a(r, "issue_category") == "account" and LOGIN_WORDS.search(a(r, "summary")))
    about_login = lambda r: bool(LOGIN_WORDS.search(a(r, "summary")))  # noqa: E731
    login_any = _weekly(rows, about_login)
    login_categories = dict(Counter(a(r, "issue_category") for r in rows if about_login(r)))
    resolved_by_region = {g: _share([r for r in rows if r["region"] == g], lambda r: a(r, "resolved")) for g in REGIONS}
    others = [v for g, v in resolved_by_region.items() if g != p.LOW_RESOLUTION_REGION and v is not None]
    doorbell = _weekly(rows, lambda r: r["product"] == "Kestrel Doorbell" and a(r, "issue_category") == "shipping")
    cancellations = _weekly(rows, lambda r: a(r, "issue_category") == "cancellation")
    technical = _weekly(rows, lambda r: a(r, "issue_category") == "technical")
    technical_not_login = _weekly(rows, lambda r: a(r, "issue_category") == "technical" and not about_login(r))

    def flat(weekly):
        mean = _mean(weekly)
        return bool(mean) and max(weekly) <= 1.6 * mean

    def ratio(x, y):
        return round(x / y, 2) if x is not None and y else None

    midwest = resolved_by_region.get(p.LOW_RESOLUTION_REGION)
    return {
        "payment_outage": {
            "measure": "calls analysed as billing per day",
            "outage_days_mean": outage_mean,
            "other_weekdays_weeks_6_to_8_mean": other_mean,
            "ratio": ratio(outage_mean, other_mean),
            "visible": (ratio(outage_mean, other_mean) or 0) >= 2,
        },
        "refund_policy_change": {
            "measure": "negative share of billing calls whose summary mentions a refund",
            "calls_before_and_after": [len(refund) - sum(r["started_at"][:10] >= policy for r in refund),
                                       sum(r["started_at"][:10] >= policy for r in refund)],
            "negative_share_before": neg_before,
            "negative_share_after": neg_after,
            "visible": neg_before is not None and neg_after is not None and neg_after - neg_before >= 0.2,
        },
        "app_login_rise": {
            "measure": "account calls whose summary mentions logging or signing in, per week",
            "per_week": login,
            "visible": (_mean(login[9:12]) or 0) > 2 * max(_mean(login[3:6]) or 0, 1) and (_mean(login[:3]) or 0) < (_mean(login[9:12]) or 0) / 4,
            # Added after the full run: the analysis filed the login bug under technical, not account.
            "per_week_any_category": login_any,
            "categories_of_login_calls": login_categories,
        },
        "midwest_low_resolution": {
            "measure": "share of calls analysed as resolved, by region",
            "resolved_share": resolved_by_region,
            "visible": midwest is not None and bool(others) and midwest <= min(others) - 0.15,
        },
        "doorbell_shipping_delay": {
            "measure": "Kestrel Doorbell calls analysed as shipping, per week",
            "per_week": doorbell,
            "visible": _mean(doorbell[9:11]) is not None and _mean(doorbell[9:11]) >= 2 * max(_mean(doorbell[:9] + doorbell[11:]) or 0, 1),
        },
        "cancellations_flat": {"per_week": cancellations, "visible_as_flat": flat(cancellations)},
        "technical_flat": {
            "per_week": technical,
            "visible_as_flat": flat(technical),
            "per_week_excluding_login_calls": technical_not_login,
        },
    }


def build_report() -> dict:
    plan = {r["call_id"]: r for r in read_jsonl(PLAN_PATH)}
    calls = {r["call_id"]: r for r in read_jsonl(CALLS_PATH)}
    redacted = {r["call_id"]: r for r in read_jsonl(REDACTED_PATH)}
    analyzed = {r["call_id"]: r for r in read_jsonl(ANALYZED_PATH)}
    rows = [{**calls[cid], "analysis": row["analysis"]} for cid, row in analyzed.items() if cid in calls]
    cost = {name: round(sum(r.get("cost_usd", 0) for r in d.values()), 4)
            for name, d in (("generate", calls), ("redact", redacted), ("analyze", analyzed))}
    return {
        "counts": {"planned": len(plan), "generated": len(calls), "redacted": len(redacted), "analyzed": len(analyzed)},
        "cost_usd": {**cost, "total": round(sum(cost.values()), 4)},
        "calls_with_stt_noise": sum(1 for c in calls.values() if c.get("stt_noise")),
        "pii": pii_leaks(plan, calls, redacted),
        "label_agreement": label_agreement(plan, analyzed),
        "events": event_checks(rows),
    }


def main() -> None:
    report = build_report()
    CHECK_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHECK_PATH.write_text(json.dumps(report, indent=2) + "\n")

    c = report["counts"]
    print(f"Planned {c['planned']}, generated {c['generated']}, redacted {c['redacted']}, analysed {c['analyzed']}.")
    print(f"Cost so far: ${report['cost_usd']['total']:.4f}")
    pii = report["pii"]
    print(f"Personal data: {pii['planted_values_leaked']} of {pii['planted_values_present']} planted values leaked "
          f"(recall {pii['recall']}); {len(pii['regex_hits_left_in_redacted_text'])} redacted calls still have regex hits.")
    for field, v in report["label_agreement"].items():
        print(f"Analysis agrees with the plan on {field}: {v['agreement']}")
    for name, e in report["events"].items():
        ok = e.get("visible", e.get("visible_as_flat"))
        print(f"  {'visible' if ok else 'NOT visible':12} {name}")
    print(f"Wrote {CHECK_PATH.relative_to(CHECK_PATH.parents[2])}")


if __name__ == "__main__":
    main()
