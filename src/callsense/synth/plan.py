"""Plan every synthetic call before any transcript is written. Free, local, and deterministic.

The plan fixes each call's date, region, product, scenario, outcome, the customer's closing mood
and any personal data they say. Five trends are planted on purpose (see PLANTED_EVENTS), so later
questions like "why did billing calls spike in week 7?" have a known true answer.

Usage:
    uv run python -m callsense.synth.plan    # writes data/plan.jsonl and evals/planted_events.json
"""

import json
import random
from collections import Counter, defaultdict
from datetime import date, datetime, time, timedelta

from callsense.synth import scenarios as sc
from callsense.synth.store import EVENTS_PATH, PLAN_PATH, write_jsonl

DEFAULT_SEED = 20260601
START = date(2026, 6, 1)  # a Monday
WEEKS = 12
END = START + timedelta(days=7 * WEEKS - 1)

BASE_WEEKDAY_CALLS = 34
BASE_WEEKEND_CALLS = 17

# Planted events
OUTAGE_DAYS = (date(2026, 7, 14), date(2026, 7, 15), date(2026, 7, 16))  # Tue-Thu of week 7
REFUND_POLICY_DATE = date(2026, 7, 27)  # Monday of week 9: refund window cut from 30 to 14 days
REFUND_DENIED_SHARE = 0.75  # share of refund requests after the change that fall outside 14 days
LOGIN_BUG_START = date(2026, 6, 22)  # Monday of week 4: app 5.2 release
LOGIN_PEAK_PER_WEEKDAY = 12  # extra login calls per weekday by the last day
LOW_RESOLUTION_REGION = "Midwest"
LOW_RESOLUTION_FLIP = 0.55  # share of would-be-resolved Midwest calls that get escalated instead
DOORBELL_DELAY = (date(2026, 8, 3), date(2026, 8, 16))  # weeks 10-11
DOORBELL_EXTRA_PER_WEEKDAY = 6

PII_CALL_SHARE = 0.35
SPOKEN_SHARE = 0.2  # share of phone, card and email values read out as words
LENGTHS = (("short", 6, 10, 0.35), ("medium", 11, 18, 0.45), ("long", 19, 30, 0.20))

PLANTED_EVENTS = [
    {
        "id": "payment_outage",
        "kind": "spike",
        "start": OUTAGE_DAYS[0].isoformat(),
        "end": OUTAGE_DAYS[-1].isoformat(),
        "weeks": [7],
        "what_happened": "The payment processor had an outage for three days. Checkouts and renewals failed, "
        "and some customers were charged without getting an order.",
        "what_shows_in_the_data": "Billing calls roughly triple on July 14-16. The extra calls are about failed "
        "payments, mostly unresolved (agents say it's a known outage) and mostly negative.",
        "scenario": "billing_payment_failed_outage",
    },
    {
        "id": "refund_policy_change",
        "kind": "shift",
        "start": REFUND_POLICY_DATE.isoformat(),
        "end": END.isoformat(),
        "weeks": list(range(9, WEEKS + 1)),
        "what_happened": "On July 27 the refund window was cut from 30 days to 14 days.",
        "what_shows_in_the_data": "From week 9, most refund-request calls end with the refund denied under the new "
        "14-day window, so negative sentiment and unresolved calls rise sharply on refund calls. Call volume "
        "doesn't change.",
        "scenario": "billing_refund_request",
    },
    {
        "id": "app_login_rise",
        "kind": "trend",
        "start": LOGIN_BUG_START.isoformat(),
        "end": END.isoformat(),
        "weeks": list(range(4, WEEKS + 1)),
        "what_happened": "App version 5.2, released on June 22, has a login bug: after entering a password the "
        "app returns to the login screen. It isn't fixed by the end of the data.",
        "what_shows_in_the_data": "Account calls about not being able to log in to the app, which don't exist "
        "before week 4, grow steadily every week from week 4 to week 12. Customers mention updating to 5.2.",
        "scenario": "account_app_login_loop",
    },
    {
        "id": "midwest_low_resolution",
        "kind": "segment",
        "start": START.isoformat(),
        "end": END.isoformat(),
        "weeks": list(range(1, WEEKS + 1)),
        "what_happened": "Midwest agents lack system access to complete refunds, replacements and account changes "
        "themselves, so they escalate to a back-office team.",
        "what_shows_in_the_data": "The Midwest region's resolution rate is far below the other regions in every "
        "week. Its unresolved calls end with the agent saying they lack access and a back-office team will call "
        "back in 3 to 5 business days.",
        "region": LOW_RESOLUTION_REGION,
    },
    {
        "id": "doorbell_shipping_delay",
        "kind": "spike",
        "start": DOORBELL_DELAY[0].isoformat(),
        "end": DOORBELL_DELAY[1].isoformat(),
        "weeks": [10, 11],
        "what_happened": "A carrier backlog at the central warehouse delayed Kestrel Doorbell orders for two weeks.",
        "what_shows_in_the_data": "Shipping calls about the Kestrel Doorbell jump in weeks 10 and 11, mostly about "
        "late orders and a carrier backlog, then return to normal in week 12.",
        "scenario": "ship_doorbell_backorder",
        "product": "Kestrel Doorbell",
    },
]

NON_EVENTS = [
    {
        "id": "cancellations_flat",
        "claim": "Cancellation calls stay roughly flat across all 12 weeks. Nothing was planted, so an answer "
        "claiming a cancellation spike or trend is wrong.",
    },
    {
        "id": "technical_flat",
        "claim": "Technical calls (cameras, hub, app crashes) stay roughly flat. The login bug is an account issue, "
        "not a technical one, under the CallAnalysis definitions.",
    },
]

_DIGIT_WORDS = dict(zip("0123456789", ("oh", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")))


def _spoken_digits(value: str) -> str:
    """'(415) 555-0123' -> 'four one five, five five five, oh one two three'."""
    groups = [g for g in "".join(c if c.isdigit() else " " for c in value).split()]
    return ", ".join(" ".join(_DIGIT_WORDS[d] for d in g) for g in groups)


def _spoken_email(value: str) -> str:
    """'maya.chen42@example.com' -> 'maya dot chen four two at example dot com'."""
    out = []
    for ch in value:
        if ch.isdigit():
            out.append(f" {_DIGIT_WORDS[ch]} ".replace("oh", "zero"))
        elif ch == ".":
            out.append(" dot ")
        elif ch == "@":
            out.append(" at ")
        else:
            out.append(ch)
    return " ".join("".join(out).split())


_PII_PURPOSE = {
    "PHONE": "a phone number for a callback",
    "EMAIL": "the email address on their account",
    "ADDRESS": "their street address",
    "CARD_NUMBER": "a card number, read out to pay or to identify a charge",
    "DATE_OF_BIRTH": "their date of birth, to verify their identity",
}


def _pii_value(kind: str, first: str, rng: random.Random) -> tuple[str, bool]:
    """A synthetic value and whether it's spoken as words."""
    spoken = kind in ("PHONE", "CARD_NUMBER", "EMAIL") and rng.random() < SPOKEN_SHARE
    if kind == "PHONE":
        value = f"({rng.choice(sc.AREA_CODES)}) 555-01{rng.randint(0, 99):02d}"  # 555-01xx is reserved for fiction
        return (_spoken_digits(value) if spoken else value), spoken
    if kind == "EMAIL":
        value = f"{first.lower()}.{rng.choice(sc.LAST_NAMES)}{rng.randint(1, 99)}@example.com"
        return (_spoken_email(value) if spoken else value), spoken
    if kind == "ADDRESS":
        return f"{rng.randint(10, 9899)} {rng.choice(sc.STREETS)} {rng.choice(sc.STREET_SUFFIXES)}", False
    if kind == "CARD_NUMBER":
        value = rng.choice(sc.TEST_CARDS)
        return (_spoken_digits(value) if spoken else value), spoken
    if kind == "DATE_OF_BIRTH":
        return f"{rng.choice(sc.MONTHS)} {rng.randint(1, 28)}, {rng.randint(1950, 2004)}", False
    raise ValueError(kind)


def _mood(resolved: bool, rng: random.Random, negative_share: float | None = None) -> str:
    if negative_share is not None:
        return "negative" if rng.random() < negative_share else "neutral"
    if resolved:
        return rng.choices(("positive", "neutral", "negative"), (0.5, 0.4, 0.1))[0]
    return rng.choices(("negative", "neutral", "positive"), (0.55, 0.4, 0.05))[0]


def _plan_call(scenario: sc.Scenario, day: date, rng: random.Random) -> dict:
    region = rng.choices(sc.REGIONS, sc.REGION_WEIGHTS)[0]
    product = rng.choice(scenario.products)
    events = []
    if scenario.id == "billing_payment_failed_outage":
        events.append("payment_outage")
    elif scenario.id == "account_app_login_loop":
        events.append("app_login_rise")
    elif scenario.id == "ship_doorbell_backorder":
        events.append("doorbell_shipping_delay")

    situation = scenario.situation
    resolved = rng.random() < scenario.resolve_rate
    outcome = scenario.resolved_outcome if resolved else scenario.unresolved_outcome
    negative_share = {"payment_outage": 0.75, "app_login_rise": 0.65}.get(events[0]) if events and not resolved else None

    refund_window = 14 if day >= REFUND_POLICY_DATE else 30
    if scenario.id == "billing_refund_request" and day >= REFUND_POLICY_DATE and rng.random() < REFUND_DENIED_SHARE:
        situation = "The customer bought it about three weeks ago (between 15 and 30 days) and wants a refund."
        resolved, outcome, negative_share = False, sc.REFUND_DENIED_OUTCOME, 0.85
        events.append("refund_policy_change")
    elif region == LOW_RESOLUTION_REGION and resolved and rng.random() < LOW_RESOLUTION_FLIP:
        resolved, outcome = False, sc.MIDWEST_ESCALATION_OUTCOME
        events.append("midwest_low_resolution")

    mood = _mood(resolved, rng, negative_share)
    _, low, high, _ = rng.choices(LENGTHS, [w for *_, w in LENGTHS])[0]
    first = rng.choice(sc.FIRST_NAMES)

    pii = []
    if scenario.pii_kinds and rng.random() < PII_CALL_SHARE:
        count = 2 if len(scenario.pii_kinds) > 1 and rng.random() < 0.2 else 1
        for kind in rng.sample(scenario.pii_kinds, count):
            value, spoken = _pii_value(kind, first, rng)
            pii.append({"kind": kind, "value": value, "spoken": spoken, "purpose": _PII_PURPOSE[kind]})

    started = datetime.combine(day, time(8, 0)) + timedelta(minutes=rng.randint(0, 12 * 60 - 1))
    return {
        "started_at": started.isoformat(),
        "week": (day - START).days // 7 + 1,
        "region": region,
        "product": product,
        "scenario": scenario.id,
        "situation": situation,
        "outcome": outcome,
        "refund_window_days": refund_window,
        "lines": rng.randint(low, high),
        "small_talk": rng.choice(sc.SMALL_TALK_TOPICS) if rng.random() < 0.25 else None,
        "customer_name": first,
        "agent_name": rng.choice(sc.AGENT_NAMES),
        "pii": pii,
        "events": events,
        # What a correct CallAnalysis of this call should say, by construction.
        "intended": {"issue_category": scenario.category, "customer_sentiment": mood, "resolved": resolved},
    }


def build_plan(seed: int = DEFAULT_SEED) -> list[dict]:
    rng = random.Random(seed)
    ordinary = defaultdict(list)
    for s in sc.SCENARIOS:
        if not s.event_only:
            ordinary[s.category].append(s)
    categories = list(sc.CATEGORY_WEIGHTS)
    weights = list(sc.CATEGORY_WEIGHTS.values())
    total_days = (END - START).days

    calls = []
    for offset in range(total_days + 1):
        day = START + timedelta(days=offset)
        weekday = day.weekday() < 5
        base = BASE_WEEKDAY_CALLS if weekday else BASE_WEEKEND_CALLS
        n = round(base * rng.uniform(0.88, 1.12))
        todays = []
        for _ in range(n):
            options = ordinary[rng.choices(categories, weights)[0]]
            todays.append(rng.choices(options, [s.weight for s in options])[0])

        if day in OUTAGE_DAYS:  # two extra billing calls for every normal one: billing triples
            todays += [sc.SCENARIOS_BY_ID["billing_payment_failed_outage"]] * round(2 * n * sc.CATEGORY_WEIGHTS["billing"])
        if day >= LOGIN_BUG_START:
            progress = (day - LOGIN_BUG_START).days / (END - LOGIN_BUG_START).days
            extra = LOGIN_PEAK_PER_WEEKDAY * progress * (1 if weekday else 0.5) * rng.uniform(0.8, 1.2)
            todays += [sc.SCENARIOS_BY_ID["account_app_login_loop"]] * round(extra)
        if DOORBELL_DELAY[0] <= day <= DOORBELL_DELAY[1]:
            extra = DOORBELL_EXTRA_PER_WEEKDAY * (1 if weekday else 0.5) * rng.uniform(0.8, 1.2)
            todays += [sc.SCENARIOS_BY_ID["ship_doorbell_backorder"]] * round(extra)

        calls += [_plan_call(s, day, rng) for s in todays]

    calls.sort(key=lambda c: c["started_at"])
    for i, call in enumerate(calls, start=1):
        call["call_id"] = f"call-{i:05d}"
    return [{"call_id": c.pop("call_id"), **c} for c in calls]


def _weekly(plan: list[dict], keep) -> list[int]:
    counts = Counter(c["week"] for c in plan if keep(c))
    return [counts.get(w, 0) for w in range(1, WEEKS + 1)]


def _share(rows: list[dict], test) -> float | None:
    return round(sum(map(test, rows)) / len(rows), 3) if rows else None


def planted_events(plan: list[dict], seed: int = DEFAULT_SEED) -> dict:
    """The planted trends, with the numbers the plan actually produced (what a perfect analysis would show)."""
    by_day = defaultdict(list)
    for c in plan:
        by_day[c["started_at"][:10]].append(c)
    billing = lambda c: c["intended"]["issue_category"] == "billing"  # noqa: E731
    outage = {d.isoformat() for d in OUTAGE_DAYS}
    nearby_weekdays = [
        d for d in by_day
        if d not in outage and date.fromisoformat(d).weekday() < 5 and 6 <= (date.fromisoformat(d) - START).days // 7 + 1 <= 8
    ]
    refund = [c for c in plan if c["scenario"] == "billing_refund_request"]
    before = [c for c in refund if c["started_at"][:10] < REFUND_POLICY_DATE.isoformat()]
    after = [c for c in refund if c["started_at"][:10] >= REFUND_POLICY_DATE.isoformat()]
    negative = lambda c: c["intended"]["customer_sentiment"] == "negative"  # noqa: E731

    numbers = {
        "payment_outage": {
            "billing_calls_per_outage_day": [sum(map(billing, by_day[d])) for d in sorted(outage)],
            "billing_calls_per_other_weekday_weeks_6_to_8": round(
                sum(sum(map(billing, by_day[d])) for d in nearby_weekdays) / len(nearby_weekdays), 1
            ),
        },
        "refund_policy_change": {
            "refund_calls_before_and_after": [len(before), len(after)],
            "negative_share_before": _share(before, negative),
            "negative_share_after": _share(after, negative),
            "resolved_share_before": _share(before, lambda c: c["intended"]["resolved"]),
            "resolved_share_after": _share(after, lambda c: c["intended"]["resolved"]),
        },
        "app_login_rise": {"login_calls_per_week": _weekly(plan, lambda c: c["scenario"] == "account_app_login_loop")},
        "midwest_low_resolution": {
            "resolved_share_by_region": {
                r: _share([c for c in plan if c["region"] == r], lambda c: c["intended"]["resolved"]) for r in sc.REGIONS
            }
        },
        "doorbell_shipping_delay": {
            "doorbell_shipping_calls_per_week": _weekly(
                plan, lambda c: c["product"] == "Kestrel Doorbell" and c["intended"]["issue_category"] == "shipping"
            )
        },
    }
    non_event_numbers = {
        "cancellations_flat": {"cancellation_calls_per_week": _weekly(plan, lambda c: c["intended"]["issue_category"] == "cancellation")},
        "technical_flat": {"technical_calls_per_week": _weekly(plan, lambda c: c["intended"]["issue_category"] == "technical")},
    }
    return {
        "about": "Ground truth for the synthetic dataset. Every number below is from the plan, i.e. what each call "
        "was written to be. The analysed data (dataset-check.json) is compared against these.",
        "dataset": {
            "seed": seed,
            "start": START.isoformat(),
            "end": END.isoformat(),
            "weeks": WEEKS,
            "week_1_starts": START.isoformat(),
            "calls": len(plan),
            "calls_per_week": _weekly(plan, lambda c: True),
            "regions": list(sc.REGIONS),
            "products": list(sc.PRODUCTS),
        },
        "events": [{**e, "planned_numbers": numbers[e["id"]]} for e in PLANTED_EVENTS],
        "non_events": [{**e, "planned_numbers": non_event_numbers[e["id"]]} for e in NON_EVENTS],
    }


def main() -> None:
    plan = build_plan()
    write_jsonl(PLAN_PATH, plan)
    EVENTS_PATH.write_text(json.dumps(planted_events(plan), indent=2) + "\n")
    pii_calls = sum(1 for c in plan if c["pii"])
    print(f"Planned {len(plan)} calls from {START} to {END} ({pii_calls} with synthetic personal data).")
    print(f"Wrote {PLAN_PATH.relative_to(PLAN_PATH.parents[1])} and {EVENTS_PATH.relative_to(EVENTS_PATH.parents[1])}.")


if __name__ == "__main__":
    main()
