import re
from collections import Counter

import pytest

from callsense.redact import redact
from callsense.synth import plan as p
from callsense.synth.noise import add_noise


@pytest.fixture(scope="module")
def plan():
    return p.build_plan()


@pytest.fixture(scope="module")
def events(plan):
    return {e["id"]: e for e in p.planted_events(plan)["events"]}


def test_plan_is_deterministic(plan):
    assert p.build_plan() == plan


def test_plan_size_dates_and_ids(plan):
    assert 2700 <= len(plan) <= 3200
    ids = [c["call_id"] for c in plan]
    assert len(set(ids)) == len(ids)
    assert all(re.fullmatch(r"[A-Za-z0-9_-]{1,64}", i) for i in ids)  # valid batch custom_id
    assert {c["week"] for c in plan} == set(range(1, 13))
    assert plan[0]["started_at"] >= p.START.isoformat() and plan[-1]["started_at"][:10] <= p.END.isoformat()


def test_outage_triples_billing(events):
    n = events["payment_outage"]["planned_numbers"]
    assert min(n["billing_calls_per_outage_day"]) >= 2.5 * n["billing_calls_per_other_weekday_weeks_6_to_8"]


def test_refund_policy_turns_refund_calls_negative(plan, events):
    n = events["refund_policy_change"]["planned_numbers"]
    assert n["negative_share_after"] - n["negative_share_before"] >= 0.3
    denied = [c for c in plan if "refund_policy_change" in c["events"]]
    assert denied and all(c["started_at"][:10] >= p.REFUND_POLICY_DATE.isoformat() for c in denied)
    assert all(c["refund_window_days"] == 14 for c in denied)


def test_login_calls_start_in_week_4_and_keep_rising(events):
    weekly = events["app_login_rise"]["planned_numbers"]["login_calls_per_week"]
    assert weekly[:3] == [0, 0, 0]
    assert weekly[11] > weekly[8] > weekly[5] > weekly[3] > 0


def test_midwest_resolves_far_less(events):
    shares = events["midwest_low_resolution"]["planned_numbers"]["resolved_share_by_region"]
    assert shares["Midwest"] <= min(v for k, v in shares.items() if k != "Midwest") - 0.2


def test_doorbell_delay_only_in_weeks_10_and_11(plan):
    weeks = Counter(c["week"] for c in plan if "doorbell_shipping_delay" in c["events"])
    assert set(weeks) == {10, 11}


def test_non_events_are_recorded():
    ids = {e["id"] for e in p.planted_events(p.build_plan())["non_events"]}
    assert ids == {"cancellations_flat", "technical_flat"}


def test_written_pii_is_caught_by_regex_and_spoken_pii_has_no_digits(plan):
    items = [item for c in plan for item in c["pii"]]
    assert {i["kind"] for i in items} == {"PHONE", "EMAIL", "ADDRESS", "CARD_NUMBER", "DATE_OF_BIRTH"}
    for item in items:
        if item["spoken"]:
            assert not any(ch.isdigit() for ch in item["value"]), item
        elif item["kind"] != "DATE_OF_BIRTH":  # birth dates need the Claude pass by design
            assert redact(f"Customer: it's {item['value']}.").text != f"Customer: it's {item['value']}.", item


def test_spoken_forms():
    assert p._spoken_digits("(415) 555-0123") == "four one five, five five five, oh one two three"
    assert p._spoken_email("maya.chen40@example.com") == "maya dot chen four zero at example dot com"


TRANSCRIPT = "Agent: Thanks for calling, how can I help?\nCustomer: My number is (415) 555-0123, call me there.\n" + "\n".join(
    f"Customer: I think their app is to slow, and your hub too, line {i}." for i in range(30)
)


def test_noise_is_deterministic_and_protects_pii_lines():
    noisy_ids = [cid for cid in (f"call-{i:05d}" for i in range(1, 40)) if add_noise(TRANSCRIPT, cid, [])[1]]
    assert noisy_ids  # about half of calls get noise
    cid = noisy_ids[0]
    first, changed = add_noise(TRANSCRIPT, cid, ["(415) 555-0123"])
    assert changed and first == add_noise(TRANSCRIPT, cid, ["(415) 555-0123"])[0]
    lines = first.split("\n")
    assert lines[1] == "Customer: My number is (415) 555-0123, call me there."
    assert all(line.startswith(("Agent: ", "Customer: ")) for line in lines)
