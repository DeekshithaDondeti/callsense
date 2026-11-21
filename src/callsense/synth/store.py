"""File locations and JSON Lines helpers for the synthetic dataset."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
PLAN_PATH = DATA_DIR / "plan.jsonl"  # what each call should be: scenario, outcome, mood, planted data
CALLS_PATH = DATA_DIR / "calls.jsonl"  # generated raw transcripts (synthetic personal data included)
REDACTED_PATH = DATA_DIR / "redacted.jsonl"  # transcripts after regex + Claude redaction
ANALYZED_PATH = DATA_DIR / "analyzed.jsonl"  # CallAnalysis of each redacted transcript
BATCH_STATE_PATH = DATA_DIR / "batches.json"  # in-flight batch IDs, so a crashed run resumes
EVENTS_PATH = ROOT / "evals" / "planted_events.json"  # the true trends planted in the data
CHECK_PATH = ROOT / "evals" / "results" / "dataset-check.json"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")


def done_ids(path: Path) -> set[str]:
    return {r["call_id"] for r in read_jsonl(path)}
