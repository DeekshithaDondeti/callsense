"""Send many Messages API requests: through the Message Batches API (half price), or one at a time.

Batch mode saves the batch ID to a state file before waiting, so an interrupted run resumes the
same batch instead of submitting (and paying for) the requests again.
"""

import json
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import anthropic

from callsense.extract import get_client


@dataclass
class Outcome:
    """The reply to one request, or why there isn't one."""

    custom_id: str
    message: object | None = None  # anthropic.types.Message on success
    error: str | None = None


def _load_state(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2) + "\n")


def _run_one_by_one(requests: dict[str, dict], workers: int) -> Iterator[Outcome]:
    client = get_client()

    def send(item):
        custom_id, params = item
        try:
            return Outcome(custom_id, message=client.messages.create(**params))
        except anthropic.APIError as e:
            return Outcome(custom_id, error=f"{type(e).__name__}: {e}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        yield from pool.map(send, requests.items())


def _run_as_batch(
    requests: dict[str, dict], stage: str, state_path: Path, poll_s: float, log: Callable[[str], None]
) -> Iterator[Outcome]:
    client = get_client()
    state = _load_state(state_path)

    if stage in state:
        batch_id = state[stage]
        log(f"[{stage}] resuming batch {batch_id} from an earlier run (re-run afterwards to send anything it didn't cover)")
    else:
        batch = client.messages.batches.create(
            requests=[{"custom_id": cid, "params": params} for cid, params in requests.items()]
        )
        batch_id = batch.id
        state[stage] = batch_id
        _save_state(state_path, state)
        log(f"[{stage}] submitted batch {batch_id} with {len(requests)} requests")

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break
        counts = batch.request_counts
        log(f"[{stage}] {batch.processing_status}: {counts.processing} processing, {counts.succeeded} done, {counts.errored} errored")
        time.sleep(poll_s)

    for item in client.messages.batches.results(batch_id):
        result = item.result
        if result.type == "succeeded":
            yield Outcome(item.custom_id, message=result.message)
        elif result.type == "errored":
            yield Outcome(item.custom_id, error=f"errored: {result.error}")
        else:
            yield Outcome(item.custom_id, error=result.type)  # canceled or expired

    # Only forget the batch once every result has been handed to the caller.
    state = _load_state(state_path)
    state.pop(stage, None)
    _save_state(state_path, state)


def run_requests(
    requests: dict[str, dict],
    *,
    stage: str,
    state_path: Path,
    use_batch: bool = True,
    workers: int = 4,
    poll_s: float = 30,
    log: Callable[[str], None] = print,
) -> Iterator[Outcome]:
    """Send each request (custom_id -> Messages API params) and yield an Outcome per reply.

    Results can arrive in any order. Save each one as it is yielded: the state file is only cleared
    after the last result, so a crash part-way through resumes the same batch on the next run.
    """
    if not requests and stage not in _load_state(state_path):
        return
    if use_batch:
        yield from _run_as_batch(requests, stage, state_path, poll_s, log)
    else:
        yield from _run_one_by_one(requests, workers)
