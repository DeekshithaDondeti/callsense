# CallSense

Turns support call transcripts into structured data with Claude, and measures how accurate it is. See README.md for results and ROADMAP.md for the phase plan.

## Commands

```bash
uv run pytest                                        # free: tests use a fake Claude client
uv run python -m callsense.evaluate --split dev      # extraction eval (paid, about $0.01 on Haiku)
uv run python -m callsense.evaluate_redaction        # free; add --with-claude for the paid second pass
uv run python -m callsense.synth.plan                # free: rebuilds data/plan.jsonl and evals/planted_events.json
uv run python -m callsense.synth.generate --dry-run  # free: cost estimate, sends nothing
uv run python -m callsense.synth.check               # free: reads local files only
```

`synth.generate` writes dialogue with Meta's Model API by default (`--provider anthropic` for Claude). `synth.process` redacts and analyses with Claude. Both are paid. They accept `--limit N` (a sample spread over all 12 weeks) and `--sync` (no batch wait, full price), and ask before sending unless given `--yes`.

## API budget

Two keys in `.env`: `ANTHROPIC_API_KEY` (small balance, about $5) and `META_API_KEY` (Meta Model API, $150 of free credits with a card on file, so usage beyond the credits is charged).

- Ask the user before any use of the Anthropic API key, however cheap, and say the estimated cost. Never pass `--yes` or start a paid command without their go-ahead.
- Use Meta for work that doesn't need to be Claude, such as writing synthetic transcripts. Say the expected Meta spend before big runs.
- For Claude, default to `claude-haiku-5-5` and single runs. Use the Batch API (half price) for bulk work.
- Prefer a 20-call `--limit` sample before a full run.

## Rules

- Tests must never call a real API. `tests/conftest.py` fails any test that builds a real Anthropic or Meta client; use the `fake_client`, `fake_api` or `fake_meta` fixtures. Check this whenever a default provider changes.
- Keep `uv run pytest` green and CI passing.
- Tune prompts and definitions on **dev** cases only, and report on held-out **test** cases.
- Change an eval label only when it contradicts the written definition, and record why in `label_notes`.
- Put real numbers in the README, including the bad ones.
- Synthetic data only. Never use Dhan.ai or client data.
- Synthetic personal data must stay obviously fake: 555-01xx phone numbers, card networks' test card numbers, example.com emails.

## Code layout

- `extract.py` / `redact_llm.py` each build their request in one place (`analysis_request`, `redaction_request`), so the sync path and `batch.py` send identical requests. Keep it that way.
- `schema.py` field descriptions are the labeling rules for both the model and human labelers. Changing one changes phase 1 results, so re-run the eval afterwards.
- `synth/plan.py` decides every call (outcome, mood, planted trends) with a fixed seed; the generator model only writes dialogue. If you change scenario wording, rebuild the plan and check that `evals/planted_events.json` numbers are unchanged.

## Data

`data/` is git-ignored. `plan.jsonl` rebuilds for free; `calls.jsonl`, `redacted.jsonl` and `analyzed.jsonl` cost money to regenerate, so don't delete them. Every stage skips calls already done, and `data/batches.json` holds in-flight batch IDs so an interrupted run resumes instead of paying twice.
