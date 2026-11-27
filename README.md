# CallSense

[![Tests](https://github.com/DeekshithaDondeti/callsense/actions/workflows/tests.yml/badge.svg)](https://github.com/DeekshithaDondeti/callsense/actions/workflows/tests.yml)

Turn customer support call transcripts into structured, validated data with Claude, and measure how accurate that data is.

Support teams review only a small share of their calls by hand. CallSense reads a transcript and returns the fields a QA lead or support manager would want for every call: what the call was about, how the customer felt at the end, whether the issue was resolved, whether someone still needs to act, and whether the customer shared personal data that should be redacted.

## Example

Input transcript:

```text
Customer: My card was declined. Can you check it?
Agent: Sure, please read me the card number.
Customer: It's 4111 1111 1111 1111, expiry 09/28, CVV 123.
Agent: Thank you. The card issuer is blocking the transaction. Please try a different card or contact your bank.
Customer: Ugh, okay. I'll call the bank.
Agent: Is there anything else?
Customer: No, that's all.
```

Actual output from `claude-haiku-5-5` (1,241 input and 323 output tokens, about $0.0003):

```json
{
  "summary": "Customer reported a declined card. The agent said the card issuer is blocking the transaction and advised trying a different card or contacting the bank, and the customer plans to call the bank.",
  "issue_category": "billing",
  "customer_sentiment": "negative",
  "resolved": false,
  "follow_up_required": true,
  "action_items": [
    "Customer to contact their bank about the blocked transaction"
  ],
  "pii_mentioned": true
}
```

## Results: which model should you use?

All three current Claude models were run on the same 32 synthetic, hand-labeled transcripts, twice each, at each model's default effort level.

| Model | Every field correct | Dev (18) | Held-out test (14) | Cost per 1,000 calls | Latency p50 / p95 |
|---|---|---|---|---|---|
| Haiku 5.5 | 25–26/32 (80%) | 13–14 | 12 | **$0.22** | 1.7s / 3.2s |
| Sonnet 5.5 | 29–30/32 (92%) | 16 | 13–14 | $3.73 | 2.1s / 2.4s |
| Opus 5.5 | **30/32 (94%)** | 16 | **14** | $8.04 | 2.4s / 3.4s |

Scores were identical across both runs except where a range is shown.

| Field (mean of both runs) | Haiku 5.5 | Sonnet 5.5 | Opus 5.5 |
|---|---|---|---|
| Issue category | 97% | 97% | 94% |
| Customer sentiment | 81% | 97% | 100% |
| Resolved | 98% | 100% | 100% |
| Follow-up required | 98% | 98% | 100% |
| Personal data mentioned | 100% | 100% | 100% |

### What this shows

- **Opus is the most accurate.** It got every held-out test case right in both runs, and it costs about 36 times as much per call as Haiku.
- **Sonnet nearly matches Opus at under half the cost.** It's the best default when sentiment matters.
- **Haiku's gap is almost all sentiment.** On the other four fields Haiku scores 97–100%, close to the larger models. Its sentiment misses are borderline endings, such as "Fine." read as negative or "Great, thank you." read as neutral. If a customer mainly needs issue routing, resolution tracking and personal-data flags, Haiku does that at about 1/17 of Sonnet's cost.
- **Latency doesn't separate them.** All three answer in about 2–3 seconds per call for this task.
- **Volume decides the cost question.** At 100,000 calls a month, this costs about $22 on Haiku, $373 on Sonnet, or $804 on Opus.

### Did a written sentiment definition help?

Sentiment was originally the only field without a written definition. One was added (judge the customer's last few turns, with rules for positive, neutral and negative), written from dev-case labels only, and every model was re-run:

| Every field correct | Without definition | With definition |
|---|---|---|
| Haiku 5.5 | 25, 25 | 27, 27, 26, 25 (4 runs) |
| Sonnet 5.5 | 29, 29 | 29, 30 |
| Opus 5.5 | 30, 30 | 30, 30 |

Haiku's gain of about one case is within its run-to-run variation, so the definition didn't reliably close the gap. Haiku's sentiment errors look like a model limit on borderline cases rather than a missing rule. The definition stays anyway: it documents the labeling rules for human reviewers, and it didn't hurt any model.

The full results for both versions are in [`evals/results/`](evals/results/). Each comparison cost $0.77.

**Method notes:** latency was measured with 4 requests in parallel, including network time. Costs use Anthropic's list prices in [`extract.py`](src/callsense/extract.py). With only 32 cases, one case moves a score by about 3 points, so differences of one or two cases between models are within noise.

## Redacting personal data

[`redact.py`](src/callsense/redact.py) masks personal data with placeholders before a transcript is stored or sent anywhere. It runs locally with regular expressions, so it's instant and free:

```text
Customer: It's 4111 1111 1111 1111, expiry 09/28, CVV 123.
Customer: It's [CARD_NUMBER], expiry [CARD_EXPIRY], CVV [CVV].
```

It covers card numbers (with a checksum, so order numbers aren't masked), CVVs and expiry dates when labeled, SSNs, phone numbers, full email addresses and street addresses. Masked values like `j.m****@example.com` and last-four digits are left alone.

Regex alone can't handle context: a date is only personal data when it's someone's birth date, and speech-to-text often writes numbers as words. So [`redact_llm.py`](src/callsense/redact_llm.py) adds a second pass:

1. **Regex first (free).** Written-format data is masked locally and never sent to the API.
2. **Claude second.** The masked transcript goes to Claude, which *lists* any remaining personal data. It never rewrites the transcript: each finding must match the text exactly, the code does the replacement, and anything Claude returns that isn't in the text is ignored. Every place a found value appears is masked, so an agent reading a number back doesn't leak it.

Measured on 22 labeled cases ([`evals/redaction_cases.json`](evals/redaction_cases.json)). A miss here is a privacy leak, so **recall** is the headline number:

| | Regex only | Regex + Claude (Haiku 5.5) |
|---|---|---|
| Written formats: cards, CVVs, expiries, SSNs, phones, emails, addresses | **15/15** | **15/15** |
| Dates of birth | 0/2 | **2/2** |
| Spoken forms ("four one one one…", "priya dot n at example dot com") | 0/5 | **5/5** |
| Safe strings kept (prices, dates, versions, order and tracking numbers) | 16/16 | 16/16 |
| Cost for all 22 cases | $0 | about $0.0024 |

The two-pass results were identical across two runs. On the 32 call transcripts, regex alone flags 7 of the 8 that contain personal data (it misses a date of birth). Regex plus Claude matches all 32 labels: it flags all 8 and none of the other 24.

**Caveats.** The redaction cases are few and were written by the same person who wrote the prompt, so the prompt's "do not list" rules (prices, ordinary dates, order numbers) mirror what the cases test. There's no held-out redaction set yet. Also, spoken-form data regex can't catch does reach the API before it's redacted; a fully local setup would need a local model for the second pass.

A CI test fails if a change makes the regex redactor leak any written-format item it catches today or remove a safe string.

## Synthetic dataset with planted trends (phase 2)

To test questions across many calls ("why did billing complaints spike in week 7?"), the true answer has to be known. So the dataset is built in two steps:

1. **Plan in code (free, deterministic).** [`synth/plan.py`](src/callsense/synth/plan.py) decides every call's date, region, product, scenario (from a fixed list of 28 in [`synth/scenarios.py`](src/callsense/synth/scenarios.py)), how it ends, how the customer feels at the end, and any synthetic personal data they say. That gives 2,874 calls over 12 weeks (June 1 to August 23, 2026) for Kestrel Home, a fictional smart-home company.
2. **A model writes the dialogue only.** [`synth/generate.py`](src/callsense/synth/generate.py) asks Meta's Muse Spark ([`meta.py`](src/callsense/meta.py)) to write each call to its spec, then adds speech-to-text noise in code (swapped homophones, dropped punctuation, repeated words, "um"). Because the code chose the outcome and mood, the trends don't depend on the model. Using a different model family to write the calls also means Claude never analyses its own writing. On a 20-call sample, Muse Spark wrote exactly the requested length 17 times out of 20, against once for Claude Haiku.

Five trends are planted on purpose and written to [`evals/planted_events.json`](evals/planted_events.json), with the numbers the plan produced:

| Event | When | What a correct analysis should show |
|---|---|---|
| Payment processor outage | July 14–16 (week 7) | Billing calls about 3x normal (24–28 a day vs about 8) |
| Refund window cut from 30 to 14 days | from July 27 (week 9) | Negative refund calls go from 25% to 70%, resolved from 66% to 19% |
| App 5.2 login bug | from June 22 (week 4) | Login calls go from 0 a week to 70 a week by week 12 |
| Midwest agents lack system access | all 12 weeks | Midwest resolves 32% of calls vs 66–67% elsewhere |
| Doorbell carrier backlog | weeks 10–11 | Doorbell shipping calls go from about 10 a week to over 40 |

Two non-events are recorded too (cancellations and technical calls stay flat), so an answer that invents a trend can be marked wrong.

After generation, [`synth/process.py`](src/callsense/synth/process.py) redacts each call (regex, then Claude) and analyses the *redacted* transcript, so raw personal data never reaches the analysis step. Both steps send exactly the same requests as `redact_with_claude` and `analyze_call`, through the [Message Batches API](src/callsense/batch.py) at half price. Every stage skips calls it has already done, and an interrupted batch is resumed rather than paid for twice.

[`synth/check.py`](src/callsense/synth/check.py) then measures, from local files:

- **Redaction at scale.** 737 calls carry planted synthetic personal data (fake 555-01xx phone numbers, card networks' test card numbers, example.com emails, some read out as words). Each planted value counts as removed only if redaction covers every character of it.
- **Label agreement** between what each call was written to be and what the analysis says.
- **Whether each planted trend is visible in the analysed data**, which is what the analyst agent will query in phase 4.

### Results on the full dataset

All 2,874 calls were generated, redacted and analysed, with no failed requests. Full numbers are in [`evals/results/dataset-check.json`](evals/results/dataset-check.json).

**Are the planted trends visible in the analysed data?** This is the data an analyst agent will query, so it's the check that matters.

| Event | Analysed data | Visible? |
|---|---|---|
| Payment outage, week 7 | 22.7 billing calls a day vs 7.7 on nearby weekdays (3.0x) | Yes |
| Refund window cut, week 9 | Negative share of refund calls goes from 27% to 56% | Yes |
| Midwest region | 32.6% of calls resolved vs 65–66% in the other regions | Yes |
| Doorbell delay, weeks 10–11 | 48 and 44 calls a week vs 6–15 in other weeks | Yes |
| Login bug, from week 4 | Calls about logging in go from 3–10 a week to 74 by week 12, but filed as **technical**, not account | Yes, under another category |
| Cancellations flat (non-event) | 19–24 a week, with one week at 35 | Yes |
| Technical flat (non-event) | 65 a week in week 1 to 148 in week 12 | **No**, because of the login calls |

The login result is a finding about the field definitions, not the data. Haiku filed 319 of the 320 login-bug calls as *technical* (the app is broken), while the plan labelled them *account*, which the written definition lists for login problems. Both readings are defensible, so the definitions don't say what to do when a bug breaks login. The trend is still there: technical calls that aren't about logging in stay flat (53–80 a week), and calls about logging in grow every week from week 4. For an analyst agent that makes "why did technical calls double?" a realistic question, because the answer isn't in the category name.

**How often does the analysis match what each call was written to be?**

| Field | Agreement | Main disagreements |
|---|---|---|
| Resolved | **99.0%** | 21 resolved calls read as unresolved, 9 the other way |
| Customer sentiment | 83.8% | 312 calm endings read as positive, 133 read as negative |
| Issue category | 78.4% | 329 account calls read as technical (319 of them the login bug), 199 how-to questions ("does it work with my Echo?") read as technical |

The sentiment misses match the phase 1 result: Haiku's weak spot is borderline endings. The how-to misses go against the written definition, which puts questions where nothing is broken under *other*. Both are candidates for the next round of definition work, tuned on the dev cases only.

**Redaction at scale.** 789 planted values appeared in the transcripts. Every one is removed:

| Kind | Written | Spoken as words |
|---|---|---|
| Card numbers | 35/35 | 12/12 |
| Phone numbers | 86/86 | 18/18 |
| Emails | 315/315 | 71/71 |
| Street addresses | 230/230 | — |
| Dates of birth | 22/22 | — |

The first run leaked 7 of them (99.1% recall). All 7 were an agent reading a value back: Claude had found the value, but the code only masked its first occurrence. The fix masks every occurrence of a found value, and the stored data was repaired from the saved offsets without calling the API again. Those 7 calls were analysed before the fix, so the analysis step saw one read-back value in each; all of it is synthetic.

Two smaller redaction flaws remain:
- **Some first names get masked under a wrong label**, for example "James" as GOVERNMENT_ID: 37 names in 18 calls (0.6%). This is over-redaction, not a leak, but the labels are wrong and names aren't meant to be redacted.
- **The generator sometimes invents personal data that wasn't planted**, such as "daniel22 at mailhub dot com" or a made-up street address. Redaction caught the cases seen, but the check can only measure planted values, so unplanned data isn't counted.

**Cost of the full run**

| Step | Model | Cost |
|---|---|---|
| Generate 2,874 calls | Meta Muse Spark | $10.03 of Meta credits |
| Redact | Claude Haiku 5.5, Batch API | $0.23 |
| Analyse | Claude Haiku 5.5, Batch API | $0.53 |

Claude's share was $0.76, or $0.26 per 1,000 calls for redaction plus analysis together.

```bash
uv run python -m callsense.synth.plan                # free: data/plan.jsonl and evals/planted_events.json
uv run python -m callsense.synth.generate --dry-run  # free: cost estimate and an example prompt
uv run python -m callsense.synth.generate            # about $10 of Meta credits (Muse Spark); --provider anthropic for Haiku
uv run python -m callsense.synth.process             # about $0.76 of Anthropic credits: redact, then analyse
uv run python -m callsense.synth.check               # free: writes evals/results/dataset-check.json
```

Each paid command prints its estimate and asks before sending anything. `--limit N` runs a sample spread over all 12 weeks, and `--sync` skips the batch wait at full price.

## Database and pipeline (phase 3)

One command builds a Postgres database (with [pgvector](https://github.com/pgvector/pgvector)) from the processed dataset, then checks it. It makes no API calls.

```bash
uv run python -m callsense.db.build          # start Postgres, load, embed, run the checks
uv run python -m callsense.db.build --reset  # drop the tables and rebuild
uv run python -m callsense.db.quality        # checks only
```

```text
data/*.jsonl ──► load (redacted text only) ──► embed locally ──► upsert ──► data-quality checks
```

- **Tables** ([`schema.sql`](src/callsense/db/schema.sql)): `calls` (redacted transcript, time, region, product, embedding, full-text index) and `call_analysis` (the CallAnalysis fields, model, tokens, cost). Latency is empty for Batch API results. The plan holding the true answers is not loaded, so an analyst agent can't read it.
- **Embeddings:** `bge-small-en-v1.5`, run locally through [fastembed](https://github.com/qdrant/fastembed). The model reads at most 512 tokens and 510 of the 2,874 calls (18%) are longer, so each embedded text starts with the call's summary.
- **Safe to re-run:** rows are upserted by call ID, a call is embedded again only if its text or the model changes, and loading commits in chunks, so an interrupted run resumes. Redaction and analysis stay in `synth.process`, which skips calls it has already paid for.

On the full dataset (Apple Silicon laptop), the first build embedded and loaded 2,874 calls in about 6 minutes; that time includes the one-off model download of about 2 minutes. A re-run took 1 second and changed nothing.

| Data-quality check | Result |
|---|---|
| No personal-data patterns in any stored text | Pass |
| No planted personal data in any stored text (including spoken forms) | Pass |
| Every call has an analysis that passes the schema | Pass, 2,874 rows |
| Row counts match across generate, redact, analyse and both tables | Pass |
| Stored rows match the source files | Pass |
| Every call has an embedding | Pass |
| Daily volume matches the source, with no missing days | Pass, 84 days |
| Daily volume in the usual range | **Warning** on July 14–16: the planted payment outage |

The volume check only warns, because a real spike isn't a pipeline fault. The personal-data checks cover only what a regex or the planted values can find, so invented spoken data would not be detected.

## How it works

```text
transcript ──► Claude (structured output, constrained to the CallAnalysis JSON schema) ──► CallAnalysis
```

- **Structured output.** The `CallAnalysis` Pydantic model ([`schema.py`](src/callsense/schema.py)) is passed to Claude's structured-output mode, which constrains the response to that JSON schema. The SDK parses it into a validated `CallAnalysis` object ([`extract.py`](src/callsense/extract.py)), and the same code works on Haiku, Sonnet and Opus.
- **Definitions as the prompt.** Each field's description spells out exactly what counts. For example, *resolved* means the agent completed every action needed on the call, and masked values like `j.m****@example.com` are not personal data. These descriptions travel inside the schema, so the model and the human labeler work from the same rules.
- **Failure handling.** Refusals, truncated responses and missing output raise a clear error instead of returning partial data.
- **Cost and latency tracking.** Every call records tokens, cost and response time.

## Evaluation method

The eval ([`evaluate.py`](src/callsense/evaluate.py), cases in [`evals/cases.json`](evals/cases.json)) follows a few rules to keep the score honest:

- **Dev/test split.** Field definitions were only refined using **dev** cases. **Test** cases are held out and used only for measurement, never for tuning.
- **Labels come from the definitions, not the model.** Every case was labeled against the written definitions before it was run. A label was changed only when it contradicted the written definition. Each such change is recorded in that case's `label_notes`.
- **Repeated runs.** Model output varies between runs, so results were checked across several runs rather than one.
- **Hard cases on purpose.** The set includes a refund that was denied, an outage, a customer who calms down by the end of the call, an SSN shared without being asked, and masked versus unmasked personal data.

Iterating on the definitions took Haiku's dev score from 5/8 to 17/18 in an earlier version that used forced tool calls instead of structured output. Most early misses were caused by vague definitions or mislabeled cases, not by the model. Switching to structured output (needed because Sonnet 5.5 and Opus 5.5 don't accept forced tool calls) fixed Haiku's *resolved* and *follow-up* misses but exposed its weakness on sentiment.

## Run it

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and an Anthropic API key. The database needs Docker. Generating the synthetic dataset also needs a [Meta Model API](https://dev.meta.ai/docs/) key as `META_API_KEY`.

```bash
git clone https://github.com/DeekshithaDondeti/callsense.git
cd callsense
uv sync
echo "ANTHROPIC_API_KEY=your-key-here" > .env
```

Run the eval:

```bash
uv run python -m callsense.evaluate               # all 32 cases
uv run python -m callsense.evaluate --split dev   # dev cases only
uv run python -m callsense.evaluate --split test  # held-out cases only
uv run python -m callsense.evaluate --model claude-sonnet-5-5
```

Redact a transcript, and measure the redactor (both free, no API calls):

```bash
echo "Call me at (415) 555-0123" | uv run python -m callsense.redact
uv run python -m callsense.evaluate_redaction
```

Add the Claude second pass to the redaction eval (about $0.0024 on Haiku):

```bash
uv run python -m callsense.evaluate_redaction --with-claude
```

Run the tests (free: they use fake model clients and need no API key; the database tests are skipped unless Postgres is running):

```bash
uv run pytest
```

Compare models (about $0.77 for all three models, two runs each):

```bash
uv run python -m callsense.compare
```

Analyze a single transcript from Python:

```python
from callsense.extract import analyze_call

analysis, usage = analyze_call("Customer: I was charged twice...\nAgent: ...")
print(analysis.model_dump_json(indent=2))
print(usage)
```

## Project structure

```text
src/callsense/
  schema.py     CallAnalysis model and field definitions
  extract.py    Claude call with structured output; tokens, cost, latency
  evaluate.py   Eval runner with per-field, per-split accuracy
  compare.py    Runs the eval across models and saves a comparison
  redact.py     Regex redaction of personal data
  redact_llm.py Claude second pass for dates of birth and spoken forms
  evaluate_redaction.py  Redaction recall and false-positive check
  batch.py      Message Batches API runner that resumes interrupted batches
  meta.py       Meta Model API client, used to write synthetic transcripts
  db/           Postgres schema, loader, local embeddings, data-quality checks
  synth/        Synthetic dataset: scenarios, call plan, generation, redact + analyse, checks
evals/
  cases.json    32 labeled synthetic transcripts (dev and test)
  redaction_cases.json   22 labeled redaction cases (written and spoken)
  planted_events.json    The true trends planted in the synthetic dataset
  results/      Saved model comparison results
tests/          Unit tests with a fake Claude client, run by GitHub Actions on every push
```

## Limitations

- All transcripts are synthetic and short; real calls are longer, messier, and come from speech-to-text with errors.
- 32 cases is a small eval: one case moves a split's score by 5–7 points.
- Regex redaction only covers US formats; dates of birth and spoken forms need the Claude pass. Names are not redacted.

## Next steps

- Have a second person label the sentiment cases to measure how often humans disagree on them.
- Compare effort levels, and a local open-source model, on the same eval.
- Grow the redaction cases with a held-out set, names, and non-US formats.
- Grow the eval with longer, noisier transcripts.
- A small web UI to paste a transcript and see the analysis.
