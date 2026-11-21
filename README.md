# CallSense

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

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and an Anthropic API key.

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
evals/
  cases.json    32 labeled synthetic transcripts (dev and test)
  results/      Saved model comparison results
```

## Limitations

- All transcripts are synthetic and short; real calls are longer, messier, and come from speech-to-text with errors.
- 32 cases is a small eval: one case moves a split's score by 5–7 points.
- `pii_mentioned` only detects personal data; it does not yet redact it.

## Next steps

- Have a second person label the sentiment cases to measure how often humans disagree on them.
- Compare effort levels, and a local open-source model, on the same eval.
- Redact personal data from transcripts before storage.
- Grow the eval with longer, noisier transcripts.
- A small web UI to paste a transcript and see the analysis.
