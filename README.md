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

Actual output from `claude-haiku-5-5`, validated against a Pydantic schema (1,604 input and 282 output tokens):

```json
{
  "summary": "Customer's card was declined and the agent checked it; the card issuer is blocking the transaction. The customer was advised to try a different card or contact their bank and said they would call the bank.",
  "issue_category": "billing",
  "customer_sentiment": "negative",
  "resolved": false,
  "follow_up_required": true,
  "action_items": [
    "Customer to contact their card issuer/bank to resolve the blocked transaction",
    "Customer to try a different payment card if needed"
  ],
  "pii_mentioned": true
}
```

## Results

Evaluated on 32 synthetic, hand-labeled transcripts with `claude-haiku-5-5`:

| Field | Dev (18 cases) | Test (14 held-out cases) |
|---|---|---|
| Issue category | 100% | 100% |
| Customer sentiment | 100% | 93% |
| Resolved | 94% | 93% |
| Follow-up required | 100% | 93% |
| Personal data mentioned | 100% | 100% |
| **Every field correct** | **17/18 (94%)** | **12/14 (86%)** |

Across repeated runs the overall score is stable at 28–29 of 32 calls fully correct. A full eval run costs about **$0.01** (about 50K input and 8K output tokens; roughly $0.0003 per call).

### What the remaining errors show

- **Routine company steps.** When the company still has a routine step to do, such as inspecting a returned item or shipping a promised replacement, the model tends to treat it as automatic. The written definitions count it as a person acting, so these calls score as wrong on *resolved* or *follow-up required*.
- **Mild sentiment.** Phrases like "Sounds good." or "that's disappointing… okay, thanks" sit between two sentiment labels and can flip between runs.

## How it works

```text
transcript ──► Claude (forced tool call with JSON schema) ──► Pydantic validation ──► CallAnalysis
                                       ▲                              │
                                       └──── retry with the error ◄───┘ (if invalid)
```

- **Structured output.** The `CallAnalysis` Pydantic model ([`schema.py`](src/callsense/schema.py)) is converted to a JSON schema and given to Claude as a tool it must call. This makes the response machine-readable instead of free text.
- **Validation and retry.** The tool input is validated with Pydantic. If it fails, the validation error is sent back to Claude once so it can correct itself ([`extract.py`](src/callsense/extract.py)).
- **Definitions as the prompt.** Each field's description spells out exactly what counts. For example, *resolved* means the agent completed every action needed on the call, and masked values like `j.m****@example.com` are not personal data. These descriptions travel inside the schema, so the model and the human labeler work from the same rules.
- **Cost tracking.** Token usage is summed across retries for every call.

## Evaluation method

The eval ([`evaluate.py`](src/callsense/evaluate.py), cases in [`evals/cases.json`](evals/cases.json)) follows a few rules to keep the score honest:

- **Dev/test split.** Field definitions were only refined using **dev** cases. **Test** cases are held out and used only for measurement, never for tuning.
- **Labels come from the definitions, not the model.** Every case was labeled against the written definitions before it was run. A label was changed only when it contradicted the written definition. Each such change is recorded in that case's `label_notes`.
- **Repeated runs.** Model output varies between runs, so results were checked across several runs rather than one.
- **Hard cases on purpose.** The set includes a refund that was denied, an outage, a customer who calms down by the end of the call, an SSN shared without being asked, and masked versus unmasked personal data.

Iterating on the definitions took the dev score from 5/8 to 17/18. Most early misses were caused by vague definitions or mislabeled cases, not by the model.

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
  extract.py    Claude call, validation, retry, token usage
  evaluate.py   Eval runner with per-field, per-split accuracy
evals/
  cases.json    32 labeled synthetic transcripts (dev and test)
```

## Limitations

- All transcripts are synthetic and short; real calls are longer, messier, and come from speech-to-text with errors.
- 32 cases is a small eval: one case moves a split's score by 5–7 points.
- `pii_mentioned` only detects personal data; it does not yet redact it.

## Next steps

- Redact personal data from transcripts before storage.
- Grow the eval with dev cases on the "routine company step" boundary and longer, noisier transcripts.
- Compare models (Haiku, Sonnet, a local open-source model) on accuracy, cost, and latency.
- A small web UI to paste a transcript and see the analysis.
