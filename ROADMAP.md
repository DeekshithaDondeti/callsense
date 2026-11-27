# CallSense roadmap

**Done (phase 1):** per-call structured extraction with Claude, a 32-case labeled eval with a dev/test split, a Haiku/Sonnet/Opus comparison on accuracy, cost and latency, and two-pass PII redaction with a recall eval. 84 unit tests run in CI.

**Goal of the next phases:** move from *one call at a time* to *questions across thousands of calls*, which is what a support lead actually asks:

> "Why did complaints spike last week?" · "Which issue grew fastest this month?" · "Show me unresolved billing calls where the customer mentioned cancelling."

Each phase ends with something you can show, and a resume line you can defend in an interview.

---

## Phase 2: a dataset with known answers (about 1 week)

You can't evaluate an agent on "why did X spike?" unless you know the true answer. So generate the data with trends planted on purpose.

- [x] `generate.py`: create about 3,000 synthetic calls spread over 12 weeks. Use Haiku with a fixed seed list of scenarios so the cost stays around $1–2. *(2,874 calls, written by Meta's Muse Spark for $10.03 of free Meta credits.)*
- [x] Plant 4–6 known events and record them in `evals/planted_events.json`. *(Five events plus two non-events.)* Examples:
  - week 7: a payment-gateway outage, so billing complaints triple for 3 days
  - week 9: a new refund policy, so negative sentiment rises on refund calls
  - a slow rise in "app login" issues from week 4 onwards
  - one region or product line with a much lower resolution rate
- [x] Add realistic noise: longer calls, small speech-to-text errors, off-topic small talk.
- [ ] Run every call through `redact` and then `analyze_call` (Haiku: about $0.22 per 1,000 calls). *(Done through the Batch API: $0.76 of Anthropic credits.)*
- [x] Run `synth/check.py` and put the results in the README: redaction recall at scale, label agreement, and whether each planted trend is visible in the analysed data. *(789/789 planted values removed after a fix; four of five trends visible as designed, the login bug visible under "technical".)*

**Done when:** you have about 3,000 analysed calls and a file stating the true trends. **Done.**

Carried forward from the results:
- [ ] Decide whether a login broken by a bug is "account" or "technical", and say so in the field definition. Tune on dev cases only.
- [ ] How-to questions are often analysed as "technical" instead of "other".
- [ ] Stop the redaction second pass masking first names.
- [ ] Write the phase 4 eval questions against the analysed data's categories (for example, the login trend sits under "technical").

## Phase 3: storage and pipeline (about 1 week). This is where your data-engineering experience shows.

- [ ] Run Postgres with pgvector through `docker compose`. Tables: `calls` (raw transcript after redaction, timestamp, product, region) and `call_analysis` (the CallAnalysis fields, model, cost, latency).
- [ ] Embeddings for semantic search. Anthropic has no embeddings API, so use a free local model (e.g. `sentence-transformers` `bge-small-en-v1.5`), or Voyage AI if you want a hosted one.
- [ ] An idempotent batch pipeline: load raw calls → redact → analyse → embed → upsert. Re-running it must not duplicate rows or pay twice for already-analysed calls.
- [ ] Data-quality checks, the same idea as your Dhan.ai framework: no PII patterns in stored text, every analysis row passes the schema, row counts reconcile between stages, and a daily volume check.
- [ ] Optional streaming mode: replay calls through Kafka and raise a spike alert with a windowed count, a smaller version of your Flink work.

**Done when:** one command builds the whole database from scratch, and the data-quality checks pass.

## Phase 4: the analyst agent and its eval (about 1–1.5 weeks). This is the main event.

- [ ] Give Claude three tools:
  - `run_sql(query)`: **read-only** access through a restricted Postgres role, limited to whitelisted views, with a row limit and a timeout
  - `search_calls(text, filters)`: hybrid search (Postgres full-text plus pgvector), returning call IDs and snippets
  - `get_call(id)`: one redacted transcript plus its analysis
- [ ] The agent must cite call IDs and the numbers behind every claim.
- [ ] Agent eval in `evals/agent_cases.json`: 50–100 questions built from the planted events.
  - **Numeric questions** ("how many billing complaints in week 7?") are scored by code against SQL ground truth.
  - **"Why" questions** are scored by a Claude judge with a rubric ("does the answer name the payment outage and week 7?"). Spot-check the judge against your own grading on 20 answers, and report how often you agree.
  - Also track tool-call count, cost and latency per question, and the failure types (wrong SQL, no answer found, made-up claims).
- [ ] Iterate as you did in phase 1, using dev questions only, then report on held-out questions.

**Done when:** a README table shows accuracy on held-out questions before and after your improvements, plus cost and latency per question.

## Phase 5: ship it (about 1 week)

- [ ] A FastAPI app with two endpoints: `/analyze` (one transcript) and `/ask` (an agent question). Add a minimal web page.
- [ ] Docker image, deployed on AWS (ECS Fargate or App Runner, with RDS for Postgres). Set a budget alert, and take it down after the demo recording if cost matters.
- [ ] Log every LLM call (model, tokens, cost, latency, tool calls) to a table, and add a small cost dashboard.
- [ ] A 2-minute demo video: ask "why did complaints spike in week 7?" and show the agent find the outage.
- [ ] Optional, but strong for forward-deployed engineer roles: expose the same three tools as an **MCP server**, so Claude Desktop or any MCP client can query CallSense.

---

## Resume lines this unlocks

| After | Line you can honestly write |
|---|---|
| Phase 1 (now) | Structured extraction with a labeled eval, a three-model cost/accuracy comparison, PII redaction at 100% recall on 22 cases |
| Phase 3 | Idempotent pipeline into Postgres/pgvector with data-quality checks, about 3K calls |
| Phase 4 | A tool-using agent answering trend questions, X% accurate on held-out questions, $Y per question |
| Phase 5 | Deployed on AWS, with per-call cost and latency logging; MCP server |

## Ground rules (keep them, they're why the project is credible)

- Tune on dev, report on held-out test, and keep the split for the agent eval too.
- Put the real numbers in the README, including the bad ones.
- Synthetic data only. Never use Dhan.ai or client data.
- Every phase leaves `uv run pytest` green and CI passing.
