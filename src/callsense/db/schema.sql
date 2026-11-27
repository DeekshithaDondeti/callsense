-- CallSense storage. Safe to run repeatedly.
-- Only redacted text is stored. The synthetic plan (the true answers) is deliberately not loaded,
-- so the phase 4 analyst agent can't read it.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS calls (
    call_id          text PRIMARY KEY,
    started_at       timestamp NOT NULL,
    region           text NOT NULL,
    product          text NOT NULL,
    transcript       text NOT NULL,                 -- redacted
    redaction_count  integer NOT NULL CHECK (redaction_count >= 0),
    content_hash     text NOT NULL,                 -- of the embedded text and model; re-embed only when it changes
    embedding        vector(384),                   -- bge-small-en-v1.5
    search_text      tsvector GENERATED ALWAYS AS (to_tsvector('english', transcript)) STORED
);

CREATE TABLE IF NOT EXISTS call_analysis (
    call_id             text PRIMARY KEY REFERENCES calls (call_id) ON DELETE CASCADE,
    summary             text NOT NULL,
    issue_category      text NOT NULL
        CHECK (issue_category IN ('billing', 'technical', 'account', 'shipping', 'cancellation', 'other')),
    customer_sentiment  text NOT NULL CHECK (customer_sentiment IN ('positive', 'neutral', 'negative')),
    resolved            boolean NOT NULL,
    follow_up_required  boolean NOT NULL,
    action_items        jsonb NOT NULL DEFAULT '[]',
    pii_mentioned       boolean NOT NULL,
    model               text NOT NULL,
    input_tokens        integer NOT NULL,
    output_tokens       integer NOT NULL,
    cost_usd            numeric(12, 6) NOT NULL,
    latency_s           real                        -- NULL for Batch API results, which have no per-call latency
);

CREATE INDEX IF NOT EXISTS calls_started_at_idx ON calls (started_at);
CREATE INDEX IF NOT EXISTS calls_search_idx ON calls USING gin (search_text);
CREATE INDEX IF NOT EXISTS calls_embedding_idx ON calls USING hnsw (embedding vector_cosine_ops);
