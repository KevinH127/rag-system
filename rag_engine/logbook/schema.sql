-- One row per conversation. Created on its first logged message.
CREATE TABLE IF NOT EXISTS sessions (
    id             UUID PRIMARY KEY,
    channel        TEXT NOT NULL,                 -- cli, discord
    started_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_active_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS sessions_started_idx ON sessions(started_at);

-- Set when the bot closes the conversation; NULL while it is open (or was just left).
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS ended_at TIMESTAMPTZ;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS end_reason TEXT;       -- handoff

-- One row per customer message: what was asked and what the bot did.
CREATE TABLE IF NOT EXISTS question_log (
    id             BIGSERIAL PRIMARY KEY,
    session_id     UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    turn           INT NOT NULL,                  -- 1, 2, ... within the session
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    message        TEXT NOT NULL,                 -- redacted; the raw text is never stored
    redacted       TEXT[] NOT NULL DEFAULT '{}',  -- kinds of secret removed
    intent         TEXT NOT NULL,
    action         TEXT NOT NULL,
    reply          TEXT NOT NULL,
    summary        TEXT,                          -- ticket summary on handoff
    source_path    TEXT,                          -- section an answer was taken from
    source_heading TEXT,
    latency_ms     INT NOT NULL,
    UNIQUE (session_id, turn)
);

CREATE INDEX IF NOT EXISTS question_log_created_idx ON question_log(created_at);

-- How each reply was reached. Internal and bulkier: can be disabled (LOG_DEBUG=false) or pruned.
-- Sections are stored by path and heading, not chunk id: ingest re-creates chunks.
CREATE TABLE IF NOT EXISTS question_debug (
    question_id   BIGINT PRIMARY KEY REFERENCES question_log(id) ON DELETE CASCADE,
    query         TEXT NOT NULL,                  -- unresolved messages joined
    clarify_count INT NOT NULL,
    hits          JSONB NOT NULL,                 -- [{path, heading, distance}]
    gated         BOOLEAN NOT NULL,               -- declined before any LLM call
    decision      JSONB,                          -- the model's proposal; NULL if gated
    checks        JSONB NOT NULL,                 -- [{path, heading, quote, supported}]
    timings_ms    JSONB NOT NULL,                 -- {retrieve, decide, verify}
    settings      JSONB NOT NULL                  -- models and thresholds in effect
);
