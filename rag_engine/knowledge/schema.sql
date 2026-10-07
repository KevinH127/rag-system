CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id           BIGSERIAL PRIMARY KEY,
    path         TEXT NOT NULL UNIQUE,
    content_hash TEXT NOT NULL,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    ord         INT NOT NULL,
    heading     TEXT NOT NULL,
    content     TEXT NOT NULL,
    embedding   vector(768) NOT NULL
);

CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks(document_id);

ALTER TABLE chunks
    ADD COLUMN IF NOT EXISTS tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;
CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON chunks USING gin(tsv);
