"""Stores every agent run and every step inside it in SQLite."""
import json
import os
import sqlite3
import time
import uuid

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id         TEXT PRIMARY KEY,
    started_at     REAL,
    model          TEXT,
    prompt_version TEXT,
    user_input     TEXT,
    final_output   TEXT,
    status         TEXT,
    latency_ms     INTEGER,
    input_tokens   INTEGER,
    output_tokens  INTEGER
);
CREATE TABLE IF NOT EXISTS steps (
    run_id  TEXT,
    step_no INTEGER,
    kind    TEXT,
    payload TEXT,
    ts      REAL
);
"""


class Tracer:
    def __init__(self, db_path=None):
        self.db_path = db_path or os.getenv("AGENTPROBE_TRACE_DB", "traces.db")
        self.conn = sqlite3.connect(self.db_path)
        self.conn.executescript(SCHEMA)
        self._step_counts = {}

    def start_run(self, model, prompt_version, user_input):
        run_id = uuid.uuid4().hex[:12]
        self.conn.execute(
            "INSERT INTO runs (run_id, started_at, model, prompt_version, user_input, status) "
            "VALUES (?, ?, ?, ?, ?, 'running')",
            (run_id, time.time(), model, prompt_version, user_input),
        )
        self.conn.commit()
        self._step_counts[run_id] = 0
        return run_id

    def log_step(self, run_id, kind, payload):
        self._step_counts[run_id] += 1
        self.conn.execute(
            "INSERT INTO steps VALUES (?, ?, ?, ?, ?)",
            (run_id, self._step_counts[run_id], kind,
             json.dumps(payload, ensure_ascii=False, default=str), time.time()),
        )
        self.conn.commit()

    def end_run(self, run_id, final_output, status, latency_ms, input_tokens, output_tokens):
        self.conn.execute(
            "UPDATE runs SET final_output=?, status=?, latency_ms=?, input_tokens=?, output_tokens=? "
            "WHERE run_id=?",
            (final_output, status, latency_ms, input_tokens, output_tokens, run_id),
        )
        self.conn.commit()
