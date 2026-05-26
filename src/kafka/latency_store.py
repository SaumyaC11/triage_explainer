"""
latency_store.py
================
SQLite foundation for the KTAS triage metrics dashboard.
All latency, throughput, and auditability data flows through here.

Two tables
----------
runs
    One row per producer run, written/updated by kafka_consumer.py.
    Stores run-level aggregates (P50, P95, throughput, rail hit rate).

messages
    One row per processed message, written by kafka_consumer.py.
    Stores four raw epoch timestamps + four derived segment durations in ms.

Segment definitions
-------------------
    kafka_transit_ms   = (consumer_received_at - produced_at)      * 1000
    ml_inference_ms    = (ml_done_at - consumer_received_at)        * 1000
    safety_pipeline_ms = (safety_done_at - ml_done_at)             * 1000
    e2e_ms             = (safety_done_at - produced_at)            * 1000
    llm_ms             = (explained_at - safety_done_at)           * 1000  (nullable)

Note: LLM is excluded from e2e — it is async and never blocks the
clinical decision.  It is tracked separately for explainability measurement.

Auditability data
-----------------
Written by feedback_consumer.py via write_feedback().
Stored in the `messages` table (explained_at + llm_ms) and the
`overrides` table (model prediction vs clinician final label).

"""

import json
import sqlite3
import time
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional


DEFAULT_DB = "triage_latency.db"


# ─────────────────────────────────────────────────────────────
#  SCHEMA
# ─────────────────────────────────────────────────────────────

_DDL = """
-- One row per producer run ------------------------------------------------
CREATE TABLE IF NOT EXISTS runs (
    run_id              TEXT    PRIMARY KEY,
    started_at          TEXT,               -- ISO datetime string
    message_count       INTEGER DEFAULT 0,
    duration_secs       REAL,               -- wall-clock seconds, set on finalize
    throughput_msg_sec  REAL,               -- message_count / duration_secs
    p50_e2e_ms          REAL,               -- median end-to-end latency (ms)
    p95_e2e_ms          REAL,               -- 95th-pct end-to-end latency (ms)
    p50_kafka_ms        REAL,
    p95_kafka_ms        REAL,
    p50_ml_ms           REAL,
    p95_ml_ms           REAL,
    p50_safety_ms       REAL,
    p95_safety_ms       REAL,
    p50_llm_ms          REAL,               -- nullable — async, may be incomplete
    p95_llm_ms          REAL,
    rail_hit_rate       REAL,               -- fraction of messages that hit a rail
    finalized           INTEGER DEFAULT 0   -- 1 after finalize_run() is called
);

-- One row per processed message --------------------------------------------
CREATE TABLE IF NOT EXISTS messages (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id               TEXT    NOT NULL,
    patient_id           TEXT    NOT NULL,

    -- raw epoch floats (seconds)
    produced_at          REAL,
    consumer_received_at REAL,
    ml_done_at           REAL,
    safety_done_at       REAL,
    explained_at         REAL,              -- nullable, written by feedback_consumer

    -- derived durations (milliseconds) — computed on insert
    kafka_transit_ms     REAL,
    ml_inference_ms      REAL,
    safety_pipeline_ms   REAL,
    e2e_ms               REAL,
    llm_ms               REAL,             -- nullable

    -- safety outcome flags (for scatter chart colouring)
    safety_rail_triggered INTEGER DEFAULT 0,
    true_acuity           INTEGER,
    predicted_acuity      INTEGER,

    FOREIGN KEY (run_id) REFERENCES runs(run_id)
);

-- Override / accept decisions for the auditability matrix -----------------
CREATE TABLE IF NOT EXISTS overrides (
    id                     INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id             TEXT    NOT NULL,
    feedback_type          TEXT    NOT NULL,  -- override | accepted | guardrail_accepted
    model_predicted_acuity INTEGER,           -- pre-guardrail ML output
    guardrail_acuity       INTEGER,           -- what was shown to clinician
    final_label            INTEGER,           -- clinician's ground truth
    safety_rail_triggered  INTEGER DEFAULT 0,
    safety_rail_reasons    TEXT,              -- JSON list stored as text
    submitted_at           TEXT               -- ISO datetime
);

-- Indices ------------------------------------------------------------------
CREATE INDEX IF NOT EXISTS idx_msg_run       ON messages  (run_id);
CREATE INDEX IF NOT EXISTS idx_msg_patient   ON messages  (patient_id);
CREATE INDEX IF NOT EXISTS idx_ovr_patient   ON overrides (patient_id);
CREATE INDEX IF NOT EXISTS idx_ovr_model     ON overrides (model_predicted_acuity);
CREATE INDEX IF NOT EXISTS idx_ovr_label     ON overrides (final_label);
"""


# ─────────────────────────────────────────────────────────────
#  STORE
# ─────────────────────────────────────────────────────────────

class LatencyStore:
    """Thread-safe SQLite store for latency and auditability data."""

    def __init__(self, db_path: str = DEFAULT_DB):
        self.db_path = str(db_path)
        self._conn   = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        for stmt in _DDL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                self._conn.execute(stmt)
        self._conn.commit()

    # ── Write side ───────────────────────────────────────────

    def ensure_run(self, run_id: str):
        """Create a run row if it doesn't exist yet."""
        exists = self._conn.execute(
            "SELECT 1 FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        if not exists:
            self._conn.execute(
                "INSERT INTO runs (run_id, started_at) VALUES (?, ?)",
                (run_id, datetime.now().isoformat()),
            )
            self._conn.commit()

    def write_message(
        self,
        run_id:               str,
        patient_id:           str,
        produced_at:          float,
        consumer_received_at: float,
        ml_done_at:           float,
        safety_done_at:       float,
        safety_rail_triggered: bool  = False,
        true_acuity:          Optional[int] = None,
        predicted_acuity:     Optional[int] = None,
    ):
        """
        Insert one processed-message row.
        All four core timestamps are required; derived ms columns computed here.
        """
        self.ensure_run(run_id)

        kafka_ms  = (consumer_received_at - produced_at)  * 1000
        ml_ms     = (ml_done_at - consumer_received_at)   * 1000
        safety_ms = (safety_done_at - ml_done_at)         * 1000
        e2e_ms    = (safety_done_at - produced_at)        * 1000

        self._conn.execute("""
            INSERT INTO messages (
                run_id, patient_id,
                produced_at, consumer_received_at, ml_done_at, safety_done_at,
                kafka_transit_ms, ml_inference_ms, safety_pipeline_ms, e2e_ms,
                safety_rail_triggered, true_acuity, predicted_acuity
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id, patient_id,
            produced_at, consumer_received_at, ml_done_at, safety_done_at,
            kafka_ms, ml_ms, safety_ms, e2e_ms,
            int(safety_rail_triggered), true_acuity, predicted_acuity,
        ))

        # Increment run message count
        self._conn.execute(
            "UPDATE runs SET message_count = message_count + 1 WHERE run_id = ?",
            (run_id,),
        )
        self._conn.commit()

    def write_feedback(
        self,
        patient_id:              str,
        feedback_type:           str,
        model_predicted_acuity:  Optional[int],
        final_label:             Optional[int],
        guardrail_acuity:        Optional[int]  = None,
        safety_rail_triggered:   bool           = False,
        safety_rail_reasons:     list           = None,
        explained_at:            Optional[float] = None,
        enrichment_requested_at: Optional[float] = None,
    ):
        """
        Called by feedback_consumer.py or kafka_listener (streamlit) when a
        clinician decision or LLM explanation arrives.

        1. Writes to `overrides` table for the auditability matrix.
        2. Updates `messages.explained_at` + `llm_ms` if explained_at is set.

        llm_ms = (explained_at - enrichment_requested_at) * 1000
            — time from LLM request being dispatched to explanation arriving.
        If enrichment_requested_at is None, falls back to safety_done_at
        (less accurate but still recorded).
        """
        # 1 — auditability row (skip for llm_explained — that's not a clinician decision)
        if feedback_type != "llm_explained":
            self._conn.execute("""
                INSERT INTO overrides (
                    patient_id, feedback_type,
                    model_predicted_acuity, guardrail_acuity, final_label,
                    safety_rail_triggered, safety_rail_reasons, submitted_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                patient_id, feedback_type,
                model_predicted_acuity, guardrail_acuity, final_label,
                int(safety_rail_triggered),
                json.dumps(safety_rail_reasons or []),
                datetime.now().isoformat(),
            ))

        # 2 — close the LLM segment if we have explained_at
        if explained_at is not None:
            if enrichment_requested_at is not None:
                # Accurate: time from LLM dispatch to explanation received
                llm_ms_expr = (explained_at - enrichment_requested_at) * 1000
            else:
                # Fallback: diff from safety_done_at (includes queue time)
                llm_ms_expr = None

            if llm_ms_expr is not None:
                self._conn.execute("""
                    UPDATE messages
                    SET    explained_at = ?,
                           llm_ms       = ?
                    WHERE  patient_id   = ?
                      AND  explained_at IS NULL
                """, (explained_at, llm_ms_expr, patient_id))
            else:
                self._conn.execute("""
                    UPDATE messages
                    SET    explained_at = ?,
                           llm_ms = (? - safety_done_at) * 1000
                    WHERE  patient_id   = ?
                      AND  explained_at IS NULL
                """, (explained_at, explained_at, patient_id))

        self._conn.commit()

    def finalize_run(self, run_id: str):
        """
        Compute and store run-level aggregates (P50/P95, throughput, etc.).
        Call this when the producer signals end-of-run, or on consumer shutdown.
        """
        rows = self._conn.execute("""
            SELECT
                e2e_ms, kafka_transit_ms, ml_inference_ms,
                safety_pipeline_ms, llm_ms, safety_rail_triggered,
                produced_at, safety_done_at
            FROM messages
            WHERE run_id = ?
        """, (run_id,)).fetchall()

        if not rows:
            return

        def _pct(vals, p):
            clean = [v for v in vals if v is not None]
            return float(np.percentile(clean, p)) if clean else None

        e2e      = [r["e2e_ms"]             for r in rows]
        kafka    = [r["kafka_transit_ms"]   for r in rows]
        ml       = [r["ml_inference_ms"]    for r in rows]
        safety   = [r["safety_pipeline_ms"] for r in rows]
        llm      = [r["llm_ms"]             for r in rows]
        rail_hit = [r["safety_rail_triggered"] for r in rows]

        # Wall-clock duration: earliest produced_at → latest safety_done_at
        t_start = min(r["produced_at"]    for r in rows if r["produced_at"])
        t_end   = max(r["safety_done_at"] for r in rows if r["safety_done_at"])
        duration = t_end - t_start if (t_start and t_end) else None
        n        = len(rows)

        self._conn.execute("""
            UPDATE runs SET
                duration_secs      = ?,
                throughput_msg_sec = ?,
                p50_e2e_ms         = ?,
                p95_e2e_ms         = ?,
                p50_kafka_ms       = ?,
                p95_kafka_ms       = ?,
                p50_ml_ms          = ?,
                p95_ml_ms          = ?,
                p50_safety_ms      = ?,
                p95_safety_ms      = ?,
                p50_llm_ms         = ?,
                p95_llm_ms         = ?,
                rail_hit_rate      = ?,
                finalized          = 1
            WHERE run_id = ?
        """, (
            duration,
            n / duration if duration else None,
            _pct(e2e,    50), _pct(e2e,    95),
            _pct(kafka,  50), _pct(kafka,  95),
            _pct(ml,     50), _pct(ml,     95),
            _pct(safety, 50), _pct(safety, 95),
            _pct(llm,    50), _pct(llm,    95),
            sum(rail_hit) / n if n else None,
            run_id,
        ))
        self._conn.commit()

    # ── Read side ────────────────────────────────────────────

    def get_runs(self) -> list[dict]:
        """Return all runs ordered newest-first."""
        rows = self._conn.execute(
            "SELECT * FROM runs ORDER BY started_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_messages(self, run_id: str) -> list[dict]:
        """Return all messages for a given run, ordered by produced_at."""
        rows = self._conn.execute(
            "SELECT * FROM messages WHERE run_id = ? ORDER BY produced_at ASC",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_override_matrix(self) -> dict:
        """
        Return the override confusion matrix.

        Structure:
            {
              model_predicted_acuity (int): {
                  final_label (int): count (int),
                  ...
              },
              ...
            }

        Covers all feedback types (override, accepted, guardrail_accepted).
        Only rows where both model_predicted_acuity and final_label are set.
        """
        rows = self._conn.execute("""
            SELECT
                model_predicted_acuity,
                final_label,
                COUNT(*) AS cnt
            FROM overrides
            WHERE model_predicted_acuity IS NOT NULL
              AND final_label            IS NOT NULL
            GROUP BY model_predicted_acuity, final_label
        """).fetchall()

        matrix: dict = {}
        for r in rows:
            pred  = int(r["model_predicted_acuity"])
            label = int(r["final_label"])
            matrix.setdefault(pred, {})[label] = int(r["cnt"])
        return matrix

    def get_rail_reasons(self) -> list[tuple]:
        """
        Return a ranked list of (reason_text, count) for all fired safety rails.
        Parses the JSON-encoded safety_rail_reasons column.

        Deduplicates by patient_id — one patient counts once per reason even
        if they have multiple override rows (e.g. accepted then overridden).

        Returns: [(reason_str, count), ...] sorted by count descending.
        """
        # Use DISTINCT patient_id + safety_rail_reasons to avoid double-counting
        rows = self._conn.execute("""
            SELECT DISTINCT patient_id, safety_rail_reasons
            FROM overrides
            WHERE safety_rail_triggered = 1
              AND safety_rail_reasons IS NOT NULL
              AND safety_rail_reasons != '[]'
              AND safety_rail_reasons != ''
        """).fetchall()

        counts: dict = {}
        for r in rows:
            try:
                reasons = json.loads(r["safety_rail_reasons"] or "[]")
                for reason in reasons:
                    # Trim to the human-readable part before " → "
                    label = str(reason).split("→")[0].strip() if "→" in str(reason) else str(reason)
                    if label:
                        counts[label] = counts.get(label, 0) + 1
            except (json.JSONDecodeError, TypeError):
                pass

        return sorted(counts.items(), key=lambda x: x[1], reverse=True)

    def get_latest_run_id(self) -> Optional[str]:
        """Return the run_id of the most recent run, or None."""
        row = self._conn.execute(
            "SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
        return row["run_id"] if row else None

    def close(self):
        self._conn.close()