"""
feedback_consumer.py
====================
Consumes clinician feedback events from the `triage-feedback` Kafka topic
and persists them to a local SQLite database for future model retraining.

Two feedback types are stored:
  - "override"           : clinician corrected the triage level + wrote a reason
  - "accepted"           : clinician accepted the displayed level (no change)
  - "guardrail_accepted" : rails had already escalated the model; clinician agreed

The `final_label` column is the ground-truth KTAS to use for retraining:
  - override           → clinician's chosen KTAS
  - accepted           → guardrail/predicted KTAS shown to the clinician
  - guardrail_accepted → same as accepted (post-guardrail level)

Usage:
    python feedback_consumer.py --broker localhost:9092 --topic triage-feedback --db triage_feedback.db
"""

import json
import time
import sqlite3
import argparse
import logging
from datetime import datetime
from pathlib import Path

from kafka import KafkaConsumer

# Latency / metrics store — graceful fallback if not yet in place
try:
    from latency_store import LatencyStore
    HAS_LATENCY_STORE = True
except ImportError:
    HAS_LATENCY_STORE = False

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("feedback_consumer")


# ─────────────────────────────────────────────────────────────
#  ALL FIELDS WE PERSIST
#  Grouped so it's easy to see what each is for
# ─────────────────────────────────────────────────────────────

# Identity
IDENTITY_FIELDS = [
    "patient_id",
    "timestamp",
    "processed_at",
    "feedback_type",          # override | accepted | guardrail_accepted
]

# Clinician decision
DECISION_FIELDS = [
    "final_label",            # THE training label (int 1-5)
    "override_reason",        # free text, null if accepted
    "model_predicted_acuity", # raw ML output before any guardrails
    "guardrail_acuity",       # what was displayed to the clinician
    "original_acuity",        # alias kept for audit clarity
    "true_acuity",            # ground-truth from producer (if available)
]

# Vitals (features)
VITAL_FIELDS = [
    "heart_rate",
    "systolic_bp",
    "diastolic_bp",
    "mean_arterial_pressure",
    "pulse_pressure",
    "respiratory_rate",
    "temperature_c",
    "spo2",
    "gcs_total",
    "pain_score",
    "news2_score",
    "shock_index",
]

# Demographics (features)
DEMOGRAPHIC_FIELDS = [
    "age",
    "sex",
    "weight_kg",
    "height_cm",
    "bmi",
    "age_group",
    "language",
    "insurance_type",
    "transport_origin",
]

# Arrival context (features)
ARRIVAL_FIELDS = [
    "arrival_mode",
    "arrival_hour",
    "arrival_day",
    "arrival_month",
    "arrival_season",
    "shift",
]

# Clinical context (features)
CLINICAL_FIELDS = [
    "chief_complaint",
    "chief_complaint_raw",
    "chief_complaint_system",
    "pain_location",
    "mental_status_triage",
    "num_prior_ed_visits_12m",
    "num_prior_admissions_12m",
    "num_active_medications",
    "num_comorbidities",
]

# Medical history binary flags (features)
HISTORY_FIELDS = [
    "hx_hypertension",
    "hx_diabetes_type2",
    "hx_diabetes_type1",
    "hx_asthma",
    "hx_copd",
    "hx_heart_failure",
    "hx_atrial_fibrillation",
    "hx_ckd",
    "hx_liver_disease",
    "hx_malignancy",
    "hx_obesity",
    "hx_depression",
    "hx_anxiety",
    "hx_dementia",
    "hx_epilepsy",
    "hx_hypothyroidism",
    "hx_hyperthyroidism",
    "hx_hiv",
    "hx_coagulopathy",
    "hx_immunosuppressed",
    "hx_pregnant",
    "hx_substance_use_disorder",
    "hx_coronary_artery_disease",
    "hx_stroke_prior",
    "hx_peripheral_vascular_disease",
]

# Derived flag features (features — same ones triage_system.py engineers)
FLAG_FIELDS = [
    "flag_hypotension",
    "flag_hypertension",
    "flag_tachycardia",
    "flag_bradycardia",
    "flag_tachypnea",
    "flag_bradypnea",
    "flag_fever",
    "flag_hypothermia",
    "flag_hypoxia",
    "flag_severe_pain",
    "flag_low_gcs",
    "flag_critical_gcs",
    "hx_total_burden",
    "hx_high_risk_flag",
]

# Guardrail / model metadata (drift analysis, NOT training features)
GUARDRAIL_FIELDS = [
    "confidence",
    "confidence_pct",
    "confidence_state",
    "confidence_gap",
    "confidence_top2_acuity",
    "confidence_top2_pct",
    "safety_rail_triggered",
    "safety_rail_reasons",    # stored as JSON string
    "has_feature_conflict",
    "conflict_reasons",       # stored as JSON string
    "intervention_flags",     # stored as JSON string
    "safety_notes",           # stored as JSON string
    "requires_review",
    "model",                  # model name/version for traceability
]

ALL_COLUMNS = (
    IDENTITY_FIELDS
    + DECISION_FIELDS
    + VITAL_FIELDS
    + DEMOGRAPHIC_FIELDS
    + ARRIVAL_FIELDS
    + CLINICAL_FIELDS
    + HISTORY_FIELDS
    + FLAG_FIELDS
    + GUARDRAIL_FIELDS
)

# Columns stored as JSON blobs (list/dict values)
JSON_COLUMNS = {
    "safety_rail_reasons",
    "conflict_reasons",
    "intervention_flags",
    "safety_notes",
    "probabilities",
}


# ─────────────────────────────────────────────────────────────
#  DATABASE SETUP
# ─────────────────────────────────────────────────────────────

def _col_type(col: str) -> str:
    """SQLite column type inference."""
    if col in ("final_label", "model_predicted_acuity", "guardrail_acuity",
               "original_acuity", "true_acuity", "arrival_hour", "arrival_month",
               "num_prior_ed_visits_12m", "num_prior_admissions_12m",
               "num_active_medications", "num_comorbidities", "gcs_total",
               "pain_score", "confidence_top2_acuity") or col.startswith("hx_") or col.startswith("flag_"):
        return "INTEGER"
    if col in ("confidence", "confidence_gap", "heart_rate", "systolic_bp",
               "diastolic_bp", "mean_arterial_pressure", "pulse_pressure",
               "respiratory_rate", "temperature_c", "spo2", "news2_score",
               "shock_index", "age", "weight_kg", "height_cm", "bmi"):
        return "REAL"
    if col in ("safety_rail_triggered", "has_feature_conflict", "requires_review"):
        return "INTEGER"   # SQLite has no bool; 0/1
    return "TEXT"


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    {cols}
)
""".format(cols=",\n    ".join(f"{c} {_col_type(c)}" for c in ALL_COLUMNS))

# Useful indices for retraining queries
CREATE_INDICES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_feedback_type      ON feedback (feedback_type)",
    "CREATE INDEX IF NOT EXISTS idx_final_label        ON feedback (final_label)",
    "CREATE INDEX IF NOT EXISTS idx_timestamp          ON feedback (timestamp)",
    "CREATE INDEX IF NOT EXISTS idx_model_pred         ON feedback (model_predicted_acuity)",
    "CREATE INDEX IF NOT EXISTS idx_guardrail_acuity   ON feedback (guardrail_acuity)",
    "CREATE INDEX IF NOT EXISTS idx_rail_triggered     ON feedback (safety_rail_triggered)",
    "CREATE INDEX IF NOT EXISTS idx_conf_state         ON feedback (confidence_state)",
]


def init_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # safe for concurrent reads
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(CREATE_TABLE_SQL)
    for sql in CREATE_INDICES_SQL:
        conn.execute(sql)
    conn.commit()
    log.info("DB ready: %s", db_path)
    return conn


# ─────────────────────────────────────────────────────────────
#  ROW BUILDER
# ─────────────────────────────────────────────────────────────

def _safe(val):
    """Flatten lists/dicts to JSON strings for SQLite TEXT columns."""
    if isinstance(val, (list, dict)):
        return json.dumps(val)
    if isinstance(val, bool):
        return int(val)
    return val


def build_row(event: dict) -> dict:
    """
    Map a triage-feedback Kafka event → a flat dict ready for INSERT.

    The event schema (produced by streamlit_app.py) must contain:
        patient      : the full result dict from kafka_consumer.py
        feedback_type: "override" | "accepted" | "guardrail_accepted"
        final_label  : int (1-5) — the clinician's ground-truth KTAS
        override_reason : str | None
    """
    patient       = event.get("patient", {})
    vitals        = patient.get("vitals", {})
    feedback_type = event.get("feedback_type", "unknown")
    final_label   = event.get("final_label")
    override_reason = event.get("override_reason")

    row = {}

    # ── Identity ──────────────────────────────────────────────
    row["patient_id"]   = patient.get("patient_id")
    row["timestamp"]    = patient.get("timestamp", datetime.now().isoformat())
    row["processed_at"] = patient.get("processed_at", datetime.now().isoformat())
    row["feedback_type"]= feedback_type

    # ── Decision ──────────────────────────────────────────────
    row["final_label"]            = final_label
    row["override_reason"]        = override_reason
    row["model_predicted_acuity"] = patient.get("original_acuity")   # pre-guardrail
    row["guardrail_acuity"]       = patient.get("predicted_acuity")   # post-guardrail (what was shown)
    row["original_acuity"]        = patient.get("original_acuity")
    row["true_acuity"]            = patient.get("true_acuity")

    # ── Vitals (from nested vitals dict) ─────────────────────
    for f in VITAL_FIELDS:
        row[f] = _safe(vitals.get(f))

    # ── Demographics, arrival, clinical ──────────────────────
    for f in DEMOGRAPHIC_FIELDS + ARRIVAL_FIELDS + CLINICAL_FIELDS:
        row[f] = _safe(patient.get(f))

    # ── Medical history flags ─────────────────────────────────
    for f in HISTORY_FIELDS:
        raw = patient.get(f)
        row[f] = int(raw) if raw is not None else None

    # ── Derived flag features ─────────────────────────────────
    for f in FLAG_FIELDS:
        raw = patient.get(f)
        row[f] = int(raw) if raw is not None else None

    # ── Guardrail / model metadata ────────────────────────────
    row["confidence"]             = patient.get("confidence")
    row["confidence_pct"]         = patient.get("confidence_pct")
    row["confidence_state"]       = patient.get("confidence_state")
    row["confidence_gap"]         = patient.get("confidence_gap")
    row["confidence_top2_acuity"] = patient.get("confidence_top2_acuity")
    row["confidence_top2_pct"]    = patient.get("confidence_top2_pct")
    row["safety_rail_triggered"]  = int(bool(patient.get("safety_rail_triggered", False)))
    row["safety_rail_reasons"]    = json.dumps(patient.get("safety_rail_reasons", []))
    row["has_feature_conflict"]   = int(bool(patient.get("has_feature_conflict", False)))
    row["conflict_reasons"]       = json.dumps(patient.get("conflict_reasons", []))
    row["intervention_flags"]     = json.dumps(patient.get("intervention_flags", []))
    row["safety_notes"]           = json.dumps(patient.get("safety_notes", []))
    row["requires_review"]        = int(bool(patient.get("requires_review", False)))
    row["model"]                  = patient.get("model")

    return row


def insert_row(conn: sqlite3.Connection, row: dict):
    cols   = [c for c in ALL_COLUMNS if c in row]
    values = [row[c] for c in cols]
    sql    = f"INSERT INTO feedback ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})"
    conn.execute(sql, values)
    conn.commit()


# ─────────────────────────────────────────────────────────────
#  KAFKA CONSUMER
# ─────────────────────────────────────────────────────────────

def build_consumer(broker: str, topic: str, group: str) -> KafkaConsumer:
    return KafkaConsumer(
        topic,
        bootstrap_servers=[broker],
        group_id=group,
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        consumer_timeout_ms=1000,
    )


# ─────────────────────────────────────────────────────────────
#  STATS HELPER  (printed periodically)
# ─────────────────────────────────────────────────────────────

def print_stats(conn: sqlite3.Connection):
    rows = conn.execute("""
        SELECT
            feedback_type,
            COUNT(*)                                          AS n,
            ROUND(AVG(final_label), 2)                        AS avg_label,
            SUM(CASE WHEN safety_rail_triggered=1 THEN 1 ELSE 0 END) AS rail_hits,
            SUM(CASE WHEN has_feature_conflict=1  THEN 1 ELSE 0 END) AS conflicts,
            SUM(CASE WHEN model_predicted_acuity != final_label
                     AND model_predicted_acuity IS NOT NULL
                     THEN 1 ELSE 0 END)                       AS model_wrong
        FROM feedback
        GROUP BY feedback_type
        ORDER BY feedback_type
    """).fetchall()

    total = conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
    log.info("── Feedback DB stats (total=%d) ──────────────────────", total)
    for r in rows:
        log.info(
            "  %-20s  n=%-4d  avg_label=%-4s  rails=%-3d  conflicts=%-3d  model_wrong=%d",
            r["feedback_type"], r["n"], r["avg_label"],
            r["rail_hits"], r["conflicts"], r["model_wrong"],
        )


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Triage Feedback Kafka → SQLite Consumer")
    parser.add_argument("--broker",  default="localhost:9092")
    parser.add_argument("--topic",   default="triage-feedback")
    parser.add_argument("--group",   default="feedback-consumer-group")
    parser.add_argument("--db",      default="triage_feedback.db",
                        help="Path to SQLite database file (created if missing)")
    parser.add_argument("--stats-every", type=int, default=20,
                        help="Print DB stats every N records (default: 20)")
    parser.add_argument("--metrics-db", default="triage_latency.db",
                        help="Path to shared latency/metrics SQLite DB (default: triage_latency.db)")
    args = parser.parse_args()

    conn      = init_db(args.db)
    consumer  = build_consumer(args.broker, args.topic, args.group)
    lat_store = LatencyStore(args.metrics_db) if HAS_LATENCY_STORE else None
    if lat_store:
        log.info("Metrics DB  : %s", args.metrics_db)
    saved    = 0

    log.info("Listening on '%s' → writing to '%s'", args.topic, args.db)
    log.info("Schema: %d columns  (%d feature cols, %d guardrail cols)",
             len(ALL_COLUMNS),
             len(VITAL_FIELDS + DEMOGRAPHIC_FIELDS + ARRIVAL_FIELDS
                 + CLINICAL_FIELDS + HISTORY_FIELDS + FLAG_FIELDS),
             len(GUARDRAIL_FIELDS))
    log.info("-" * 70)

    try:
        while True:
            for msg in consumer:
                event = msg.value
                try:
                    row = build_row(event)

                    if row.get("final_label") is None:
                        log.warning("Skipping event with no final_label: %s",
                                    event.get("patient", {}).get("patient_id"))
                        continue

                    insert_row(conn, row)
                    saved += 1

                    # ── Write to shared latency store for auditability matrix ──
                    if lat_store:
                        try:
                            lat_store.write_feedback(
                                patient_id             = row.get("patient_id"),
                                feedback_type          = row.get("feedback_type"),
                                model_predicted_acuity = row.get("model_predicted_acuity"),
                                final_label            = row.get("final_label"),
                                guardrail_acuity       = row.get("guardrail_acuity"),
                                safety_rail_triggered  = bool(row.get("safety_rail_triggered")),
                                safety_rail_reasons    = json.loads(
                                    row.get("safety_rail_reasons") or "[]"
                                ),
                                # explained_at not available here — LLM segment
                                # is closed by kafka_consumer.py via write_message
                                explained_at           = None,
                            )
                        except Exception as _lat_exc:
                            log.warning("write_feedback failed: %s", _lat_exc)

                    pid    = row.get("patient_id", "?")
                    ftype  = row.get("feedback_type", "?")
                    label  = row.get("final_label", "?")
                    orig   = row.get("model_predicted_acuity", "?")
                    rail   = "RAIL" if row.get("safety_rail_triggered") else ""
                    reason = f' | "{row["override_reason"][:60]}"' \
                             if row.get("override_reason") else ""

                    log.info(
                        "[%4d] Patient %-8s  %-20s  model=%s → label=%s  %s%s",
                        saved, pid, ftype, orig, label, rail, reason,
                    )

                    if saved % args.stats_every == 0:
                        print_stats(conn)

                except Exception as exc:
                    log.error("Failed to save event %s: %s",
                              event.get("patient", {}).get("patient_id", "?"), exc,
                              exc_info=True)

            time.sleep(0.2)

    except KeyboardInterrupt:
        log.info("Stopped after %d records saved.", saved)
        print_stats(conn)
    finally:
        consumer.close()
        conn.close()


if __name__ == "__main__":
    main()