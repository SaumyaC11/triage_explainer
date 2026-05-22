"""
kafka_producer.py
=================
Generates synthetic ED patient records sampled from REAL data distributions
(mean/std per acuity level) and publishes them to Kafka topic `triage-input`.

Usage:
    pip install kafka-python pandas numpy
    python kafka_producer.py --csv data.csv
    python kafka_producer.py --csv data.csv --broker localhost:9092 --interval 2
"""

import json
import random
import time
import uuid
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from kafka import KafkaProducer


# ── KTAS labels ───────────────────────────────────────────────────────────────
KTAS_LABELS = {
    1: "KTAS 1 – Resuscitation",
    2: "KTAS 2 – Emergent",
    3: "KTAS 3 – Urgent",
    4: "KTAS 4 – Less Urgent",
    5: "KTAS 5 – Non-Urgent",
}

# ── Fallback complaint pools (used only if chief_complaint_raw not in CSV) ────
COMPLAINTS_BY_ACUITY = {
    1: ["cardiac arrest", "unresponsive patient", "severe respiratory failure",
        "massive haemorrhage", "status epilepticus", "anaphylactic shock"],
    2: ["chest pain with diaphoresis", "acute stroke symptoms", "severe dyspnoea",
        "altered mental status", "severe allergic reaction", "high fever with rigors"],
    3: ["abdominal pain", "moderate chest pain", "syncope", "severe headache",
        "fracture with deformity", "acute back pain with radiation"],
    4: ["laceration", "mild headache", "urinary tract infection symptoms",
        "sprained ankle", "mild abdominal discomfort", "ear pain"],
    5: ["prescription refill", "mild sore throat", "minor rash", "cold symptoms",
        "routine wound check", "follow-up after discharge"],
}

# ── Numeric columns to sample from real distributions ────────────────────────
NUMERIC_COLS = [
    "age", "systolic_bp", "diastolic_bp", "mean_arterial_pressure",
    "pulse_pressure", "heart_rate", "respiratory_rate", "temperature_c",
    "spo2", "gcs_total", "pain_score", "weight_kg", "height_cm", "bmi",
    "shock_index", "news2_score", "num_prior_ed_visits_12m",
    "num_prior_admissions_12m", "num_active_medications", "num_comorbidities",
    "arrival_hour", "arrival_month",
]

# ── Categorical columns to sample weighted by real frequency ─────────────────
CATEGORICAL_COLS = [
    "arrival_mode", "arrival_day", "arrival_season", "shift",
    "age_group", "sex", "language", "insurance_type",
    "transport_origin", "pain_location", "mental_status_triage",
    "chief_complaint_system",
]

# ── Medical history binary columns ───────────────────────────────────────────
HX_COLS = [
    "hx_hypertension", "hx_diabetes_type2", "hx_diabetes_type1",
    "hx_asthma", "hx_copd", "hx_heart_failure", "hx_atrial_fibrillation",
    "hx_ckd", "hx_liver_disease", "hx_malignancy", "hx_obesity",
    "hx_depression", "hx_anxiety", "hx_dementia", "hx_epilepsy",
    "hx_hypothyroidism", "hx_hyperthyroidism", "hx_hiv",
    "hx_coagulopathy", "hx_immunosuppressed", "hx_pregnant",
    "hx_substance_use_disorder", "hx_coronary_artery_disease",
    "hx_stroke_prior", "hx_peripheral_vascular_disease",
]

# ── Integer columns (round to whole number after sampling) ───────────────────
INT_COLS = {
    "gcs_total", "pain_score", "arrival_hour", "arrival_month",
    "num_prior_ed_visits_12m", "num_prior_admissions_12m",
    "num_active_medications", "num_comorbidities",
}


# ─────────────────────────────────────────────────────────────
#  REAL DATA LOADER  (runs once at startup)
# ─────────────────────────────────────────────────────────────

_real_df      = None
_acuity_cache = {}


def _load_real_data(csv_path: str):
    global _real_df
    if _real_df is None:
        print(f"[PRODUCER] Loading real data from: {csv_path}")
        _real_df = pd.read_csv(csv_path)
        print(f"[PRODUCER] Loaded {len(_real_df):,} rows")
        for acuity in [1, 2, 3, 4, 5]:
            subset = _real_df[_real_df["triage_acuity"] == acuity]
            _acuity_cache[acuity] = subset
            print(f"  KTAS-{acuity}: {len(subset):,} patients")
    return _real_df


def _get_subset(acuity: int) -> pd.DataFrame:
    subset = _acuity_cache.get(acuity)
    if subset is None or len(subset) == 0:
        return _real_df
    return subset


# ─────────────────────────────────────────────────────────────
#  SAMPLING HELPERS
# ─────────────────────────────────────────────────────────────

def _sample_numeric(subset: pd.DataFrame, col: str):
    """
    Sample from N(mean, std) fitted on this acuity level,
    clipped to the observed [min, max] so no impossible values slip through.
    """
    series = subset[col].dropna()
    if len(series) == 0:
        return np.nan
    mean    = series.mean()
    std     = series.std() if series.std() > 0 else 0.01
    col_min = series.min()
    col_max = series.max()
    val     = float(np.clip(np.random.normal(mean, std), col_min, col_max))
    if col in INT_COLS:
        return int(round(val))
    return round(val, 2)


def _sample_categorical(subset: pd.DataFrame, col: str):
    """
    Pick a category weighted by its real frequency for this acuity level.
    """
    series = subset[col].dropna()
    if len(series) == 0:
        return None
    counts = series.value_counts(normalize=True)
    return str(np.random.choice(counts.index, p=counts.values))


# ─────────────────────────────────────────────────────────────
#  MAIN GENERATE FUNCTION
# ─────────────────────────────────────────────────────────────

def generate_patient(acuity: int = None, csv_path: str = "data.csv", run_id: str = None) -> dict:
    """
    Generate one synthetic patient record whose vitals and demographics
    are sampled from the real data distribution for the given acuity level.

    run_id is stamped on every record so kafka_consumer.py can group messages
    into runs for latency and throughput measurement.
    """
    _load_real_data(csv_path)

    if acuity is None:
        acuity = random.choices([1, 2, 3, 4, 5], weights=[3, 7, 30, 35, 25])[0]

    subset = _get_subset(acuity)

    record = {
        "patient_id":  str(uuid.uuid4())[:8].upper(),
        "timestamp":   datetime.now().isoformat(),
        "produced_at": time.time(),     # epoch float — used for latency calculation
        "run_id":      run_id,          # groups messages into a single measurable run
        "true_acuity": acuity,
    }

    # ── Numeric vitals from real distributions ────────────────
    for col in NUMERIC_COLS:
        if col in subset.columns:
            record[col] = _sample_numeric(subset, col)

    # ── Categorical fields weighted by real frequency ─────────
    for col in CATEGORICAL_COLS:
        if col in subset.columns:
            record[col] = _sample_categorical(subset, col)

    # ── Chief complaint raw text ──────────────────────────────
    if "chief_complaint_raw" in subset.columns:
        record["chief_complaint_raw"] = _sample_categorical(subset, "chief_complaint_raw")
    else:
        record["chief_complaint_raw"] = random.choice(COMPLAINTS_BY_ACUITY[acuity])

    # ── Medical history binary flags ──────────────────────────
    # mean of a binary column = probability of being 1 for this acuity
    for col in HX_COLS:
        if col in subset.columns:
            prob = float(subset[col].dropna().mean())
            record[col] = int(np.random.random() < prob)
        else:
            record[col] = 0

    return record


# ─────────────────────────────────────────────────────────────
#  KAFKA PRODUCER
# ─────────────────────────────────────────────────────────────

def build_producer(broker: str) -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=[broker],
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        acks="all",
        retries=3,
    )


def main():
    parser = argparse.ArgumentParser(description="KTAS Triage Kafka Producer")
    parser.add_argument("--broker",   default="localhost:9092")
    parser.add_argument("--topic",    default="triage-input")
    parser.add_argument("--csv",      default="data.csv",
                        help="Path to real data CSV for distribution sampling")
    parser.add_argument("--interval", type=float, default=2.0,
                        help="Seconds between messages (default: 2)")
    parser.add_argument("--count",    type=int,   default=0,
                        help="Total messages to send (0 = infinite)")
    parser.add_argument("--run-id",   default=None,
                        help="Human-readable run label (default: auto timestamp YYYYMMDD_HHMMSS)")
    args = parser.parse_args()

    # Generate run_id once — shared across all messages in this run.
    # Auto-format: YYYYMMDD_HHMMSS so runs sort chronologically in the metrics tab.
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")

    producer = build_producer(args.broker)
    sent     = 0

    print(f"[PRODUCER] Broker  : {args.broker}")
    print(f"[PRODUCER] Topic   : {args.topic}")
    print(f"[PRODUCER] CSV     : {args.csv}")
    print(f"[PRODUCER] Run ID  : {run_id}")
    print(f"[PRODUCER] Interval: {args.interval}s | "
          f"Count: {'∞' if args.count == 0 else args.count}\n")

    try:
        while args.count == 0 or sent < args.count:
            patient = generate_patient(csv_path=args.csv, run_id=run_id)
            pid     = patient["patient_id"]
            acuity  = patient["true_acuity"]

            producer.send(args.topic, key=pid, value=patient)
            producer.flush()
            sent += 1

            print(f"  [{sent:>4}] Patient {pid} | {KTAS_LABELS[acuity]} | "
                  f"HR={patient.get('heart_rate','?')} "
                  f"SpO2={patient.get('spo2','?')} "
                  f"BP={patient.get('systolic_bp','?')}/{patient.get('diastolic_bp','?')} "
                  f"NEWS2={patient.get('news2_score','?')}")

            time.sleep(args.interval)

    except KeyboardInterrupt:
        print(f"\n[PRODUCER] Stopped after {sent} messages.")
    finally:
        producer.close()


if __name__ == "__main__":
    main()