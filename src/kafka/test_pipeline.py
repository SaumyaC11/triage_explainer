"""
test_safety_pipeline.py
========================
Runs the full safety pipeline (rails + confidence + conflict detection)
against rows from a CSV file — no Kafka, no live model needed.

Two modes:
  --mode real        Load model .pkl + CSV, run real predictions
  --mode mock        Run against hand-crafted synthetic cases (no model needed)

Usage:
    python test_safety_pipeline.py --mode mock
    python test_safety_pipeline.py --mode real --model triage_model.pkl --csv data.csv --n 200
    python test_safety_pipeline.py --mode real --model triage_model.pkl --csv data.csv --n 500 --verbose
"""

import argparse
import sys
import json
import pickle
import textwrap
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

# ── Import the pipeline from kafka_consumer ───────────────────────────────────
# If running from the same directory:
try:
    from kafka_consumer import (
        run_safety_pipeline,
        apply_safety_rails,
        classify_and_resolve_confidence,
        detect_feature_conflicts,
        predict_patient,
        load_model_bundle,
        KTAS_LABELS,
    )
except ImportError:
    print("[ERROR] Cannot import kafka_consumer.py — make sure it is in the same directory.")
    sys.exit(1)


# =============================================================================
#  MOCK TEST CASES  (no model needed)
# =============================================================================

MOCK_CASES = [
    # ── Safety rail cases ─────────────────────────────────────────────────────
    {
        "label": "GCS 6 → rail forces KTAS-1",
        "record": {
            "patient_id": "MOCK-01", "true_acuity": 1,
            "chief_complaint_raw": "unresponsive at home",
            "heart_rate": 112, "systolic_bp": 88, "diastolic_bp": 56,
            "spo2": 90, "respiratory_rate": 24, "temperature_c": 37.2,
            "gcs_total": 6, "pain_score": 0, "news2_score": 11,
            "age": 72, "sex": "M", "mental_status_triage": "unresponsive",
        },
        "expect_acuity": 1,
        "expect_rail": True,
    },
    {
        "label": "SpO2 85% → rail forces KTAS-1",
        "record": {
            "patient_id": "MOCK-02", "true_acuity": 1,
            "chief_complaint_raw": "severe shortness of breath",
            "heart_rate": 130, "systolic_bp": 95, "diastolic_bp": 62,
            "spo2": 85, "respiratory_rate": 32, "temperature_c": 37.8,
            "gcs_total": 14, "pain_score": 5, "news2_score": 10,
            "age": 67, "sex": "F", "mental_status_triage": "alert",
        },
        "expect_acuity": 1,
        "expect_rail": True,
    },
    {
        "label": "Sepsis triad → rail forces KTAS-2",
        "record": {
            "patient_id": "MOCK-03", "true_acuity": 2,
            "chief_complaint_raw": "high fever and shaking",
            "heart_rate": 118, "systolic_bp": 102, "diastolic_bp": 64,
            "spo2": 95, "respiratory_rate": 22, "temperature_c": 39.1,
            "gcs_total": 15, "pain_score": 4, "news2_score": 5,
            "age": 54, "sex": "M", "mental_status_triage": "alert",
        },
        "expect_acuity": 2,
        "expect_rail": True,
    },
    {
        "label": "Keyword 'cardiac arrest' → rail forces KTAS-1",
        "record": {
            "patient_id": "MOCK-04", "true_acuity": 1,
            "chief_complaint_raw": "cardiac arrest witnessed by family",
            "heart_rate": 0, "systolic_bp": 0, "diastolic_bp": 0,
            "spo2": 78, "respiratory_rate": 0, "temperature_c": 35.8,
            "gcs_total": 3, "pain_score": 0, "news2_score": 15,
            "age": 61, "sex": "M", "mental_status_triage": "unresponsive",
        },
        "expect_acuity": 1,
        "expect_rail": True,
    },

    # ── Confidence ambiguity cases ────────────────────────────────────────────
    {
        "label": "Chest pain + normal vitals (model might waver KTAS-2/3)",
        "record": {
            "patient_id": "MOCK-05", "true_acuity": 2,
            "chief_complaint_raw": "chest pain",
            "heart_rate": 92, "systolic_bp": 138, "diastolic_bp": 88,
            "spo2": 97, "respiratory_rate": 18, "temperature_c": 37.0,
            "gcs_total": 15, "pain_score": 6, "news2_score": 2,
            "age": 55, "sex": "M", "mental_status_triage": "alert",
        },
        "expect_rail": False,   # no hard rail — keyword KTAS-2 floor will catch it
    },

    # ── Conflict detection cases ──────────────────────────────────────────────
    {
        "label": "Benign complaint + critical vitals → conflict",
        "record": {
            "patient_id": "MOCK-06", "true_acuity": 2,
            "chief_complaint_raw": "mild headache",
            "heart_rate": 155, "systolic_bp": 82, "diastolic_bp": 50,
            "spo2": 89, "respiratory_rate": 28, "temperature_c": 37.2,
            "gcs_total": 12, "pain_score": 3, "news2_score": 9,
            "age": 45, "sex": "F", "mental_status_triage": "verbal",
        },
        "expect_acuity": 1,
        "expect_conflict": True,
    },
    {
        "label": "Elderly + high-risk hx + benign complaint → conflict",
        "record": {
            "patient_id": "MOCK-07", "true_acuity": 3,
            "chief_complaint_raw": "mild sore throat",
            "heart_rate": 88, "systolic_bp": 118, "diastolic_bp": 76,
            "spo2": 96, "respiratory_rate": 18, "temperature_c": 38.6,
            "gcs_total": 15, "pain_score": 2, "news2_score": 3,
            "age": 82, "sex": "M", "mental_status_triage": "alert",
            "hx_heart_failure": 1, "hx_copd": 1, "hx_dementia": 1,
        },
        "expect_conflict": True,
    },
    {
        "label": "NEWS2=8 vs KTAS-4 prediction → conflict",
        "record": {
            "patient_id": "MOCK-08", "true_acuity": 2,
            "chief_complaint_raw": "feeling unwell",
            "heart_rate": 122, "systolic_bp": 94, "diastolic_bp": 60,
            "spo2": 91, "respiratory_rate": 26, "temperature_c": 38.9,
            "gcs_total": 14, "pain_score": 3, "news2_score": 8,
            "age": 66, "sex": "F", "mental_status_triage": "verbal",
        },
        "expect_conflict": True,
    },

    # ── Clean cases (no layer should fire) ───────────────────────────────────
    {
        "label": "Clean KTAS-5 — prescription refill, normal vitals",
        "record": {
            "patient_id": "MOCK-09", "true_acuity": 5,
            "chief_complaint_raw": "prescription refill request",
            "heart_rate": 72, "systolic_bp": 122, "diastolic_bp": 78,
            "spo2": 99, "respiratory_rate": 14, "temperature_c": 36.8,
            "gcs_total": 15, "pain_score": 0, "news2_score": 0,
            "age": 34, "sex": "F", "mental_status_triage": "alert",
        },
        "expect_rail": False,
        "expect_conflict": False,
    },
    {
        "label": "Clean KTAS-3 — abdominal pain, mildly abnormal vitals",
        "record": {
            "patient_id": "MOCK-10", "true_acuity": 3,
            "chief_complaint_raw": "abdominal pain",
            "heart_rate": 98, "systolic_bp": 124, "diastolic_bp": 80,
            "spo2": 97, "respiratory_rate": 18, "temperature_c": 37.5,
            "gcs_total": 15, "pain_score": 6, "news2_score": 1,
            "age": 42, "sex": "M", "mental_status_triage": "alert",
        },
        "expect_rail": False,
        "expect_conflict": False,
    },
]


# =============================================================================
#  MOCK MODE — runs safety layers directly without a trained model
# =============================================================================

def _mock_result(record: dict, assumed_acuity: int) -> dict:
    """Fabricate a base prediction result to feed into safety layers."""
    # Simulate a model that returns 'assumed_acuity' with moderate confidence
    proba_by_class = {1: 0.03, 2: 0.07, 3: 0.10, 4: 0.10, 5: 0.05}
    # Give the assumed class a score that creates an AMBIGUOUS scenario for MOCK-05
    if assumed_acuity == 3 and record["patient_id"] == "MOCK-05":
        proba_by_class = {1: 0.02, 2: 0.49, 3: 0.41, 4: 0.05, 5: 0.03}
    else:
        proba_by_class[assumed_acuity] = 0.62
    total = sum(proba_by_class.values())
    proba_by_class = {k: v / total for k, v in proba_by_class.items()}

    top_class = max(proba_by_class, key=proba_by_class.get)
    confidence = proba_by_class[top_class]

    probabilities = {
        KTAS_LABELS.get(c, str(c)): round(p, 4)
        for c, p in proba_by_class.items()
    }

    vitals_keys = ["heart_rate", "systolic_bp", "diastolic_bp", "spo2",
                   "respiratory_rate", "temperature_c", "gcs_total",
                   "pain_score", "news2_score"]

    return {
        "patient_id":          record["patient_id"],
        "timestamp":           "2025-01-01T00:00:00",
        "predicted_acuity":    int(top_class),
        "original_acuity":     int(top_class),
        "true_acuity":         record.get("true_acuity"),
        "label":               KTAS_LABELS.get(int(top_class), ""),
        "color":               "#888",
        "confidence":          round(confidence, 4),
        "confidence_pct":      f"{confidence*100:.1f}%",
        "confidence_state":    "UNKNOWN",
        "is_emergency":        int(top_class) <= 2,
        "probabilities":       probabilities,
        "_proba_by_class":     proba_by_class,
        "chief_complaint":     record.get("chief_complaint_raw", ""),
        "vitals":              {k: record.get(k) for k in vitals_keys},
        "age":                 record.get("age"),
        "sex":                 record.get("sex"),
        "model":               "mock",
        "safety_rail_triggered": False,
        "safety_rail_reasons":   [],
        "has_feature_conflict":  False,
        "conflict_reasons":      [],
        "requires_review":       False,
        "safety_notes":          [],
        "processed_at":          "2025-01-01T00:00:00",
    }


def run_mock_tests(verbose: bool = True):
    print("\n" + "=" * 70)
    print("  MOCK TEST MODE — no model required")
    print("=" * 70)

    passed = failed = 0

    for case in MOCK_CASES:
        rec   = case["record"]
        label = case["label"]

        # Use true_acuity as the assumed model output for mock
        assumed = rec.get("true_acuity", 3)
        result  = _mock_result(rec, assumed)

        # Run all three safety layers
        result = apply_safety_rails(result, rec)
        result = classify_and_resolve_confidence(result)
        result = detect_feature_conflicts(result, rec)
        result.pop("_proba_by_class", None)

        final  = result["predicted_acuity"]
        orig   = result["original_acuity"]
        rail   = result["safety_rail_triggered"]
        conf   = result["confidence_state"]
        cflt   = result["has_feature_conflict"]
        flags  = []
        if rail:  flags.append("RAIL")
        if conf in ("AMBIGUOUS", "LOW"): flags.append(conf)
        if cflt:  flags.append("CONFLICT")

        # Validate expectations
        ok = True
        failures = []
        if "expect_acuity" in case and final != case["expect_acuity"]:
            ok = False
            failures.append(
                f"acuity={final} (expected {case['expect_acuity']})"
            )
        if "expect_rail" in case and rail != case["expect_rail"]:
            ok = False
            failures.append(
                f"rail={rail} (expected {case['expect_rail']})"
            )
        if "expect_conflict" in case and cflt != case["expect_conflict"]:
            ok = False
            failures.append(
                f"conflict={cflt} (expected {case['expect_conflict']})"
            )

        status = "✓ PASS" if ok else "✗ FAIL"
        if ok:
            passed += 1
        else:
            failed += 1

        escalated = f" ↑{orig}→{final}" if orig != final else ""
        flag_str  = f"[{','.join(flags)}]" if flags else "[clean]"
        print(f"\n  {status}  {label}")
        print(f"         KTAS-{orig}{escalated}  conf={conf}  {flag_str}")

        if not ok:
            for f in failures:
                print(f"         ❌ {f}")

        if verbose:
            for note in result.get("safety_notes", []):
                print(f"         › {note}")
            for cr in result.get("conflict_reasons", []):
                print(f"         ⚡ {cr}")

    print(f"\n{'='*70}")
    print(f"  Results: {passed} passed, {failed} failed out of {len(MOCK_CASES)} cases")
    print(f"{'='*70}\n")
    return failed == 0


# =============================================================================
#  REAL MODE — loads model + CSV, runs full pipeline
# =============================================================================

def run_real_tests(model_path: str, csv_path: str,
                   n: int = 200, verbose: bool = False):
    print("\n" + "=" * 70)
    print(f"  REAL DATA TEST MODE")
    print(f"  Model : {model_path}")
    print(f"  CSV   : {csv_path}  |  n={n}")
    print("=" * 70)

    bundle = load_model_bundle(model_path)
    df     = pd.read_csv(csv_path)

    # Sample up to n rows, stratified by acuity if possible
    if len(df) > n:
        try:
            df = df.groupby("triage_acuity", group_keys=False).apply(
                lambda x: x.sample(min(len(x), n // 5), random_state=42)
            ).reset_index(drop=True).head(n)
        except Exception:
            df = df.sample(n, random_state=42).reset_index(drop=True)

    print(f"  Sampled {len(df)} rows\n")

    stats = {
        "total": 0, "correct_raw": 0, "correct_final": 0,
        "rail_hits": 0, "conf_hits": 0, "conflict_hits": 0,
        "escalations": 0, "requires_review": 0,
        "acuity_changes": Counter(),
        "rail_reasons": Counter(),
        "conflict_types": Counter(),
        "conf_states": Counter(),
    }

    for _, row in df.iterrows():
        record = row.to_dict()
        record["patient_id"] = str(record.get("patient_id", f"ROW-{stats['total']}"))

        try:
            result = run_safety_pipeline(bundle, record)
        except Exception as e:
            print(f"  [ERROR] {record['patient_id']}: {e}")
            continue

        stats["total"] += 1
        true_a  = result.get("true_acuity")
        orig_a  = result["original_acuity"]
        final_a = result["predicted_acuity"]

        if true_a == orig_a:  stats["correct_raw"]   += 1
        if true_a == final_a: stats["correct_final"]  += 1
        if result["safety_rail_triggered"]:
            stats["rail_hits"] += 1
            for r in result["safety_rail_reasons"]:
                # Bucket by first word pair
                key = " ".join(r.split()[:3])
                stats["rail_reasons"][key] += 1
        if result["confidence_state"] in ("AMBIGUOUS", "LOW"):
            stats["conf_hits"] += 1
        if result["has_feature_conflict"]:
            stats["conflict_hits"] += 1
            for cr in result["conflict_reasons"]:
                t = cr[:3]  # [V], [C], [H], [N]
                stats["conflict_types"][t] += 1
        if orig_a != final_a:
            stats["escalations"] += 1
            stats["acuity_changes"][f"KTAS-{orig_a}→KTAS-{final_a}"] += 1
        if result["requires_review"]:
            stats["requires_review"] += 1

        stats["conf_states"][result["confidence_state"]] += 1

        if verbose and (result["safety_rail_triggered"] or
                        result["has_feature_conflict"] or
                        result["confidence_state"] in ("AMBIGUOUS", "LOW")):
            flag_str = ",".join(result["intervention_flags"])
            esc      = f" ↑{orig_a}→{final_a}" if orig_a != final_a else ""
            print(f"  [{result['patient_id']}] KTAS-{orig_a}{esc} "
                  f"[{flag_str}] true={true_a}")
            for note in result.get("safety_notes", [])[:2]:
                wrapped = textwrap.fill(note, width=70,
                                        initial_indent="    › ",
                                        subsequent_indent="      ")
                print(wrapped)

    n = stats["total"]
    if n == 0:
        print("  No records processed.")
        return

    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  Records processed     : {n}")
    print(f"  Raw model accuracy    : {stats['correct_raw']/n*100:.2f}%")
    print(f"  Final accuracy        : {stats['correct_final']/n*100:.2f}%")
    delta = (stats['correct_final'] - stats['correct_raw']) / n * 100
    sign  = "+" if delta >= 0 else ""
    print(f"  Accuracy delta        : {sign}{delta:.2f}% from safety layers")

    print(f"\n  Safety layer triggers:")
    print(f"    Deterministic rails : {stats['rail_hits']:>4}  ({stats['rail_hits']/n*100:.1f}%)")
    print(f"    Confidence issues   : {stats['conf_hits']:>4}  ({stats['conf_hits']/n*100:.1f}%)")
    print(f"    Feature conflicts   : {stats['conflict_hits']:>4}  ({stats['conflict_hits']/n*100:.1f}%)")
    print(f"    Total escalations   : {stats['escalations']:>4}  ({stats['escalations']/n*100:.1f}%)")
    print(f"    Flagged for review  : {stats['requires_review']:>4}  ({stats['requires_review']/n*100:.1f}%)")

    print(f"\n  Confidence distribution:")
    for state, count in sorted(stats["conf_states"].items(), key=lambda x: -x[1]):
        bar = "█" * int(count / n * 40)
        print(f"    {state:<12} {count:>5}  {count/n*100:5.1f}%  {bar}")

    if stats["acuity_changes"]:
        print(f"\n  Acuity escalation breakdown:")
        for change, count in stats["acuity_changes"].most_common(10):
            print(f"    {change:<20} {count:>4}")

    if stats["rail_reasons"]:
        print(f"\n  Top rail reasons:")
        for reason, count in stats["rail_reasons"].most_common(8):
            print(f"    {count:>4}×  {reason}")

    if stats["conflict_types"]:
        print(f"\n  Conflict type breakdown:")
        type_labels = {"[V]": "Vital vs acuity", "[C]": "Complaint vs vitals",
                       "[H]": "History vs complaint", "[N]": "NEWS2 vs acuity"}
        for ctype, count in stats["conflict_types"].most_common():
            print(f"    {ctype} {type_labels.get(ctype, ctype):<28} {count:>4}")

    print(f"\n{'='*70}\n")


# =============================================================================
#  CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test the KTAS safety pipeline (rails + confidence + conflicts)"
    )
    parser.add_argument("--mode",    choices=["mock", "real"], default="mock",
                        help="mock = hand-crafted cases, real = CSV + model (default: mock)")
    parser.add_argument("--model",   default="triage_model.pkl",
                        help="Path to triage_model.pkl (real mode only)")
    parser.add_argument("--csv",     default="data.csv",
                        help="Path to data CSV (real mode only)")
    parser.add_argument("--n",       type=int, default=200,
                        help="Rows to sample from CSV (default: 200)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-patient detail for triggered cases")
    args = parser.parse_args()

    if args.mode == "mock":
        ok = run_mock_tests(verbose=args.verbose or True)
        sys.exit(0 if ok else 1)
    else:
        run_real_tests(
            model_path=args.model,
            csv_path=args.csv,
            n=args.n,
            verbose=args.verbose,
        )


if __name__ == "__main__":
    main()