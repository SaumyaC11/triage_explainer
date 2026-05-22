"""
kafka_consumer.py
=================
Consumes patient records from `triage-input`, runs them through the saved
triage model bundle (triage_model.pkl), and publishes prediction results to
`triage-output`.

Three post-prediction safety layers (applied in order):
  1. Deterministic safety rails  — hard vital/keyword overrides, always win
  2. Confidence classification   — HIGH / MODERATE / AMBIGUOUS / LOW
  3. Feature conflict detection  — vitals vs complaint vs history disagreements

Usage:
    pip install kafka-python scikit-learn xgboost numpy pandas scipy
    python kafka_consumer.py
    python kafka_consumer.py --model triage_model.pkl --broker localhost:9092
"""

import json
import argparse
import pickle
import time
import threading
import os
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.sparse import issparse

from kafka import KafkaConsumer, KafkaProducer
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Latency / metrics store — graceful fallback if file not yet in place
try:
    from latency_store import LatencyStore
    HAS_LATENCY_STORE = True
except ImportError:
    HAS_LATENCY_STORE = False
    print("[CONSUMER] ⚠ latency_store not found — metrics will not be recorded")


# Groq enrichment — optional, only fires on escalation rails
try:
    from explainer.triage_explainer import needs_escalation_review, build_prompt, call_groq, parse_llm_response
    HAS_EXPLAINER = True
except ImportError as _import_err:
    HAS_EXPLAINER = False
    print(f"[CONSUMER] ⚠ triage_explainer import FAILED: {_import_err}")
    print(f"[CONSUMER]   sys.path = {sys.path}")
    print(f"[CONSUMER]   LLM enrichment disabled")

# ── KTAS constants ────────────────────────────────────────────────────────────
KTAS_LABELS = {
    1: "KTAS 1 – Resuscitation",
    2: "KTAS 2 – Emergent",
    3: "KTAS 3 – Urgent",
    4: "KTAS 4 – Less Urgent",
    5: "KTAS 5 – Non-Urgent",
}

KTAS_COLORS = {
    1: "#E24B4A",
    2: "#EF9F27",
    3: "#378ADD",
    4: "#639922",
    5: "#888780",
}

HISTORY_COLS = [
    "hx_hypertension", "hx_diabetes_type2", "hx_diabetes_type1",
    "hx_asthma", "hx_copd", "hx_heart_failure", "hx_atrial_fibrillation",
    "hx_ckd", "hx_liver_disease", "hx_malignancy", "hx_obesity",
    "hx_depression", "hx_anxiety", "hx_dementia", "hx_epilepsy",
    "hx_hypothyroidism", "hx_hyperthyroidism", "hx_hiv",
    "hx_coagulopathy", "hx_immunosuppressed", "hx_pregnant",
    "hx_substance_use_disorder", "hx_coronary_artery_disease",
    "hx_stroke_prior", "hx_peripheral_vascular_disease",
]

# ── Confidence thresholds ─────────────────────────────────────────────────────
CONF_HIGH_MIN   = 0.65   # top-1 ≥ 0.65 AND gap ≥ 0.15  → HIGH
CONF_HIGH_GAP   = 0.15
CONF_LOW_MAX    = 0.50   # top-1 < 0.50                  → LOW
CONF_AMB_GAP    = 0.10   # gap < 0.10 (but top-1 ≥ 0.50) → AMBIGUOUS

# ── Critical complaint keywords that force KTAS-1 ─────────────────────────────
CRITICAL_KEYWORDS_KTAS1 = {
    "cardiac arrest", "not breathing", "unresponsive", "pulseless",
    "anaphylactic shock", "anaphylaxis", "status epilepticus",
    "massive haemorrhage", "massive hemorrhage", "choking",
    "respiratory arrest", "hanging", "drowning",
}

# ── High-risk keywords that floor at KTAS-2 ──────────────────────────────────
HIGH_RISK_KEYWORDS_KTAS2 = {
    "chest pain", "stroke", "acute stroke", "altered mental status",
    "severe dyspnoea", "severe dyspnea", "difficulty breathing",
    "severe allergic reaction", "overdose", "drug overdose",
    "septic shock", "sepsis", "meningitis", "aortic dissection",
    "pulmonary embolism", "active seizure", "eclampsia",
}

# ── Benign keywords used in conflict detection ────────────────────────────────
BENIGN_KEYWORDS = {
    "sore throat", "prescription refill", "cold symptoms", "minor rash",
    "routine wound check", "follow-up", "ear pain", "mild headache",
    "sprained ankle", "insect bite", "cold", "flu",
}


# =============================================================================
#  LAYER 0 — MODEL LOADING
# =============================================================================

def load_model_bundle(path: str) -> dict:
    print(f"[CONSUMER] Loading model bundle from: {path}")
    with open(path, "rb") as f:
        bundle = pickle.load(f)
    print(f"[CONSUMER] Model loaded  : {bundle.get('model_name', 'unknown')}")
    print(f"[CONSUMER] Numeric cols  : {len(bundle['numeric_cols'])}")
    print(f"[CONSUMER] Categorical   : {len(bundle['categorical_cols'])}")
    return bundle


# =============================================================================
#  LAYER 0 — FEATURE ENGINEERING  (mirrors triage_system.py)
# =============================================================================

def _add_flag_features(row: dict) -> dict:
    """Compute all derived clinical flag features the model expects."""
    p = dict(row)

    sbp  = p.get("systolic_bp",      np.nan)
    hr   = p.get("heart_rate",       np.nan)
    rr   = p.get("respiratory_rate", np.nan)
    bt   = p.get("temperature_c",    np.nan)
    spo2 = p.get("spo2",             np.nan)
    gcs  = p.get("gcs_total",        np.nan)
    pain = p.get("pain_score",       np.nan)

    def _f(val, cond):
        try:
            return int(cond) if not (isinstance(val, float) and np.isnan(val)) else 0
        except Exception:
            return 0

    p["flag_hypotension"]  = _f(sbp,  sbp  < 90)
    p["flag_hypertension"] = _f(sbp,  sbp  > 180)
    p["flag_tachycardia"]  = _f(hr,   hr   > 100)
    p["flag_bradycardia"]  = _f(hr,   hr   < 60)
    p["flag_tachypnea"]    = _f(rr,   rr   > 20)
    p["flag_bradypnea"]    = _f(rr,   rr   < 12)
    p["flag_fever"]        = _f(bt,   bt   > 38.3)
    p["flag_hypothermia"]  = _f(bt,   bt   < 35.0)
    p["flag_hypoxia"]      = _f(spo2, spo2 < 94)
    p["flag_severe_pain"]  = _f(pain, pain >= 7)
    p["flag_low_gcs"]      = _f(gcs,  gcs  < 14)
    p["flag_critical_gcs"] = _f(gcs,  gcs  <= 8)

    hx_vals = [int(p.get(c, 0) or 0) for c in HISTORY_COLS]
    p["hx_total_burden"]  = sum(hx_vals)
    high_risk = ["hx_heart_failure", "hx_copd", "hx_malignancy",
                 "hx_ckd", "hx_dementia", "hx_coagulopathy", "hx_immunosuppressed"]
    p["hx_high_risk_flag"] = int(any(int(p.get(c, 0) or 0) for c in high_risk))
    return p


# =============================================================================
#  LAYER 0 — RAW PREDICTION
# =============================================================================

def combine_features(X_structured, X_tfidf):
    if issparse(X_structured):
        X_structured = X_structured.toarray()
    return np.hstack([X_structured, X_tfidf.toarray()])


def predict_patient(bundle: dict, raw_record: dict) -> dict:
    """Run the ML model and return a base prediction dict (no safety layers yet)."""
    model            = bundle["model"]
    preprocessor     = bundle["preprocessor"]
    tfidf            = bundle["tfidf"]
    numeric_cols     = bundle["numeric_cols"]
    categorical_cols = bundle["categorical_cols"]
    is_xgb           = bundle.get("is_xgb", False)
    le               = bundle.get("le")

    patient   = _add_flag_features(raw_record)
    struct_row = {col: [patient.get(col, np.nan)]
                  for col in numeric_cols + categorical_cols}
    df_struct  = pd.DataFrame(struct_row)
    X_struct   = preprocessor.transform(df_struct)
    raw_text   = str(patient.get("chief_complaint_raw", ""))
    X_tfidf    = tfidf.transform([raw_text])
    X_combined = combine_features(X_struct, X_tfidf)
    proba      = model.predict_proba(X_combined)[0]

    classes = (le.inverse_transform(np.arange(len(proba)))
               if is_xgb and le is not None
               else model.classes_)

    pred_idx   = int(np.argmax(proba))
    pred_class = int(classes[pred_idx])
    confidence = float(proba[pred_idx])

    probabilities = {
        KTAS_LABELS.get(int(c), str(c)): round(float(p), 4)
        for c, p in zip(classes, proba)
    }

    # Store raw proba list + classes for downstream layers
    proba_by_class = {int(c): float(p) for c, p in zip(classes, proba)}

    return {
        "patient_id":           raw_record.get("patient_id", "UNKNOWN"),
        "timestamp":            raw_record.get("timestamp", datetime.now().isoformat()),
        "predicted_acuity":     pred_class,
        "original_acuity":      pred_class,   # immutable copy for audit trail
        "true_acuity":          raw_record.get("true_acuity"),
        "label":                KTAS_LABELS.get(pred_class, str(pred_class)),
        "color":                KTAS_COLORS.get(pred_class, "#888888"),
        "confidence":           round(confidence, 4),
        "confidence_pct":       f"{confidence * 100:.1f}%",
        "confidence_state":     "UNKNOWN",    # filled by layer 2
        "is_emergency":         pred_class <= 2,
        "probabilities":        probabilities,
        "_proba_by_class":      proba_by_class,   # internal, stripped before publish
        "chief_complaint":      raw_record.get("chief_complaint_raw", ""),
        "vitals": {
            "heart_rate":       raw_record.get("heart_rate"),
            "systolic_bp":      raw_record.get("systolic_bp"),
            "diastolic_bp":     raw_record.get("diastolic_bp"),
            "spo2":             raw_record.get("spo2"),
            "respiratory_rate": raw_record.get("respiratory_rate"),
            "temperature_c":    raw_record.get("temperature_c"),
            "gcs_total":        raw_record.get("gcs_total"),
            "pain_score":       raw_record.get("pain_score"),
            "news2_score":      raw_record.get("news2_score"),
        },
        "age":   raw_record.get("age"),
        "sex":   raw_record.get("sex"),
        "model": bundle.get("model_name", "unknown"),
        # Safety-layer fields (all False/empty until layers run)
        "safety_rail_triggered":  False,
        "safety_rail_reasons":    [],
        "has_feature_conflict":   False,
        "conflict_reasons":       [],
        "requires_review":        False,
        "safety_notes":           [],
        "processed_at":           datetime.now().isoformat(),
    }


# =============================================================================
#  LAYER 1 — DETERMINISTIC SAFETY RAILS
# =============================================================================
#  Rules are ordered from most to least critical.
#  Each rule that fires appends a reason and may raise the acuity floor.
#  No rule can ever LOWER the predicted acuity.
# =============================================================================

def _safe_float(val):
    """Return float or None — never raises."""
    try:
        v = float(val)
        return None if np.isnan(v) else v
    except (TypeError, ValueError):
        return None


def apply_safety_rails(result: dict, raw_record: dict) -> dict:
    """
    Layer 1: deterministic, rule-based acuity floor.
    Operates on result['vitals'] and raw complaint text.
    Raises predicted_acuity if any rule fires; never lowers it.
    """
    vitals    = result.get("vitals", {})
    complaint = str(raw_record.get("chief_complaint_raw", "")).lower().strip()
    age       = _safe_float(raw_record.get("age"))

    sbp  = _safe_float(vitals.get("systolic_bp"))
    dbp  = _safe_float(vitals.get("diastolic_bp"))
    hr   = _safe_float(vitals.get("heart_rate"))
    rr   = _safe_float(vitals.get("respiratory_rate"))
    spo2 = _safe_float(vitals.get("spo2"))
    temp = _safe_float(vitals.get("temperature_c"))
    gcs  = _safe_float(vitals.get("gcs_total"))
    pain = _safe_float(vitals.get("pain_score"))
    news2= _safe_float(vitals.get("news2_score"))

    acuity  = result["predicted_acuity"]
    reasons = []

    # ── A. Absolute KTAS-1 floors (life-threatening, non-negotiable) ──────────
    if gcs is not None and gcs <= 8:
        reasons.append(f"GCS {gcs:.0f} ≤ 8 → KTAS-1 floor")
        acuity = min(acuity, 1)

    if spo2 is not None and spo2 < 88:
        reasons.append(f"SpO2 {spo2:.1f}% < 88% → KTAS-1 floor")
        acuity = min(acuity, 1)

    if sbp is not None and sbp < 70:
        reasons.append(f"SBP {sbp:.0f} mmHg < 70 → KTAS-1 floor")
        acuity = min(acuity, 1)

    if hr is not None and hr > 180:
        reasons.append(f"HR {hr:.0f} bpm > 180 → KTAS-1 floor")
        acuity = min(acuity, 1)

    if hr is not None and hr < 30:
        reasons.append(f"HR {hr:.0f} bpm < 30 → KTAS-1 floor")
        acuity = min(acuity, 1)

    if rr is not None and rr > 36:
        reasons.append(f"RR {rr:.0f}/min > 36 → KTAS-1 floor")
        acuity = min(acuity, 1)

    if temp is not None and temp < 32.0:
        reasons.append(f"Temp {temp:.1f}°C < 32 → KTAS-1 floor (severe hypothermia)")
        acuity = min(acuity, 1)

    # Critical keyword match → KTAS-1 floor
    for kw in CRITICAL_KEYWORDS_KTAS1:
        if kw in complaint:
            reasons.append(f"Complaint keyword '{kw}' → KTAS-1 floor")
            acuity = min(acuity, 1)
            break

    # ── B. KTAS-2 floors (emergent, high-risk) ────────────────────────────────
    if spo2 is not None and 88 <= spo2 < 92 and acuity > 2:
        reasons.append(f"SpO2 {spo2:.1f}% (88–92%) → KTAS-2 floor")
        acuity = min(acuity, 2)

    if sbp is not None and 70 <= sbp < 90 and acuity > 2:
        reasons.append(f"SBP {sbp:.0f} mmHg (70–90) → KTAS-2 floor")
        acuity = min(acuity, 2)

    if hr is not None and (150 < hr <= 180) and acuity > 2:
        reasons.append(f"HR {hr:.0f} bpm (150–180) → KTAS-2 floor")
        acuity = min(acuity, 2)

    if hr is not None and (30 <= hr < 40) and acuity > 2:
        reasons.append(f"HR {hr:.0f} bpm (30–40, severe bradycardia) → KTAS-2 floor")
        acuity = min(acuity, 2)

    if gcs is not None and 9 <= gcs <= 12 and acuity > 2:
        reasons.append(f"GCS {gcs:.0f} (9–12) → KTAS-2 floor")
        acuity = min(acuity, 2)

    if temp is not None and temp > 40.0 and acuity > 2:
        reasons.append(f"Temp {temp:.1f}°C > 40 → KTAS-2 floor (hyperpyrexia)")
        acuity = min(acuity, 2)

    if news2 is not None and news2 >= 7 and acuity > 2:
        reasons.append(f"NEWS2 {news2:.0f} ≥ 7 → KTAS-2 floor (high risk)")
        acuity = min(acuity, 2)

    # High-risk keyword → KTAS-2 floor
    for kw in HIGH_RISK_KEYWORDS_KTAS2:
        if kw in complaint and acuity > 2:
            reasons.append(f"Complaint keyword '{kw}' → KTAS-2 floor")
            acuity = min(acuity, 2)
            break

    # ── C. Combination rules (no single vital critical, but together dangerous) ─
    # Sepsis triad: fever + tachycardia + tachypnea
    if (temp is not None and temp > 38.3 and
        hr   is not None and hr   > 100   and
        rr   is not None and rr   > 20    and
        acuity > 2):
        reasons.append(
            f"Sepsis triad (temp {temp:.1f}°C, HR {hr:.0f}, RR {rr:.0f}) → KTAS-2 floor"
        )
        acuity = min(acuity, 2)

    # Shock index > 1.5 (HR/SBP) — haemodynamic instability
    if sbp is not None and hr is not None and sbp > 0:
        shock_idx = hr / sbp
        if shock_idx > 1.5 and acuity > 2:
            reasons.append(
                f"Shock index {shock_idx:.2f} > 1.5 (HR/SBP) → KTAS-2 floor"
            )
            acuity = min(acuity, 2)

    # Elderly (≥75) + confusion + fever → KTAS-2 (sepsis/delirium risk)
    mental = str(raw_record.get("mental_status_triage", "")).lower()
    if (age is not None and age >= 75 and
        mental not in ("", "alert") and
        temp is not None and temp > 38.0 and
        acuity > 2):
        reasons.append(
            f"Elderly ({age:.0f}y) + altered mental status + fever → KTAS-2 floor"
        )
        acuity = min(acuity, 2)

    # Hypoxia + hypotension together
    if (spo2 is not None and spo2 < 94 and
        sbp  is not None and sbp  < 90 and
        acuity > 1):
        reasons.append(
            f"Combined hypoxia (SpO2 {spo2:.1f}%) + hypotension (SBP {sbp:.0f}) → KTAS-1 floor"
        )
        acuity = min(acuity, 1)

    # ── D. Apply and annotate ─────────────────────────────────────────────────
    if reasons:
        original = result["predicted_acuity"]
        result["predicted_acuity"]       = acuity
        result["safety_rail_triggered"]  = True
        result["safety_rail_reasons"]    = reasons
        result["is_emergency"]           = acuity <= 2
        result["label"]                  = KTAS_LABELS.get(acuity, str(acuity))
        result["color"]                  = KTAS_COLORS.get(acuity, "#888888")
        if original != acuity:
            result["safety_notes"].append(
                f"Safety rails escalated KTAS-{original} → KTAS-{acuity}: "
                + "; ".join(reasons)
            )
        else:
            result["safety_notes"].append(
                f"Safety rails confirmed KTAS-{acuity} (no escalation needed): "
                + "; ".join(reasons)
            )

    return result


# =============================================================================
#  LAYER 2 — CONFIDENCE CLASSIFICATION & RESOLUTION
# =============================================================================

def classify_and_resolve_confidence(result: dict) -> dict:
    """
    Layer 2: classify prediction certainty, then apply conservative resolution.

    States:
      HIGH      — top-1 ≥ 0.65 and gap ≥ 0.15  →  trust as-is
      MODERATE  — top-1 ≥ 0.50 and gap ≥ 0.10  →  trust as-is
      AMBIGUOUS — gap < 0.10 (close race)       →  escalate to more severe class
      LOW       — top-1 < 0.50                  →  escalate by 1 + flag review
    """
    proba_by_class = result.get("_proba_by_class", {})
    if not proba_by_class:
        result["confidence_state"] = "UNKNOWN"
        return result

    sorted_pairs = sorted(proba_by_class.items(), key=lambda x: -x[1])
    top1_class, top1_score = sorted_pairs[0]
    top2_class, top2_score = sorted_pairs[1] if len(sorted_pairs) > 1 else (top1_class, 0.0)
    gap = top1_score - top2_score

    # Classify
    if top1_score >= CONF_HIGH_MIN and gap >= CONF_HIGH_GAP:
        state = "HIGH"
    elif top1_score >= CONF_LOW_MAX and gap >= CONF_AMB_GAP:
        state = "MODERATE"
    elif top1_score >= CONF_LOW_MAX and gap < CONF_AMB_GAP:
        state = "AMBIGUOUS"
    else:
        state = "LOW"

    result["confidence_state"] = state
    result["confidence_gap"]   = round(gap, 4)
    result["confidence_top2_acuity"] = int(top2_class)
    result["confidence_top2_pct"]    = f"{top2_score * 100:.1f}%"

    current_acuity = result["predicted_acuity"]

    if state == "AMBIGUOUS":
        # Both classes are plausible — take the more severe (lower number) one
        conservative = min(int(top1_class), int(top2_class))
        if conservative < current_acuity:
            result["safety_notes"].append(
                f"AMBIGUOUS: KTAS-{top1_class} ({top1_score*100:.1f}%) vs "
                f"KTAS-{top2_class} ({top2_score*100:.1f}%), gap={gap:.3f} — "
                f"escalated to KTAS-{conservative}"
            )
            result["predicted_acuity"] = conservative
            result["is_emergency"]     = conservative <= 2
            result["label"]            = KTAS_LABELS.get(conservative, str(conservative))
            result["color"]            = KTAS_COLORS.get(conservative, "#888888")
        else:
            result["safety_notes"].append(
                f"AMBIGUOUS: gap={gap:.3f} between KTAS-{top1_class} and "
                f"KTAS-{top2_class} — conservative choice already active"
            )
        result["requires_review"] = True

    elif state == "LOW":
        # Model is genuinely unsure — escalate one level and flag for human review
        escalated = max(1, current_acuity - 1)
        result["safety_notes"].append(
            f"LOW confidence ({top1_score*100:.1f}%) — "
            f"escalated KTAS-{current_acuity} → KTAS-{escalated}, flagged for review"
        )
        result["predicted_acuity"] = escalated
        result["is_emergency"]     = escalated <= 2
        result["label"]            = KTAS_LABELS.get(escalated, str(escalated))
        result["color"]            = KTAS_COLORS.get(escalated, "#888888")
        result["requires_review"]  = True

    elif state in ("HIGH", "MODERATE"):
        result["safety_notes"].append(
            f"{state} confidence ({top1_score*100:.1f}%, gap={gap:.3f}) — prediction stands"
        )

    return result


# =============================================================================
#  LAYER 3 — FEATURE CONFLICT DETECTION
# =============================================================================

def detect_feature_conflicts(result: dict, raw_record: dict) -> dict:
    """
    Layer 3: detect disagreements between feature groups.
    When vitals contradict the complaint or history, flag the conflict
    and conservatively escalate if the signals suggest a more severe acuity.

    Conflict types detected:
      V  — vital sign contradicts predicted acuity
      C  — chief complaint contradicts vitals (benign complaint + critical vitals)
      H  — high-risk history + mild complaint (masking risk)
      N  — NEWS2 score contradicts predicted acuity
    """
    vitals    = result.get("vitals", {})
    pred        = result.get("original_acuity", result["predicted_acuity"])
    complaint = str(raw_record.get("chief_complaint_raw", "")).lower().strip()
    conflicts = []
    escalate_to = result["predicted_acuity"]   # may tighten below

    spo2 = _safe_float(vitals.get("spo2"))
    sbp  = _safe_float(vitals.get("systolic_bp"))
    hr   = _safe_float(vitals.get("heart_rate"))
    rr   = _safe_float(vitals.get("respiratory_rate"))
    gcs  = _safe_float(vitals.get("gcs_total"))
    temp = _safe_float(vitals.get("temperature_c"))
    pain = _safe_float(vitals.get("pain_score"))
    news2= _safe_float(vitals.get("news2_score"))
    age  = _safe_float(raw_record.get("age"))

    # ── V: vital-vs-acuity conflicts ─────────────────────────────────────────
    if spo2 is not None and spo2 < 92 and pred >= 3:
        conflicts.append(
            f"[V] SpO2={spo2:.1f}% indicates ≤KTAS-2 but model → KTAS-{pred}"
        )
        escalate_to = min(escalate_to, 2)

    if gcs is not None and gcs < 13 and pred >= 3:
        conflicts.append(
            f"[V] GCS={gcs:.0f} indicates ≤KTAS-2 but model → KTAS-{pred}"
        )
        escalate_to = min(escalate_to, 2)

    if sbp is not None and sbp < 90 and pred >= 3:
        conflicts.append(
            f"[V] SBP={sbp:.0f}mmHg indicates ≤KTAS-2 but model → KTAS-{pred}"
        )
        escalate_to = min(escalate_to, 2)

    if hr is not None and (hr > 140 or hr < 45) and pred >= 4:
        conflicts.append(
            f"[V] HR={hr:.0f}bpm is extreme but model → KTAS-{pred}"
        )
        escalate_to = min(escalate_to, 3)

    if rr is not None and (rr > 28 or rr < 10) and pred >= 4:
        conflicts.append(
            f"[V] RR={rr:.0f}/min is extreme but model → KTAS-{pred}"
        )
        escalate_to = min(escalate_to, 3)

    if temp is not None and (temp > 39.5 or temp < 35.5) and pred >= 4:
        conflicts.append(
            f"[V] Temp={temp:.1f}°C is abnormal but model → KTAS-{pred}"
        )
        escalate_to = min(escalate_to, 3)

    if pain is not None and pain >= 8 and pred >= 4:
        conflicts.append(
            f"[V] Pain={pain:.0f}/10 (severe) but model → KTAS-{pred}"
        )
        escalate_to = min(escalate_to, 3)

    # ── N: NEWS2 vs acuity conflict ───────────────────────────────────────────
    if news2 is not None:
        if news2 >= 5 and pred >= 4:
            conflicts.append(
                f"[N] NEWS2={news2:.0f} (medium-high risk) but model → KTAS-{pred}"
            )
            escalate_to = min(escalate_to, 3)
        if news2 >= 7 and pred >= 3:
            conflicts.append(
                f"[N] NEWS2={news2:.0f} (high risk) but model → KTAS-{pred}"
            )
            escalate_to = min(escalate_to, 2)

    # ── C: complaint-vs-vitals conflict ──────────────────────────────────────
    is_benign_complaint = any(kw in complaint for kw in BENIGN_KEYWORDS)
    critical_vitals     = (
        (spo2 is not None and spo2 < 92) or
        (gcs  is not None and gcs  < 14) or
        (sbp  is not None and sbp  < 90) or
        (hr   is not None and (hr > 150 or hr < 40))
    )

    if is_benign_complaint and critical_vitals:
        matched_kw = next((kw for kw in BENIGN_KEYWORDS if kw in complaint), "")
        conflicts.append(
            f"[C] Complaint '{matched_kw}' appears benign "
            f"but vitals indicate critical state — possible masking"
        )
        escalate_to = min(escalate_to, 2)

    # ── H: high-risk history + mild prediction conflict ───────────────────────
    high_risk_hx = [
        "hx_heart_failure", "hx_copd", "hx_malignancy",
        "hx_ckd", "hx_coagulopathy", "hx_immunosuppressed", "hx_dementia"
    ]
    active_high_risk = [c for c in high_risk_hx if int(raw_record.get(c, 0) or 0)]

    if active_high_risk and is_benign_complaint and pred >= 4:
        risk_names = [c.replace("hx_", "").replace("_", " ") for c in active_high_risk]
        conflicts.append(
            f"[H] High-risk comorbidities ({', '.join(risk_names)}) "
            f"+ benign complaint — raised baseline risk, model → KTAS-{pred}"
        )
        escalate_to = min(escalate_to, 3)

    # ── Apply conflicts ───────────────────────────────────────────────────────
    if conflicts:
        result["has_feature_conflict"] = True
        result["conflict_reasons"]     = conflicts

        if escalate_to < pred:
            result["safety_notes"].append(
                f"CONFLICT escalated KTAS-{pred} → KTAS-{escalate_to}: "
                + " | ".join(conflicts)
            )
            result["predicted_acuity"] = escalate_to
            result["is_emergency"]     = escalate_to <= 2
            result["label"]            = KTAS_LABELS.get(escalate_to, str(escalate_to))
            result["color"]            = KTAS_COLORS.get(escalate_to, "#888888")
            result["requires_review"]  = True
        else:
            result["safety_notes"].append(
                "CONFLICT detected but acuity already at correct level: "
                + " | ".join(conflicts)
            )

    return result


# =============================================================================
#  PIPELINE ORCHESTRATOR
# =============================================================================

def run_safety_pipeline(bundle: dict, raw_record: dict) -> tuple:
    """
    Full pipeline:
      predict_patient()  ->  safety rails  ->  confidence  ->  conflict detection

    Returns (result, timing) where timing is a dict of epoch floats:
        ml_done_at      -- after predict_patient() completes
        safety_done_at  -- after all three safety layers complete

    The caller stamps consumer_received_at just before calling this function.
    """
    # Raw ML prediction -- stamp after inference completes
    result     = predict_patient(bundle, raw_record)
    ml_done_at = time.time()

    # Layer 1 -- deterministic safety rails (runs first, sets hard floors)
    result = apply_safety_rails(result, raw_record)

    # Layer 2 -- confidence classification and resolution
    result = classify_and_resolve_confidence(result)

    # Layer 3 -- feature conflict detection
    result = detect_feature_conflicts(result, raw_record)

    safety_done_at = time.time()

    # Sync final label/color after all layers
    final_acuity           = result["predicted_acuity"]
    result["label"]        = KTAS_LABELS.get(final_acuity, str(final_acuity))
    result["color"]        = KTAS_COLORS.get(final_acuity, "#888888")
    result["is_emergency"] = final_acuity <= 2

    # Build a human-readable summary of all interventions
    interventions = []
    if result["safety_rail_triggered"]:
        interventions.append("RAIL")
    if result["confidence_state"] in ("AMBIGUOUS", "LOW"):
        interventions.append(result["confidence_state"])
    if result["has_feature_conflict"]:
        interventions.append("CONFLICT")
    result["intervention_flags"] = interventions

    # Strip internal working fields before publishing
    result.pop("_proba_by_class", None)

    timing = {
        "ml_done_at":     ml_done_at,
        "safety_done_at": safety_done_at,
    }
    return result, timing


# =============================================================================
#  KAFKA WIRING
# =============================================================================

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


def build_producer(broker: str) -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=[broker],
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k else None,
    )


# =============================================================================
#  GROQ ENRICHMENT  (async, fires AFTER raw result is already published)
# =============================================================================

def _enrich_and_publish(
    result:          dict,
    raw_record:      dict,
    producer:        KafkaProducer,
    enrich_topic:    str,
    groq_model:      str,
    groq_api_key:    str,
):
    import traceback as _tb
    pid = result.get("patient_id", "UNKNOWN")
    print(f"  [ENRICH:{pid}] ── thread started ──")

    # ── Merge history fields ──────────────────────────────────────────────────
    enriched_record = dict(result)
    hx_merged = []
    for k, v in raw_record.items():
        if k.startswith("hx_") or k in (
            "medical_history", "past_medical_history",
            "arrival_mode", "arrival_transport",
            "mental_status", "mentation",
            "injury_type", "mechanism_of_injury",
        ):
            enriched_record.setdefault(k, v)
            if v:
                hx_merged.append(k)
    print(f"  [ENRICH:{pid}] history fields merged: {hx_merged or '(none)'}")

    # ── Escalation gate ───────────────────────────────────────────────────────
    try:
        esc = needs_escalation_review(enriched_record)
    except Exception as exc:
        print(f"  [ENRICH:{pid}] ERROR in needs_escalation_review: {exc}")
        _tb.print_exc()
        return

    print(f"  [ENRICH:{pid}] gate → should_call_llm={esc.should_call_llm}  "
          f"code={esc.reason_code}  label={esc.reason_label}")

    if not esc.should_call_llm:
        print(f"  [ENRICH:{pid}] routine patient — skipping Groq")
        return

    # ── Build prompt ──────────────────────────────────────────────────────────
    try:
        prompt = build_prompt(enriched_record, esc)
        print(f"  [ENRICH:{pid}] prompt built ({len(prompt)} chars)")
    except Exception as exc:
        print(f"  [ENRICH:{pid}] ERROR building prompt: {exc}")
        _tb.print_exc()
        return

    # ── Call Groq ─────────────────────────────────────────────────────────────
    print(f"  [ENRICH:{pid}] calling Groq  model={groq_model!r}  "
          f"key={'SET (' + groq_api_key[:12] + '...)' if groq_api_key else 'MISSING ❌'}")
    t0 = time.time()
    try:
        raw_llm = call_groq(prompt, groq_model, groq_api_key)
        elapsed = time.time() - t0
        print(f"  [ENRICH:{pid}] Groq responded in {elapsed:.2f}s  "
              f"({len(raw_llm)} chars)  preview: {raw_llm[:100]!r}")
    except Exception as exc:
        print(f"  [ENRICH:{pid}] ERROR calling Groq: {exc}")
        _tb.print_exc()
        return

    if raw_llm.startswith("GROQ_"):
        print(f"  [ENRICH:{pid}] Groq returned error string: {raw_llm}")

    # ── Parse response ────────────────────────────────────────────────────────
    try:
        parsed = parse_llm_response(raw_llm, pid)
        print(f"  [ENRICH:{pid}] parsed  is_ready={parsed.is_ready}  "
              f"reason_chars={len(parsed.escalation_reason)}  "
              f"flags={len(parsed.risk_flags)}  error={parsed.error!r}")
    except Exception as exc:
        print(f"  [ENRICH:{pid}] ERROR parsing response: {exc}")
        _tb.print_exc()
        return

    # ── Publish enrichment message ────────────────────────────────────────────
    enrichment_msg = {
        "patient_id": pid,
        "enrichment": True,
        "llm_explanation": {
            "escalation_reason": parsed.escalation_reason,
            "patient_summary":   parsed.patient_summary,
            "risk_flags":        parsed.risk_flags,
            "confidence_note":   parsed.confidence_note,
            "full_text":         parsed.full_text,
            "escalation_code":   esc.reason_code,
            "escalation_label":  esc.reason_label,
            "escalation_detail": esc.reason_detail,
            "is_ready":          parsed.is_ready,
            "error":             parsed.error,
        },
    }

    try:
        future = producer.send(enrich_topic, key=pid, value=enrichment_msg)
        producer.flush()
        meta = future.get(timeout=10)
        print(f"  [ENRICH:{pid}] ✓ published → topic='{enrich_topic}' "
              f"partition={meta.partition} offset={meta.offset}")
    except Exception as exc:
        print(f"  [ENRICH:{pid}] ERROR publishing to Kafka: {exc}")
        _tb.print_exc()

    print(f"  [ENRICH:{pid}] ── thread done ──")


# =============================================================================
#  MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="KTAS Triage Kafka Consumer + Predictor")
    parser.add_argument("--broker",         default="localhost:9092")
    parser.add_argument("--in-topic",       default="triage-input")
    parser.add_argument("--out-topic",      default="triage-output")
    parser.add_argument("--enrich-topic",   default="triage-enrichment",
                        help="Topic for async LLM enrichment messages (default: triage-enrichment)")
    parser.add_argument("--group",          default="triage-consumer-group")
    parser.add_argument("--model",          default="triage_model.pkl")
    parser.add_argument("--groq-model",     default="llama-3.1-8b-instant",
                        help="Groq model for LLM enrichment (default: llama-3.1-8b-instant)")
    parser.add_argument("--groq-api-key",   default="",
                        help="Groq API key (or set GROQ_API_KEY env var)")
    parser.add_argument("--no-enrich",      action="store_true",
                        help="Disable Groq enrichment entirely")
    parser.add_argument("--db",             default="triage_latency.db",
                        help="Path to SQLite latency/metrics DB (default: triage_latency.db)")
    args = parser.parse_args()

    # Resolve Groq API key: CLI > env var
    groq_api_key = args.groq_api_key or os.environ.get("GROQ_API_KEY", "")
    groq_enabled = HAS_EXPLAINER and not args.no_enrich and bool(groq_api_key)

    bundle   = load_model_bundle(args.model)
    consumer = build_consumer(args.broker, args.in_topic, args.group)
    producer = build_producer(args.broker)

    # Latency store — opened once, shared across all messages in this process
    lat_store = LatencyStore(args.db) if HAS_LATENCY_STORE else None
    if lat_store:
        print(f"[CONSUMER] Metrics DB   : {args.db}")

    icons      = {1: "🔴", 2: "🟠", 3: "🔵", 4: "🟢", 5: "⚪"}
    conf_icons = {"HIGH": "✓", "MODERATE": "~", "AMBIGUOUS": "⚠", "LOW": "!", "UNKNOWN": "?"}

    print(f"\n[CONSUMER] Listening on '{args.in_topic}' → publishing to '{args.out_topic}'")
    print(f"[CONSUMER] ── LLM enrichment diagnostics ──")
    print(f"[CONSUMER]   HAS_EXPLAINER : {HAS_EXPLAINER}")
    print(f"[CONSUMER]   GROQ_API_KEY  : {'SET (' + groq_api_key[:12] + '...)' if groq_api_key else 'NOT SET ❌  ← run:  $env:GROQ_API_KEY=\"gsk_...\"'}")
    print(f"[CONSUMER]   --no-enrich   : {args.no_enrich}")
    print(f"[CONSUMER]   groq_enabled  : {groq_enabled}")
    if groq_enabled:
        print(f"[CONSUMER]   enrich topic  : '{args.enrich_topic}'")
        print(f"[CONSUMER]   groq model    : {args.groq_model}")
        print(f"[CONSUMER]   rails         : KTAS 1/2 · conflict · critical vitals · override")
    else:
        if not HAS_EXPLAINER:
            print(f"[CONSUMER]   ❌ REASON: triage_explainer import failed (see error above)")
        elif args.no_enrich:
            print(f"[CONSUMER]   ❌ REASON: --no-enrich flag is set")
        elif not groq_api_key:
            print(f"[CONSUMER]   ❌ REASON: GROQ_API_KEY not set in this shell session")
            print(f"[CONSUMER]   FIX (PowerShell): $env:GROQ_API_KEY = \"gsk_...\"")
            print(f"[CONSUMER]   FIX (CLI arg):    --groq-api-key gsk_...")
    print(f"[CONSUMER] ───────────────────────────────")
    print(f"[CONSUMER] Safety layers: deterministic rails + confidence + conflict detection")
    print("-" * 80)

    processed = correct = rail_hits = conf_hits = conflict_hits = enrich_fired = 0

    try:
        while True:
            for msg in consumer:
                record = msg.value
                try:
                    consumer_received_at = time.time()       # stamp before pipeline
                    result, timing = run_safety_pipeline(bundle, record)

                    # ── Step 1: publish raw result immediately ─────────────────
                    producer.send(args.out_topic,
                                  key=result["patient_id"],
                                  value=result)
                    producer.flush()
                    processed += 1

                    # ── Latency store: write one row per message ───────────────
                    if lat_store:
                        run_id = record.get("run_id") or "unknown"
                        try:
                            lat_store.write_message(
                                run_id               = run_id,
                                patient_id           = result["patient_id"],
                                produced_at          = record.get("produced_at", consumer_received_at),
                                consumer_received_at = consumer_received_at,
                                ml_done_at           = timing["ml_done_at"],
                                safety_done_at       = timing["safety_done_at"],
                                safety_rail_triggered= result.get("safety_rail_triggered", False),
                                true_acuity          = result.get("true_acuity"),
                                predicted_acuity     = result.get("predicted_acuity"),
                            )
                        except Exception as _lat_exc:
                            print(f"  [METRICS] write_message failed: {_lat_exc}")

                    # ── Step 2: fire Groq enrichment in background thread ──────
                    if groq_enabled:
                        esc_check = needs_escalation_review(result) if HAS_EXPLAINER else None
                        if esc_check:
                            print(f"  [GATE] {result['patient_id']} → "
                                  f"should_call_llm={esc_check.should_call_llm}  "
                                  f"code={esc_check.reason_code}  "
                                  f"(KTAS={result['predicted_acuity']}  "
                                  f"true={result.get('true_acuity')}  "
                                  f"rail={result.get('safety_rail_triggered')}  "
                                  f"conflict={result.get('has_feature_conflict')})")
                        if esc_check and esc_check.should_call_llm:
                            enrich_fired += 1
                            print(f"  [GATE] → spawning enrich thread #{enrich_fired}")
                            threading.Thread(
                                target=_enrich_and_publish,
                                args=(
                                    result, record,
                                    producer, args.enrich_topic,
                                    args.groq_model, groq_api_key,
                                ),
                                daemon=True,
                            ).start()

                    true_acuity  = result.get("true_acuity")
                    pred_acuity  = result["predicted_acuity"]
                    orig_acuity  = result["original_acuity"]
                    conf_state   = result["confidence_state"]
                    flags        = result["intervention_flags"]

                    match = "✓" if true_acuity == pred_acuity else "✗"
                    if true_acuity == pred_acuity:
                        correct += 1
                    if result["safety_rail_triggered"]:
                        rail_hits += 1
                    if conf_state in ("AMBIGUOUS", "LOW"):
                        conf_hits += 1
                    if result["has_feature_conflict"]:
                        conflict_hits += 1

                    flag_str  = f"[{','.join(flags)}]" if flags else ""
                    escalated = f" ↑KTAS-{orig_acuity}→{pred_acuity}" if orig_acuity != pred_acuity else ""
                    enrich_str = f" [LLM→]" if (groq_enabled and esc_check and esc_check.should_call_llm) else ""

                    print(
                        f"  {icons.get(pred_acuity,'?')} [{processed:>4}] "
                        f"Patient {result['patient_id']} | "
                        f"Pred: KTAS-{pred_acuity}{escalated}  True: KTAS-{true_acuity}  {match} | "
                        f"Conf: {result['confidence_pct']} [{conf_icons.get(conf_state,'?')}{conf_state}] "
                        f"{flag_str}{enrich_str} | "
                        f"Acc: {correct/processed*100:.1f}%"
                    )

                    # Print detail lines for any triggered layers
                    for note in result.get("safety_notes", []):
                        print(f"         › {note}")

                except Exception as exc:
                    import traceback
                    print(f"  [ERROR] Patient {record.get('patient_id','?')}: {exc}")
                    traceback.print_exc()

            time.sleep(0.1)

    except KeyboardInterrupt:
        print(f"\n[CONSUMER] Stopped after {processed} records.")
        if processed > 0:
            print(f"  Accuracy        : {correct/processed*100:.2f}%  ({correct}/{processed})")
            print(f"  Safety rail hits: {rail_hits}  ({rail_hits/processed*100:.1f}%)")
            print(f"  Conf hits       : {conf_hits}  ({conf_hits/processed*100:.1f}%)")
            print(f"  Conflict hits   : {conflict_hits}  ({conflict_hits/processed*100:.1f}%)")
            if groq_enabled:
                print(f"  Enrich fired    : {enrich_fired}  ({enrich_fired/processed*100:.1f}% of patients)")
        # Finalize all runs seen in this session
        if lat_store and processed > 0:
            seen_runs = set()
            try:
                import sqlite3
                conn = sqlite3.connect(args.db)
                rows = conn.execute("SELECT DISTINCT run_id FROM messages").fetchall()
                seen_runs = {r[0] for r in rows}
                conn.close()
            except Exception:
                pass
            for rid in seen_runs:
                try:
                    lat_store.finalize_run(rid)
                    print(f"[CONSUMER] Finalized run: {rid}")
                except Exception as _fe:
                    print(f"[CONSUMER] finalize_run failed for {rid}: {_fe}")
    finally:
        consumer.close()
        producer.close()


if __name__ == "__main__":
    main()