"""
KTAS Emergency Triage Classification System
============================================
Dataset columns (exact names as they appear in data.csv):

    patient_id, site_id, triage_nurse_id, arrival_mode,
    arrival_hour, arrival_day, arrival_month, arrival_season,
    shift, age, age_group, sex, language, insurance_type,
    transport_origin, pain_location, mental_status_triage,
    chief_complaint_system, num_prior_ed_visits_12m,
    num_prior_admissions_12m, num_active_medications,
    num_comorbidities, systolic_bp, diastolic_bp,
    mean_arterial_pressure, pulse_pressure, heart_rate,
    respiratory_rate, temperature_c, spo2, gcs_total,
    pain_score, weight_kg, height_cm, bmi, shock_index,
    news2_score, disposition, ed_los_hours, triage_acuity  <- TARGET

Usage:
    pip install pandas scikit-learn xgboost matplotlib seaborn

    python triage_classifier.py --csv data.csv
    python triage_classifier.py --csv data.csv --predict
    python triage_classifier.py --csv data.csv --save-model model.pkl
    python triage_classifier.py --load-model model.pkl --predict
"""

import argparse
import sys
import pickle
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[INFO] xgboost not installed -- Random Forest will be used instead.")

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────

# Target column
TARGET = "triage_acuity"

KTAS_LABELS = {
    1: "KTAS 1 - Resuscitation",
    2: "KTAS 2 - Emergent",
    3: "KTAS 3 - Urgent",
    4: "KTAS 4 - Less Urgent",
    5: "KTAS 5 - Non-Urgent",
}

KTAS_COLORS = {
    1: "#E24B4A",
    2: "#EF9F27",
    3: "#378ADD",
    4: "#639922",
    5: "#888780",
}

# ── Exact column groups from the dataset ─────────────────────

# Columns to DROP (IDs, leakage columns, post-visit outcomes)
DROP_COLS = [
    "patient_id",
    "site_id",
    "triage_nurse_id",
    "disposition",       # determined AFTER triage -- leakage
    "ed_los_hours",      # post-visit outcome -- leakage
]

# Numeric feature columns (used directly, no renaming)
NUMERIC_COLS = [
    "age",
    "arrival_hour",
   
    "arrival_month",
    "num_prior_ed_visits_12m",
    "num_prior_admissions_12m",
    "num_active_medications",
    "num_comorbidities",
    "systolic_bp",
    "diastolic_bp",
    "mean_arterial_pressure",
    "pulse_pressure",
    "heart_rate",
    "respiratory_rate",
    "temperature_c",
    "spo2",
    "gcs_total",
    "pain_score",
    "weight_kg",
    "height_cm",
    "bmi",
    "shock_index",
    "news2_score",        # already a composite severity score -- very predictive
]

# Categorical feature columns (will be one-hot encoded)
CATEGORICAL_COLS = [
    "arrival_day",
    "arrival_mode",
    "arrival_season",
    "shift",
    "age_group",
    "sex",
    "language",
    "insurance_type",
    "transport_origin",
    "pain_location",
    "mental_status_triage",
    "chief_complaint_system",
]

# Chief complaint keyword flags (derived from chief_complaint_system text if present)
COMPLAINT_KEYWORDS = {
    "cc_chest_pain":    ["chest pain", "chest", "angina"],
    "cc_dyspnea":       ["shortness of breath", "dyspnea", "sob", "breath"],
    "cc_altered_loc":   ["altered", "unconscious", "syncope", "faint", "lethargic"],
    "cc_seizure":       ["seizure", "convuls", "epileps"],
    "cc_stroke":        ["stroke", "facial droop", "weakness", "hemiplegia"],
    "cc_trauma":        ["trauma", "accident", "fall", "injury", "hit"],
    "cc_hemorrhage":    ["hemorrhage", "bleeding", "blood"],
    "cc_cardiac":       ["cardiac", "heart", "palpitation"],
    "cc_abdominal":     ["abdominal", "abdomen", "stomach", "nausea", "vomit"],
    "cc_headache":      ["headache", "migraine"],
    "cc_back_pain":     ["back pain", "lumbar", "loin"],
    "cc_fever":         ["fever", "febrile", "pyrexia"],
    "cc_urinary":       ["urinary", "dysuria", "urine"],
    "cc_skin":          ["rash", "skin", "itching", "allergy"],
}


# ─────────────────────────────────────────────────────────────
#  STEP 1: DATA LOADING
# ─────────────────────────────────────────────────────────────

def load_data(csv_path: str) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print("  STEP 1 -- Loading data")

    df = pd.read_csv(csv_path)

    print(f"  Shape  : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  Columns: {list(df.columns)}")

    # ── Validate target ───────────────────────────────────────────────────
    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found in CSV.\n"
            f"Available columns: {list(df.columns)}"
        )

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    before = len(df)
    df = df[df[TARGET].between(1, 5)].copy()
    df[TARGET] = df[TARGET].astype(int)
    print(f"  Dropped {before - len(df):,} rows with invalid triage_acuity values")

    print(f"\n  Triage acuity distribution:")
    for level, count in df[TARGET].value_counts().sort_index().items():
        pct = count / len(df) * 100
        bar = "=" * int(pct / 2)
        label = KTAS_LABELS.get(level, str(level))
        print(f"    {label:<34} {count:>6,}  ({pct:4.1f}%)  [{bar}]")

    return df


# ─────────────────────────────────────────────────────────────
#  STEP 2: FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame):
    print(f"\n{'='*60}")
    print("  STEP 2 -- Feature engineering")
    print(f"{'='*60}")

    # Drop leakage / ID columns that exist in the dataframe
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"  Dropped columns: {cols_to_drop}")

    # ── Derived clinical flags (from raw vitals already in dataset) ───────
    derived_numeric = []

    # Clinical threshold flags
    if "systolic_bp" in df.columns:
        df["flag_hypotension"]  = (df["systolic_bp"] < 90).astype(int)
        df["flag_hypertension"] = (df["systolic_bp"] > 180).astype(int)
        derived_numeric += ["flag_hypotension", "flag_hypertension"]

    if "heart_rate" in df.columns:
        df["flag_tachycardia"] = (df["heart_rate"] > 100).astype(int)
        df["flag_bradycardia"] = (df["heart_rate"] < 60).astype(int)
        derived_numeric += ["flag_tachycardia", "flag_bradycardia"]

    if "respiratory_rate" in df.columns:
        df["flag_tachypnea"] = (df["respiratory_rate"] > 20).astype(int)
        df["flag_bradypnea"] = (df["respiratory_rate"] < 12).astype(int)
        derived_numeric += ["flag_tachypnea", "flag_bradypnea"]

    if "temperature_c" in df.columns:
        df["flag_fever"]       = (df["temperature_c"] > 38.3).astype(int)
        df["flag_hypothermia"] = (df["temperature_c"] < 35.0).astype(int)
        derived_numeric += ["flag_fever", "flag_hypothermia"]

    if "spo2" in df.columns:
        df["flag_hypoxia"] = (df["spo2"] < 94).astype(int)
        derived_numeric.append("flag_hypoxia")

    if "pain_score" in df.columns:
        df["flag_severe_pain"] = (df["pain_score"] >= 7).astype(int)
        derived_numeric.append("flag_severe_pain")

    if "gcs_total" in df.columns:
        df["flag_low_gcs"]      = (df["gcs_total"] < 14).astype(int)
        df["flag_critical_gcs"] = (df["gcs_total"] <= 8).astype(int)
        derived_numeric += ["flag_low_gcs", "flag_critical_gcs"]

    # Chief complaint keyword flags (text mining on chief_complaint_system)
    cc_flags = []
    if "chief_complaint_system" in df.columns:
        cc_text = df["chief_complaint_system"].fillna("").str.lower()
        for flag_col, keywords in COMPLAINT_KEYWORDS.items():
            pattern = "|".join(keywords)
            df[flag_col] = cc_text.str.contains(pattern, na=False).astype(int)
            cc_flags.append(flag_col)
        print(f"  Chief complaint flags created: {len(cc_flags)}")

    # ── Final feature lists ───────────────────────────────────────────────
    # Only keep columns that actually exist in the dataframe
    numeric_cols = [c for c in NUMERIC_COLS if c in df.columns] \
                   + derived_numeric + cc_flags

    categorical_cols = [c for c in CATEGORICAL_COLS if c in df.columns]

    # Deduplicate
    numeric_cols     = list(dict.fromkeys(numeric_cols))
    categorical_cols = list(dict.fromkeys(categorical_cols))

    print(f"  Numeric features    : {len(numeric_cols)}")
    print(f"  Categorical features: {len(categorical_cols)}  -> {categorical_cols}")

    return df, numeric_cols, categorical_cols


# ─────────────────────────────────────────────────────────────
#  STEP 3: PREPROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────

def build_preprocessor(numeric_cols: list, categorical_cols: list):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    steps = [("num", num_pipe, numeric_cols)]
    if categorical_cols:
        steps.append(("cat", cat_pipe, categorical_cols))
    return ColumnTransformer(steps)


# ─────────────────────────────────────────────────────────────
#  STEP 4: TRAINING & MODEL SELECTION
# ─────────────────────────────────────────────────────────────

def train_models(X_train, y_train, X_test, y_test, preprocessor):
    print(f"\n{'='*60}")
    print("  STEP 4 -- Training & selecting models")
    print(f"{'='*60}")

    candidates = {
        "Logistic Regression": Pipeline([
            ("pre", preprocessor),
            ("clf", LogisticRegression(
                max_iter=2000, class_weight="balanced", C=0.5, solver="lbfgs"
            )),
        ]),
        "Random Forest": Pipeline([
            ("pre", preprocessor),
            ("clf", RandomForestClassifier(
                n_estimators=400, max_depth=15, min_samples_leaf=3,
                class_weight="balanced", random_state=42, n_jobs=-1
            )),
        ]),
    }

    results = {}
    best_name, best_model, best_acc = None, None, 0.0

    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        acc = accuracy_score(y_test, pipe.predict(X_test))
        print(f"  {name:<28}  accuracy = {acc:.4f}")
        results[name] = {"model": pipe, "accuracy": acc, "is_xgb": False, "le": None}
        if acc > best_acc:
            best_acc, best_name, best_model = acc, name, pipe

    # XGBoost needs 0-indexed labels
    if HAS_XGB:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_tr_enc = le.fit_transform(y_train)
        y_te_enc = le.transform(y_test)

        xgb_pipe = Pipeline([
            ("pre", preprocessor),
            ("clf", XGBClassifier(
                n_estimators=500, max_depth=7, learning_rate=0.04,
                subsample=0.8, colsample_bytree=0.8, gamma=0.1,
                use_label_encoder=False, eval_metric="mlogloss",
                random_state=42, n_jobs=-1
            )),
        ])
        xgb_pipe.fit(X_train, y_tr_enc)
        acc = accuracy_score(y_te_enc, xgb_pipe.predict(X_test))
        print(f"  {'XGBoost':<28}  accuracy = {acc:.4f}")
        results["XGBoost"] = {
            "model": xgb_pipe, "accuracy": acc, "is_xgb": True, "le": le
        }
        if acc > best_acc:
            best_acc, best_name, best_model = acc, "XGBoost", xgb_pipe

    print(f"\n  Best model : {best_name}  (accuracy = {best_acc:.4f})")
    return best_model, best_name, results


# ─────────────────────────────────────────────────────────────
#  STEP 5: EVALUATION & PLOTS
# ─────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, results, best_name):
    print(f"\n{'='*60}")
    print("  STEP 5 -- Evaluation")
    print(f"{'='*60}")

    info   = results[best_name]
    is_xgb = info["is_xgb"]
    le     = info["le"]

    if is_xgb:
        y_pred = le.inverse_transform(model.predict(X_test))
        proba  = model.predict_proba(X_test)
        classes = le.inverse_transform(np.arange(len(proba[0])))
    else:
        y_pred  = model.predict(X_test)
        proba   = model.predict_proba(X_test)
        classes = model.classes_

    labels_sorted = sorted(y_test.unique())
    label_strs    = [KTAS_LABELS.get(l, str(l)) for l in labels_sorted]

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=label_strs))
    print(f"  Overall accuracy             : {accuracy_score(y_test, y_pred):.4f}")

    try:
        auc = roc_auc_score(
            pd.get_dummies(y_test).values, proba,
            multi_class="ovr", average="macro"
        )
        print(f"  Macro AUC (one-vs-rest)      : {auc:.4f}")
    except Exception:
        pass

    y_bin_t = (y_test <= 3).astype(int)
    y_bin_p = (pd.Series(y_pred) <= 3).astype(int)
    print(f"  Emergency vs Non-Emergency   : {accuracy_score(y_bin_t, y_bin_p):.4f}")

    _plot_confusion(y_test, y_pred, labels_sorted, label_strs)
    _plot_distributions(y_test, y_pred, labels_sorted)

    return y_pred, proba, classes


def _plot_confusion(y_test, y_pred, labels_sorted, label_strs):
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_strs, yticklabels=label_strs, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix -- KTAS Triage Prediction")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("  Saved: confusion_matrix.png")
    plt.close()


def _plot_distributions(y_test, y_pred, labels_sorted):
    colors = [KTAS_COLORS.get(l, "#888") for l in labels_sorted]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    actual = pd.Series(y_test).value_counts().sort_index()
    axes[0].bar([KTAS_LABELS.get(l, str(l)) for l in actual.index],
                actual.values, color=colors)
    axes[0].set_title("Actual KTAS distribution (test set)")
    axes[0].tick_params(axis="x", rotation=30)

    pred = pd.Series(y_pred).value_counts().sort_index()
    axes[1].bar([KTAS_LABELS.get(l, str(l)) for l in pred.index],
                pred.values, color=colors)
    axes[1].set_title("Predicted KTAS distribution (test set)")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig("distribution_comparison.png", dpi=150)
    print("  Saved: distribution_comparison.png")
    plt.close()


def plot_feature_importance(model, numeric_cols, categorical_cols):
    clf = model.named_steps.get("clf")
    if clf is None or not hasattr(clf, "feature_importances_"):
        return
    pre = model.named_steps["pre"]
    try:
        feat_names = list(numeric_cols)
        if categorical_cols:
            ohe = pre.named_transformers_["cat"].named_steps["ohe"]
            feat_names += list(ohe.get_feature_names_out(categorical_cols))
        fi = pd.Series(clf.feature_importances_, index=feat_names).nlargest(25)
        fig, ax = plt.subplots(figsize=(9, 7))
        fi.sort_values().plot.barh(ax=ax, color="#378ADD")
        ax.set_title("Top 25 Feature Importances -- KTAS Classifier")
        ax.set_xlabel("Importance")
        plt.tight_layout()
        plt.savefig("feature_importance.png", dpi=150)
        print("  Saved: feature_importance.png")
        plt.close()
    except Exception as e:
        print(f"  [WARN] Feature importance plot failed: {e}")


# ─────────────────────────────────────────────────────────────
#  PREDICT: SINGLE PATIENT WITH CONFIDENCE SCORE
# ─────────────────────────────────────────────────────────────

def predict_single(model, numeric_cols, categorical_cols,
                   patient: dict, is_xgb=False, le=None) -> dict:
    """
    Predict triage_acuity level + per-class probabilities for one patient.
    patient dict keys must match the column names used at training.
    """
    row   = {col: [patient.get(col, np.nan)] for col in numeric_cols + categorical_cols}
    df_in = pd.DataFrame(row)
    proba = model.predict_proba(df_in)[0]

    classes = (le.inverse_transform(np.arange(len(proba)))
               if is_xgb and le is not None
               else model.classes_)

    pred_idx   = int(np.argmax(proba))
    pred_class = int(classes[pred_idx])
    confidence = float(proba[pred_idx])

    return {
        "ktas_level"    : pred_class,
        "label"         : KTAS_LABELS.get(pred_class, str(pred_class)),
        "confidence"    : round(confidence, 4),
        "confidence_pct": f"{confidence * 100:.1f}%",
        "is_emergency"  : pred_class <= 3,
        "probabilities" : {
            KTAS_LABELS.get(int(c), str(c)): round(float(p), 4)
            for c, p in zip(classes, proba)
        },
    }


def _build_patient_features(raw: dict) -> dict:
    """Apply the same derived flags to a raw input dict as done during training."""
    p = dict(raw)

    sbp = p.get("systolic_bp", np.nan)
    hr  = p.get("heart_rate",  np.nan)
    rr  = p.get("respiratory_rate", np.nan)
    bt  = p.get("temperature_c", np.nan)
    spo2 = p.get("spo2", np.nan)
    gcs  = p.get("gcs_total", np.nan)
    pain = p.get("pain_score", np.nan)

    def _f(val, cond):
        return int(cond) if not (isinstance(val, float) and np.isnan(val)) else 0

    p["flag_hypotension"]  = _f(sbp, sbp < 90)
    p["flag_hypertension"] = _f(sbp, sbp > 180)
    p["flag_tachycardia"]  = _f(hr,  hr > 100)
    p["flag_bradycardia"]  = _f(hr,  hr < 60)
    p["flag_tachypnea"]    = _f(rr,  rr > 20)
    p["flag_bradypnea"]    = _f(rr,  rr < 12)
    p["flag_fever"]        = _f(bt,  bt > 38.3)
    p["flag_hypothermia"]  = _f(bt,  bt < 35.0)
    p["flag_hypoxia"]      = _f(spo2, spo2 < 94)
    p["flag_severe_pain"]  = _f(pain, pain >= 7)
    p["flag_low_gcs"]      = _f(gcs,  gcs < 14)
    p["flag_critical_gcs"] = _f(gcs,  gcs <= 8)

    # Chief complaint keyword flags
    cc = str(p.get("chief_complaint_system", "")).lower()
    for flag_col, keywords in COMPLAINT_KEYWORDS.items():
        pattern = "|".join(keywords)
        p[flag_col] = int(bool(pd.Series([cc]).str.contains(pattern, na=False)[0]))

    return p


def interactive_predict(model, numeric_cols, categorical_cols,
                        is_xgb=False, le=None):
    print(f"\n{'='*60}")
    print("  INTERACTIVE PATIENT PREDICTION")
    print(f"{'='*60}")
    print("  Enter values and press Enter (blank = use default/median)\n")

    # (internal_col, display_label, type, default, hint)
    prompts = [
        ("age",                  "Age (years)",                    float, 45,   ""),
        ("sex",                  "Sex",                            str,   "M",  "M / F"),
        ("arrival_mode",         "Arrival mode",                   str,   "Walk-in", "Walk-in / Ambulance / etc."),
        ("arrival_hour",         "Arrival hour (0-23)",            int,   12,   ""),
        ("arrival_season",       "Arrival season",                 str,   "Spring", "Spring/Summer/Fall/Winter"),
        ("shift",                "Shift",                          str,   "Day", "Day / Evening / Night"),
        ("mental_status_triage", "Mental status",                  str,   "Alert", "Alert / Verbal / Pain / Unresponsive"),
        ("pain_score",           "Pain score (0-10)",              float, 0,    ""),
        ("systolic_bp",          "Systolic BP (mmHg)",             float, 120,  ""),
        ("diastolic_bp",         "Diastolic BP (mmHg)",            float, 80,   ""),
        ("heart_rate",           "Heart rate (bpm)",               float, 80,   ""),
        ("respiratory_rate",     "Respiratory rate (breaths/min)", float, 16,   ""),
        ("temperature_c",        "Temperature (C)",                float, 37.0, ""),
        ("spo2",                 "SpO2 (%)",                       float, 98,   ""),
        ("gcs_total",            "GCS total (3-15)",               float, 15,   ""),
        ("shock_index",          "Shock index (HR/SBP)",           float, 0.67, "leave blank to auto-calculate"),
        ("news2_score",          "NEWS2 score",                    float, 0,    "leave blank if unknown"),
        ("num_comorbidities",    "Number of comorbidities",        int,   0,    ""),
        ("num_active_medications","Active medications count",      int,   0,    ""),
        ("chief_complaint_system","Chief complaint (text)",        str,   "",   "e.g. chest pain, shortness of breath"),
    ]

    raw = {}
    for col, label, typ, default, hint in prompts:
        suffix = f"  ({hint})" if hint else ""
        val = input(f"  {label}{suffix}  [{default}]: ").strip()
        if val == "":
            raw[col] = default
        else:
            try:
                raw[col] = typ(val)
            except ValueError:
                raw[col] = default

    # Auto-calculate shock index if not provided
    if raw.get("shock_index") == 0.67:  # still the default
        sbp = raw.get("systolic_bp", 0)
        hr  = raw.get("heart_rate",  0)
        if sbp and sbp > 0:
            raw["shock_index"] = round(hr / sbp, 3)

    patient = _build_patient_features(raw)
    result  = predict_single(model, numeric_cols, categorical_cols,
                             patient, is_xgb=is_xgb, le=le)

    icons = {1: "[!!!]", 2: "[!! ]", 3: "[ ! ]", 4: "[   ]", 5: "[   ]"}
    level = result["ktas_level"]

    print(f"\n{'='*60}")
    print("  TRIAGE RESULT")
    print(f"{'='*60}")
    print(f"  {icons[level]}  {result['label']}")
    print(f"  Confidence  : {result['confidence_pct']}")
    print(f"  Emergency   : {'YES -- needs immediate attention' if result['is_emergency'] else 'No -- non-emergency'}")
    print("\n  Per-level probabilities:")
    for lbl, prob in result["probabilities"].items():
        lvl_num = [k for k, v in KTAS_LABELS.items() if v == lbl]
        icon = icons.get(lvl_num[0], "     ") if lvl_num else "     "
        bar  = "#" * int(prob * 40)
        print(f"    {icon} {lbl:<34} {prob*100:5.1f}%  {bar}")
    print(f"{'='*60}")


# ─────────────────────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def run_pipeline(csv_path: str, save_path: str = None,
                 predict: bool = False, test_size: float = 0.2):

    df = load_data(csv_path)
    df, numeric_cols, categorical_cols = engineer_features(df)

    print(f"\n{'='*60}")
    print("  STEP 3 -- Train / test split")
    print(f"{'='*60}")

    # Only use feature columns that actually exist after engineering
    available_num = [c for c in numeric_cols     if c in df.columns]
    available_cat = [c for c in categorical_cols if c in df.columns]
    all_features  = available_num + available_cat

    X = df[all_features]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=42
    )
    print(f"  Train : {len(X_train):,}  |  Test : {len(X_test):,}")
    print(f"  Total features used: {len(all_features)}")

    preprocessor = build_preprocessor(available_num, available_cat)
    best_model, best_name, all_results = train_models(
        X_train, y_train, X_test, y_test, preprocessor
    )
    evaluate(best_model, X_test, y_test, all_results, best_name)
    plot_feature_importance(best_model, available_num, available_cat)

    is_xgb = all_results[best_name]["is_xgb"]
    le     = all_results[best_name]["le"]

    if save_path:
        bundle = {
            "model"           : best_model,
            "model_name"      : best_name,
            "numeric_cols"    : available_num,
            "categorical_cols": available_cat,
            "is_xgb"          : is_xgb,
            "le"              : le,
        }
        with open(save_path, "wb") as f:
            pickle.dump(bundle, f)
        print(f"\n  Model saved -> {save_path}")

    if predict:
        interactive_predict(best_model, available_num, available_cat,
                            is_xgb=is_xgb, le=le)

    return best_model, available_num, available_cat


def load_and_run(model_path: str):
    print(f"\nLoading model from: {model_path}")
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    print(f"Model: {bundle['model_name']}")
    interactive_predict(
        bundle["model"],
        bundle["numeric_cols"],
        bundle["categorical_cols"],
        is_xgb=bundle.get("is_xgb", False),
        le=bundle.get("le"),
    )


# ─────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="KTAS Triage ML Classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--csv",        type=str,   help="Path to data.csv")
    p.add_argument("--save-model", type=str,   default="triage_model.pkl",
                   metavar="PATH", help="Save trained model (default: triage_model.pkl)")
    p.add_argument("--load-model", type=str,   metavar="PATH",
                   help="Load a saved model and run interactive prediction")
    p.add_argument("--predict",    action="store_true",
                   help="Run interactive single-patient prediction after training")
    p.add_argument("--test-size",  type=float, default=0.2,
                   help="Test set fraction (default: 0.2)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.load_model:
        load_and_run(args.load_model)
    elif args.csv:
        run_pipeline(
            csv_path  = args.csv,
            save_path = args.save_model,
            predict   = args.predict,
            test_size = args.test_size,
        )
    else:
        print(__doc__)
        print("\nQuick start:")
        print("  python triage_classifier.py --csv data.csv --predict")
        sys.exit(1)