"""
KTAS Emergency Triage Classification System
============================================
Three data sources merged into one pipeline:

  1. MAIN TABLE  (data.csv)
     Vitals, demographics, arrival info, NEWS2, etc.
     Target: triage_acuity (1-5)

  2. COMPLAINT TABLE  (complaints.csv)
     patient_id, chief_complaint_raw (free text), chief_complaint_system
     TF-IDF on chief_complaint_raw -> dense numeric features

  3. HISTORY TABLE  (history.csv)
     patient_id + 25 binary hx_* comorbidity columns

All three are joined on patient_id before training.
If a file is missing the pipeline still runs on whatever is available.

Usage:
    pip install pandas scikit-learn xgboost matplotlib seaborn

    # All three files
    python triage_classifier.py --csv data.csv --complaints complaints.csv --history history.csv

    # Without history
    python triage_classifier.py --csv data.csv --complaints complaints.csv

    # Predict after training
    python triage_classifier.py --csv data.csv --complaints complaints.csv --history history.csv --predict

    # Load saved model
    python triage_classifier.py --load-model triage_model.pkl --predict
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from scipy.sparse import hstack, issparse
import scipy.sparse as sp

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

# Columns to drop before training (IDs + post-visit leakage)
DROP_COLS = [
    "site_id", "triage_nurse_id",
    "disposition",    # outcome decided AFTER triage
    "ed_los_hours",   # post-visit duration
    # text columns handled separately -- not fed raw into structured pipeline
    "chief_complaint_raw",
    "chief_complaint_system",
]

# ── Structured numeric features ───────────────────────────────
NUMERIC_COLS = [
    "age",
    "arrival_hour",  "arrival_month",
    "num_prior_ed_visits_12m", "num_prior_admissions_12m",
    "num_active_medications", "num_comorbidities",
    "systolic_bp", "diastolic_bp",
    "mean_arterial_pressure", "pulse_pressure",
    "heart_rate", "respiratory_rate",
    "temperature_c", "spo2", "gcs_total",
    "pain_score", "weight_kg", "height_cm",
    "bmi", "shock_index", "news2_score",
]

# ── Medical history binary columns ───────────────────────────
HISTORY_COLS = [
    "hx_hypertension", "hx_diabetes_type2", "hx_diabetes_type1",
    "hx_asthma", "hx_copd", "hx_heart_failure",
    "hx_atrial_fibrillation", "hx_ckd", "hx_liver_disease",
    "hx_malignancy", "hx_obesity", "hx_depression",
    "hx_anxiety", "hx_dementia", "hx_epilepsy",
    "hx_hypothyroidism", "hx_hyperthyroidism", "hx_hiv",
    "hx_coagulopathy", "hx_immunosuppressed", "hx_pregnant",
    "hx_substance_use_disorder", "hx_coronary_artery_disease",
    "hx_stroke_prior", "hx_peripheral_vascular_disease",
]

# ── Categorical features (one-hot encoded) ────────────────────
CATEGORICAL_COLS = [
    "arrival_mode", "arrival_day", "arrival_season", "shift",
    "age_group", "sex", "language",
    "insurance_type", "transport_origin",
    "pain_location", "mental_status_triage",
]

# TF-IDF settings for chief_complaint_raw
TFIDF_MAX_FEATURES  = 300   # top 300 unigrams + bigrams
TFIDF_NGRAM_RANGE   = (1, 2)


# ─────────────────────────────────────────────────────────────
#  STEP 1: DATA LOADING & JOINING
# ─────────────────────────────────────────────────────────────

def load_data(csv_path: str,
              complaints_path: str = None,
              history_path: str = None) -> pd.DataFrame:

    print(f"\n{'='*60}")
    print("  STEP 1 -- Loading & joining data")
    print(f"{'='*60}")

    # ── Main table ────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    print(f"  Main table   : {df.shape[0]:,} rows x {df.shape[1]} cols")

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' not found.\n"
            f"Available: {list(df.columns)}"
        )

    # ── Complaints table (TF-IDF source) ─────────────────────
    if complaints_path:
        cc = pd.read_csv(complaints_path)
        print(f"  Complaints   : {cc.shape[0]:,} rows x {cc.shape[1]} cols")

        # Keep only the columns we need
        keep = ["patient_id"]
        if "chief_complaint_raw" in cc.columns:
            keep.append("chief_complaint_raw")
        if "chief_complaint_system" in cc.columns:
            keep.append("chief_complaint_system")

        cc = cc[keep].drop_duplicates(subset=["patient_id"])
        df = df.merge(cc, on="patient_id", how="left")
        print(f"  After complaint join: {df.shape[0]:,} rows")
    else:
        print("  [SKIP] No complaints file provided")

    # ── History table ─────────────────────────────────────────
    if history_path:
        hx = pd.read_csv(history_path)
        print(f"  History      : {hx.shape[0]:,} rows x {hx.shape[1]} cols")

        hx = hx.drop_duplicates(subset=["patient_id"])
        df = df.merge(hx, on="patient_id", how="left")
        print(f"  After history join : {df.shape[0]:,} rows")
    else:
        print("  [SKIP] No history file provided")

    # ── Clean target ──────────────────────────────────────────
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    before = len(df)
    df = df[df[TARGET].between(1, 5)].copy()
    df[TARGET] = df[TARGET].astype(int)
    print(f"  Dropped {before - len(df):,} rows with invalid triage_acuity")

    print(f"\n  Class distribution:")
    for level, count in df[TARGET].value_counts().sort_index().items():
        pct   = count / len(df) * 100
        bar   = "=" * int(pct / 2)
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

    # ── Drop leakage / ID / raw text columns ─────────────────
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    # Keep patient_id for now (needed for potential joins); drop at model time
    df_clean = df.drop(columns=cols_to_drop, errors="ignore")
    print(f"  Dropped (leakage/IDs/text): {cols_to_drop}")

    # ── Clinical threshold flags (derived from vitals) ────────
    derived = []

    def _flag(condition_series, name):
        df_clean[name] = condition_series.astype(int)
        derived.append(name)

    if "systolic_bp" in df_clean.columns:
        _flag(df_clean["systolic_bp"] < 90,  "flag_hypotension")
        _flag(df_clean["systolic_bp"] > 180, "flag_hypertension")

    if "heart_rate" in df_clean.columns:
        _flag(df_clean["heart_rate"] > 100, "flag_tachycardia")
        _flag(df_clean["heart_rate"] < 60,  "flag_bradycardia")

    if "respiratory_rate" in df_clean.columns:
        _flag(df_clean["respiratory_rate"] > 20, "flag_tachypnea")
        _flag(df_clean["respiratory_rate"] < 12, "flag_bradypnea")

    if "temperature_c" in df_clean.columns:
        _flag(df_clean["temperature_c"] > 38.3, "flag_fever")
        _flag(df_clean["temperature_c"] < 35.0, "flag_hypothermia")

    if "spo2" in df_clean.columns:
        _flag(df_clean["spo2"] < 94, "flag_hypoxia")

    if "pain_score" in df_clean.columns:
        _flag(df_clean["pain_score"] >= 7, "flag_severe_pain")

    if "gcs_total" in df_clean.columns:
        _flag(df_clean["gcs_total"] < 14, "flag_low_gcs")
        _flag(df_clean["gcs_total"] <= 8, "flag_critical_gcs")

    # ── Comorbidity burden score ──────────────────────────────
    hx_present = [c for c in HISTORY_COLS if c in df_clean.columns]
    if hx_present:
        df_clean["hx_total_burden"] = df_clean[hx_present].fillna(0).sum(axis=1)
        derived.append("hx_total_burden")
        # High-risk comorbidity flag (heart failure, COPD, malignancy, CKD, dementia)
        high_risk = ["hx_heart_failure", "hx_copd", "hx_malignancy",
                     "hx_ckd", "hx_dementia", "hx_coagulopathy",
                     "hx_immunosuppressed"]
        hr_present = [c for c in high_risk if c in df_clean.columns]
        if hr_present:
            df_clean["hx_high_risk_flag"] = (
                df_clean[hr_present].fillna(0).sum(axis=1) > 0
            ).astype(int)
            derived.append("hx_high_risk_flag")
        print(f"  History cols found : {len(hx_present)} / {len(HISTORY_COLS)}")
        print(f"  Added hx_total_burden + hx_high_risk_flag")
    else:
        print("  [INFO] No history columns found in merged dataframe")

    # ── Assemble final feature lists ──────────────────────────
    numeric_cols = (
        [c for c in NUMERIC_COLS if c in df_clean.columns]
        + derived
        + [c for c in hx_present]          # raw hx_ binary columns
    )
    categorical_cols = [c for c in CATEGORICAL_COLS if c in df_clean.columns]

    # Deduplicate
    numeric_cols     = list(dict.fromkeys(numeric_cols))
    categorical_cols = list(dict.fromkeys(categorical_cols))

    print(f"  Numeric features    : {len(numeric_cols)}")
    print(f"  Categorical features: {len(categorical_cols)}")
    print(f"  Sample numeric      : {numeric_cols[:8]}")

    return df_clean, numeric_cols, categorical_cols


# ─────────────────────────────────────────────────────────────
#  STEP 3: BUILD PREPROCESSOR (structured data only)
# ─────────────────────────────────────────────────────────────

def build_structured_preprocessor(numeric_cols: list, categorical_cols: list):
    """
    Returns a ColumnTransformer that handles numeric + categorical columns.
    TF-IDF features are handled separately and hstacked after.
    """
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
#  STEP 4: TF-IDF ON chief_complaint_raw
# ─────────────────────────────────────────────────────────────

def fit_tfidf(df_train: pd.DataFrame) -> TfidfVectorizer:
    """Fit TF-IDF on the training set raw complaint text."""
    tfidf = TfidfVectorizer(
        max_features   = TFIDF_MAX_FEATURES,
        ngram_range    = TFIDF_NGRAM_RANGE,
        sublinear_tf   = True,       # log(tf+1) -- dampens very frequent words
        strip_accents  = "unicode",
        analyzer       = "word",
        token_pattern  = r"[a-zA-Z]{2,}",   # skip single chars and digits
        min_df         = 2,          # ignore terms that appear in only 1 doc
    )
    text = df_train["chief_complaint_raw"].fillna("").astype(str)
    tfidf.fit(text)
    print(f"  TF-IDF vocabulary size: {len(tfidf.vocabulary_):,} terms")
    return tfidf


def apply_tfidf(tfidf: TfidfVectorizer, df: pd.DataFrame):
    """Transform a dataframe's raw complaint text into a sparse TF-IDF matrix."""
    text = df["chief_complaint_raw"].fillna("").astype(str) \
           if "chief_complaint_raw" in df.columns \
           else pd.Series([""] * len(df))
    return tfidf.transform(text)


def combine_features(X_structured, X_tfidf):
    """
    Horizontally stack structured (dense) and TF-IDF (sparse) feature matrices.
    Returns a dense numpy array suitable for tree models.
    """
    if issparse(X_structured):
        X_structured = X_structured.toarray()
    X_tfidf_dense = X_tfidf.toarray()
    return np.hstack([X_structured, X_tfidf_dense])


# ─────────────────────────────────────────────────────────────
#  STEP 5: TRAINING & MODEL SELECTION
# ─────────────────────────────────────────────────────────────

def train_models(X_train, y_train, X_test, y_test):
    """
    X_train / X_test are already combined dense arrays
    (structured preprocessed + TF-IDF).
    """
    print(f"\n{'='*60}")
    print("  STEP 5 -- Training & selecting models")
    print(f"{'='*60}")
    print(f"  Feature matrix shape: {X_train.shape}")

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced",
            C=0.5, solver="lbfgs"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, max_depth=15, min_samples_leaf=3,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
    }

    results = {}
    best_name, best_model, best_acc = None, None, 0.0

    for name, clf in candidates.items():
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_test, clf.predict(X_test))
        print(f"  {name:<28}  accuracy = {acc:.4f}")
        results[name] = {"model": clf, "accuracy": acc, "is_xgb": False, "le": None}
        if acc > best_acc:
            best_acc, best_name, best_model = acc, name, clf

    if HAS_XGB:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_tr_enc = le.fit_transform(y_train)
        y_te_enc = le.transform(y_test)

        xgb = XGBClassifier(
            n_estimators=500, max_depth=7, learning_rate=0.04,
            subsample=0.8, colsample_bytree=0.8, gamma=0.1,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=42, n_jobs=-1
        )
        xgb.fit(X_train, y_tr_enc)
        acc = accuracy_score(y_te_enc, xgb.predict(X_test))
        print(f"  {'XGBoost':<28}  accuracy = {acc:.4f}")
        results["XGBoost"] = {
            "model": xgb, "accuracy": acc, "is_xgb": True, "le": le
        }
        if acc > best_acc:
            best_acc, best_name, best_model = acc, "XGBoost", xgb

    print(f"\n  Best model : {best_name}  (accuracy = {best_acc:.4f})")
    return best_model, best_name, results


# ─────────────────────────────────────────────────────────────
#  STEP 6: EVALUATION & PLOTS
# ─────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, results, best_name):
    print(f"\n{'='*60}")
    print("  STEP 6 -- Evaluation")
    print(f"{'='*60}")

    info   = results[best_name]
    is_xgb = info["is_xgb"]
    le     = info["le"]

    if is_xgb:
        y_pred  = le.inverse_transform(model.predict(X_test))
        proba   = model.predict_proba(X_test)
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
    axes[0].set_title("Actual KTAS (test set)")
    axes[0].tick_params(axis="x", rotation=30)

    pred = pd.Series(y_pred).value_counts().sort_index()
    axes[1].bar([KTAS_LABELS.get(l, str(l)) for l in pred.index],
                pred.values, color=colors)
    axes[1].set_title("Predicted KTAS (test set)")
    axes[1].tick_params(axis="x", rotation=30)

    plt.tight_layout()
    plt.savefig("distribution_comparison.png", dpi=150)
    print("  Saved: distribution_comparison.png")
    plt.close()


def plot_feature_importance(model, numeric_cols, categorical_cols,
                            preprocessor, tfidf, top_n=30):
    """
    Plot top features from tree model.
    Reconstructs feature names for structured cols + TF-IDF terms.
    """
    if not hasattr(model, "feature_importances_"):
        return
    try:
        # Structured feature names
        num_names = list(numeric_cols)
        cat_names = []
        if categorical_cols:
            ohe = preprocessor.named_transformers_["cat"].named_steps["ohe"]
            cat_names = list(ohe.get_feature_names_out(categorical_cols))
        structured_names = num_names + cat_names

        # TF-IDF feature names (prefixed so they stand out)
        tfidf_names = [f"tfidf::{t}" for t in tfidf.get_feature_names_out()]

        all_names = structured_names + tfidf_names
        fi = pd.Series(model.feature_importances_, index=all_names).nlargest(top_n)

        fig, ax = plt.subplots(figsize=(9, 8))
        colors_bar = ["#E24B4A" if "tfidf::" in n else "#378ADD" for n in fi.sort_values().index]
        fi.sort_values().plot.barh(ax=ax, color=colors_bar)
        ax.set_title(f"Top {top_n} Feature Importances  (red = TF-IDF text, blue = structured)")
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

def predict_single(model, preprocessor, tfidf,
                   numeric_cols, categorical_cols,
                   patient: dict,
                   is_xgb=False, le=None) -> dict:
    """
    Predict triage_acuity + confidence for one patient dict.
    patient keys must match the column names used at training.
    """
    # Build single-row structured dataframe
    struct_row = {col: [patient.get(col, np.nan)]
                  for col in numeric_cols + categorical_cols}
    df_struct = pd.DataFrame(struct_row)
    X_struct  = preprocessor.transform(df_struct)

    # TF-IDF on raw complaint text
    raw_text  = str(patient.get("chief_complaint_raw", ""))
    X_tfidf   = tfidf.transform([raw_text])

    X_combined = combine_features(X_struct, X_tfidf)

    proba = model.predict_proba(X_combined)[0]

    classes = (le.inverse_transform(np.arange(len(proba)))
               if is_xgb and le is not None
               else model.classes_)

    pred_idx   = int(np.argmax(proba))
    pred_class = int(classes[pred_idx])
    confidence = float(proba[pred_idx])

    return {
        "ktas_level"     : pred_class,
        "label"          : KTAS_LABELS.get(pred_class, str(pred_class)),
        "confidence"     : round(confidence, 4),
        "confidence_pct" : f"{confidence * 100:.1f}%",
        "is_emergency"   : pred_class <= 3,
        "probabilities"  : {
            KTAS_LABELS.get(int(c), str(c)): round(float(p), 4)
            for c, p in zip(classes, proba)
        },
    }


def _build_patient_features(raw: dict) -> dict:
    """Compute all derived flag features from raw input values."""
    p = dict(raw)

    sbp  = p.get("systolic_bp",      np.nan)
    hr   = p.get("heart_rate",       np.nan)
    rr   = p.get("respiratory_rate", np.nan)
    bt   = p.get("temperature_c",    np.nan)
    spo2 = p.get("spo2",             np.nan)
    gcs  = p.get("gcs_total",        np.nan)
    pain = p.get("pain_score",       np.nan)

    def _f(val, cond):
        return int(cond) if not (isinstance(val, float) and np.isnan(val)) else 0

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

    # Comorbidity burden
    hx_vals = [int(p.get(c, 0) or 0) for c in HISTORY_COLS]
    p["hx_total_burden"]  = sum(hx_vals)
    high_risk = ["hx_heart_failure", "hx_copd", "hx_malignancy",
                 "hx_ckd", "hx_dementia", "hx_coagulopathy", "hx_immunosuppressed"]
    p["hx_high_risk_flag"] = int(any(int(p.get(c, 0) or 0) for c in high_risk))

    return p


def interactive_predict(model, preprocessor, tfidf,
                        numeric_cols, categorical_cols,
                        is_xgb=False, le=None):
    print(f"\n{'='*60}")
    print("  INTERACTIVE PATIENT PREDICTION")
    print(f"{'='*60}")
    print("  Press Enter to use the default value shown in brackets.\n")

    # ── Vitals & demographics ─────────────────────────────────
    vitals_prompts = [
        ("age",                   "Age (years)",                     float, 45,   ""),
        ("sex",                   "Sex",                             str,   "M",  "M / F"),
        ("arrival_mode",          "Arrival mode",                    str,   "Walk-in", ""),
        ("arrival_hour",          "Arrival hour (0-23)",             int,   12,   ""),
        ("arrival_season",        "Season",                          str,   "Spring", "Spring/Summer/Fall/Winter"),
        ("shift",                 "Shift",                           str,   "Day", "Day/Evening/Night"),
        ("mental_status_triage",  "Mental status",                   str,   "Alert", "Alert/Verbal/Pain/Unresponsive"),
        ("pain_score",            "Pain score (0-10)",               float, 0,    ""),
        ("systolic_bp",           "Systolic BP (mmHg)",              float, 120,  ""),
        ("diastolic_bp",          "Diastolic BP (mmHg)",             float, 80,   ""),
        ("heart_rate",            "Heart rate (bpm)",                float, 80,   ""),
        ("respiratory_rate",      "Respiratory rate (breaths/min)",  float, 16,   ""),
        ("temperature_c",         "Temperature (C)",                 float, 37.0, ""),
        ("spo2",                  "SpO2 (%)",                        float, 98,   ""),
        ("gcs_total",             "GCS total (3-15)",                float, 15,   ""),
        ("news2_score",           "NEWS2 score (0 if unknown)",      float, 0,    ""),
        ("shock_index",           "Shock index (blank=auto)",        float, None, ""),
        ("num_comorbidities",     "Number of comorbidities",         int,   0,    ""),
        ("num_active_medications","Active medications count",        int,   0,    ""),
        ("num_prior_ed_visits_12m","Prior ED visits (12 months)",    int,   0,    ""),
    ]

    raw = {}
    for col, label, typ, default, hint in vitals_prompts:
        suffix = f"  ({hint})" if hint else ""
        default_str = str(default) if default is not None else "auto"
        val = input(f"  {label}{suffix}  [{default_str}]: ").strip()
        if val == "":
            raw[col] = default
        else:
            try:
                raw[col] = typ(val)
            except ValueError:
                raw[col] = default

    # Auto shock index
    if raw.get("shock_index") is None:
        sbp = raw.get("systolic_bp", 0) or 0
        hr  = raw.get("heart_rate",  0) or 0
        raw["shock_index"] = round(hr / sbp, 3) if sbp > 0 else np.nan

    # ── Chief complaint raw text ──────────────────────────────
    print("\n  --- Chief Complaint ---")
    raw["chief_complaint_raw"] = input(
        "  Chief complaint (free text)  [e.g. thunderclap headache, worsening with movement]: "
    ).strip()

    # ── Medical history ───────────────────────────────────────
    print("\n  --- Medical History (1=Yes, 0=No, Enter=No) ---")
    hx_display = {
        "hx_hypertension":            "Hypertension",
        "hx_diabetes_type2":          "Diabetes Type 2",
        "hx_diabetes_type1":          "Diabetes Type 1",
        "hx_asthma":                  "Asthma",
        "hx_copd":                    "COPD",
        "hx_heart_failure":           "Heart Failure",
        "hx_atrial_fibrillation":     "Atrial Fibrillation",
        "hx_ckd":                     "Chronic Kidney Disease",
        "hx_liver_disease":           "Liver Disease",
        "hx_malignancy":              "Malignancy / Cancer",
        "hx_obesity":                 "Obesity",
        "hx_depression":              "Depression",
        "hx_anxiety":                 "Anxiety",
        "hx_dementia":                "Dementia",
        "hx_epilepsy":                "Epilepsy",
        "hx_hypothyroidism":          "Hypothyroidism",
        "hx_hyperthyroidism":         "Hyperthyroidism",
        "hx_hiv":                     "HIV",
        "hx_coagulopathy":            "Coagulopathy",
        "hx_immunosuppressed":        "Immunosuppressed",
        "hx_pregnant":                "Pregnant",
        "hx_substance_use_disorder":  "Substance Use Disorder",
        "hx_coronary_artery_disease": "Coronary Artery Disease",
        "hx_stroke_prior":            "Prior Stroke",
        "hx_peripheral_vascular_disease": "Peripheral Vascular Disease",
    }
    for col, label in hx_display.items():
        val = input(f"  {label:<35} [0]: ").strip()
        raw[col] = int(val) if val in ("0", "1") else 0

    # Build derived features and predict
    patient = _build_patient_features(raw)
    result  = predict_single(model, preprocessor, tfidf,
                             numeric_cols, categorical_cols,
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

def run_pipeline(csv_path: str,
                 complaints_path: str = None,
                 history_path: str = None,
                 save_path: str = None,
                 predict: bool = False,
                 test_size: float = 0.2):

    # ── Load & join ───────────────────────────────────────────
    df = load_data(csv_path, complaints_path, history_path)

    # ── Feature engineering ───────────────────────────────────
    df, numeric_cols, categorical_cols = engineer_features(df)

    # ── Train / test split (before any fitting) ───────────────
    print(f"\n{'='*60}")
    print("  STEP 3 -- Train / test split")
    print(f"{'='*60}")

    available_num = [c for c in numeric_cols     if c in df.columns]
    available_cat = [c for c in categorical_cols if c in df.columns]

    y = df[TARGET]
    # Drop target + patient_id from feature dataframe
    drop_before_split = [TARGET, "patient_id"]
    X_df = df.drop(columns=[c for c in drop_before_split if c in df.columns])

    # Stratified split
    train_idx, test_idx = train_test_split(
        df.index, test_size=test_size, stratify=y, random_state=42
    )
    df_train = df.loc[train_idx]
    df_test  = df.loc[test_idx]
    y_train  = y.loc[train_idx]
    y_test   = y.loc[test_idx]
    print(f"  Train : {len(df_train):,}  |  Test : {len(df_test):,}")

    # ── Step 4: Fit TF-IDF on training text only ──────────────
    print(f"\n{'='*60}")
    print("  STEP 4 -- Fitting TF-IDF on chief_complaint_raw")
    print(f"{'='*60}")

    has_raw_text = "chief_complaint_raw" in df.columns
    if has_raw_text:
        tfidf = fit_tfidf(df_train)
    else:
        print("  [INFO] chief_complaint_raw not found -- using empty TF-IDF")
        tfidf = TfidfVectorizer(max_features=1)
        tfidf.fit(["placeholder"])

    # ── Step 4b: Fit structured preprocessor on training data ─
    X_train_struct_df = df_train[available_num + available_cat]
    X_test_struct_df  = df_test[available_num + available_cat]

    preprocessor = build_structured_preprocessor(available_num, available_cat)
    X_train_struct = preprocessor.fit_transform(X_train_struct_df)
    X_test_struct  = preprocessor.transform(X_test_struct_df)

    # ── Step 4c: Apply TF-IDF ─────────────────────────────────
    X_train_tfidf = apply_tfidf(tfidf, df_train)
    X_test_tfidf  = apply_tfidf(tfidf, df_test)

    # ── Step 4d: Combine structured + TF-IDF ─────────────────
    X_train_combined = combine_features(X_train_struct, X_train_tfidf)
    X_test_combined  = combine_features(X_test_struct,  X_test_tfidf)
    print(f"  Combined feature matrix (train): {X_train_combined.shape}")
    print(f"    -> {len(available_num) + (X_train_struct.shape[1] - len(available_num))} structured  +  {X_train_tfidf.shape[1]} TF-IDF")

    # ── Train models ──────────────────────────────────────────
    best_model, best_name, all_results = train_models(
        X_train_combined, y_train,
        X_test_combined,  y_test
    )

    # ── Evaluate ──────────────────────────────────────────────
    evaluate(best_model, X_test_combined, y_test, all_results, best_name)
    plot_feature_importance(
        best_model, available_num, available_cat, preprocessor, tfidf
    )

    is_xgb = all_results[best_name]["is_xgb"]
    le     = all_results[best_name]["le"]

    # ── Save ──────────────────────────────────────────────────
    if save_path:
        bundle = {
            "model"           : best_model,
            "model_name"      : best_name,
            "preprocessor"    : preprocessor,
            "tfidf"           : tfidf,
            "numeric_cols"    : available_num,
            "categorical_cols": available_cat,
            "is_xgb"          : is_xgb,
            "le"              : le,
        }
        with open(save_path, "wb") as f:
            pickle.dump(bundle, f)
        print(f"\n  Model bundle saved -> {save_path}")
        print(f"  (includes preprocessor + TF-IDF vectorizer)")

    # ── Interactive prediction ────────────────────────────────
    if predict:
        interactive_predict(
            best_model, preprocessor, tfidf,
            available_num, available_cat,
            is_xgb=is_xgb, le=le
        )

    return best_model, preprocessor, tfidf, available_num, available_cat


def load_and_run(model_path: str):
    print(f"\nLoading model bundle from: {model_path}")
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    print(f"Model: {bundle['model_name']}")
    interactive_predict(
        bundle["model"],
        bundle["preprocessor"],
        bundle["tfidf"],
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
        description="KTAS Triage ML Classifier with TF-IDF + Medical History",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--csv",          type=str, help="Path to main data CSV")
    p.add_argument("--complaints",   type=str, default=None,
                   help="Path to complaints CSV (patient_id + chief_complaint_raw)")
    p.add_argument("--history",      type=str, default=None,
                   help="Path to history CSV (patient_id + hx_* columns)")
    p.add_argument("--save-model",   type=str, default="triage_model.pkl",
                   metavar="PATH",   help="Save model bundle (default: triage_model.pkl)")
    p.add_argument("--load-model",   type=str, metavar="PATH",
                   help="Load saved bundle and run interactive prediction")
    p.add_argument("--predict",      action="store_true",
                   help="Run interactive single-patient prediction after training")
    p.add_argument("--test-size",    type=float, default=0.2,
                   help="Test fraction (default: 0.2)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.load_model:
        load_and_run(args.load_model)
    elif args.csv:
        run_pipeline(
            csv_path        = args.csv,
            complaints_path = args.complaints,
            history_path    = args.history,
            save_path       = args.save_model,
            predict         = args.predict,
            test_size       = args.test_size,
        )
    else:
        print(__doc__)
        print("\nQuick start:")
        print("  python triage_classifier.py --csv data.csv --complaints complaints.csv --history history.csv --predict")
        sys.exit(1)