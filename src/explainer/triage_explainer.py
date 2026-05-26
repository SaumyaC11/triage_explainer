"""
triage_explainer.py
===================
LLM-powered explainability engine for the KTAS triage system.

Groq API is ONLY invoked when a patient triggers one or more escalation rails:
  1. CONFLICT    — true_acuity ≠ predicted_acuity (model disagreement)
  2. REVIEW RAIL — predicted KTAS 1 or 2 (high-acuity, mandatory review)
  3. VITAL RAIL  — critical vital sign derangements detected pre-LLM
  4. OVERRIDDEN  — clinician has already overridden the prediction
"""

import os
import threading
from dataclasses import dataclass, field
from typing import Optional

try:
    from groq import Groq
    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False


# ─────────────────────────────────────────────────────────────
#  ESCALATION GATE  — decides WHEN Groq API is called
# ─────────────────────────────────────────────────────────────

# Vital sign thresholds that trigger mandatory LLM review
# regardless of KTAS level
_CRITICAL_VITAL_RAILS = {
    "heart_rate":       ("Heart Rate",       60,   130),   # bpm
    "systolic_bp":      ("Systolic BP",      80,   200),   # mmHg
    "spo2":             ("SpO2",             90,   None),  # %
    "respiratory_rate": ("Respiratory Rate", 8,    30),    # breaths/min
    "temperature_c":    ("Temperature",      35.0, 40.0),  # °C
    "gcs_total":        ("GCS",              12,   None),  # /15
    "news2_score":      ("NEWS2",            None, 6),     # score ≥7 = high-risk
}


@dataclass
class EscalationInfo:
    """Explains WHY the Groq API was (or was not) called for a patient."""
    should_call_llm: bool
    reason_code:     str   = ""    # "CONFLICT", "KTAS_RAIL", "VITAL_RAIL", "OVERRIDE", "NONE"
    reason_label:    str   = ""    # Human-readable short label
    reason_detail:   str   = ""    # One-line detail shown in UI


def needs_escalation_review(record: dict, decision: Optional[str] = None) -> EscalationInfo:
    """
    Gate function — returns True only when the patient warrants a Groq LLM call.

    Rails (in priority order):
      1. OVERRIDE   — clinician has already overridden the prediction
      2. CONFLICT   — true_acuity present and ≠ predicted_acuity  (model disagreement)
      3. KTAS_RAIL  — predicted KTAS 1 or 2 (life-threatening / emergent)
      4. VITAL_RAIL — any critical vital sign derangement exceeds hard thresholds

    Routine KTAS 3-5 patients with normal vitals and no conflict → no LLM call.
    """
    predicted = record.get("predicted_acuity")
    true_ac   = record.get("true_acuity")
    vitals    = record.get("vitals", {})

    # Rail 1 — clinician override already in flight
    if decision == "overridden":
        return EscalationInfo(
            should_call_llm=True,
            reason_code="OVERRIDE",
            reason_label="⚠️ Clinician Override",
            reason_detail="Clinician has overridden the ML prediction — LLM review triggered.",
        )

    # Rail 2 — true vs predicted conflict
    if true_ac is not None and predicted is not None and int(true_ac) != int(predicted):
        return EscalationInfo(
            should_call_llm=True,
            reason_code="CONFLICT",
            reason_label="🔀 Acuity Conflict",
            reason_detail=(
                f"Model predicted KTAS {predicted} but ground-truth is KTAS {true_ac} "
                f"— disagreement requires LLM review."
            ),
        )

    # # Rail 3 — high-acuity KTAS 1 or 2
    # if predicted in (1, 2):
    #     return EscalationInfo(
    #         should_call_llm=True,
    #         reason_code="KTAS_RAIL",
    #         reason_label=f"🚨 KTAS {predicted} — Mandatory Review",
    #         reason_detail=(
    #             f"KTAS {predicted} prediction is life-threatening/emergent — "
    #             "automatic LLM clinical review required."
    #         ),
    #     )

    # Rail 4 — critical vital sign breach
    for key, (label, low, high) in _CRITICAL_VITAL_RAILS.items():
        val = vitals.get(key)
        if val is None:
            continue
        try:
            fval = float(val)
            if (low  is not None and fval < low) or \
               (high is not None and fval > high):
                direction = "LOW" if (low is not None and fval < low) else "HIGH"
                return EscalationInfo(
                    should_call_llm=True,
                    reason_code="VITAL_RAIL",
                    reason_label=f"📉 Critical Vital: {label}",
                    reason_detail=(
                        f"{label} = {fval} is critically {direction} "
                        f"(threshold: {'≥'+str(low) if direction=='LOW' else '≤'+str(high)}) "
                        "— LLM review triggered."
                    ),
                )
        except (TypeError, ValueError):
            pass

    # No rails triggered — routine patient, skip Groq
    return EscalationInfo(
        should_call_llm=False,
        reason_code="NONE",
        reason_label="✅ Routine — No LLM Review",
        reason_detail="KTAS 3-5, no vital derangements, no acuity conflict — Groq API skipped.",
    )

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────

# Groq-hosted open-source models — all free tier
GROQ_MODELS = {
    "llama-3.3-70b-versatile": "Llama 3.3 70B — best reasoning, ~1-2s",
    "llama-3.1-8b-instant":    "Llama 3.1 8B  — fastest, great for live dashboards",
    "mixtral-8x7b-32768":      "Mixtral 8x7B  — strong clinical language",
    "gemma2-9b-it":            "Gemma 2 9B    — good balance of speed + quality",
}

DEFAULT_MODEL = "llama-3.1-8b-instant"   # fastest for a live dashboard

KTAS_LABEL = {
    1: "Resuscitation",
    2: "Emergent",
    3: "Urgent",
    4: "Less Urgent",
    5: "Non-Urgent",
}

KTAS_CLINICAL = {
    1: "immediate life-threatening emergency requiring resuscitation",
    2: "high-risk emergency needing care within 15 minutes",
    3: "urgent condition needing care within 30 minutes",
    4: "less urgent, stable condition needing care within 1 hour",
    5: "non-urgent, routine condition",
}


# ─────────────────────────────────────────────────────────────
#  RESULT DATACLASS
# ─────────────────────────────────────────────────────────────

@dataclass
class ExplanationResult:
    patient_id:        str
    full_text:         str           = ""
    escalation_reason: str           = ""
    patient_summary:   str           = ""
    risk_flags:        list          = field(default_factory=list)
    confidence_note:   str           = ""
    is_ready:          bool          = False
    error:             Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "patient_id":        self.patient_id,
            "escalation_reason": self.escalation_reason,
            "patient_summary":   self.patient_summary,
            "risk_flags":        self.risk_flags,
            "confidence_note":   self.confidence_note,
            "full_text":         self.full_text,
            "is_ready":          self.is_ready,
            "error":             self.error,
        }


# ─────────────────────────────────────────────────────────────
#  VITAL ABNORMALITY PRE-CHECKER
# ─────────────────────────────────────────────────────────────

def _flag_vitals(vitals: dict) -> list[str]:
    """
    Pre-compute abnormal vitals before the prompt is sent.
    Grounds the LLM's reasoning in concrete derangements.
    """
    checks = {
        "heart_rate":       ("Heart Rate",       "bpm",         60,   100),
        "systolic_bp":      ("Systolic BP",       "mmHg",        90,   180),
        "spo2":             ("SpO2",              "%",           94,   None),
        "respiratory_rate": ("Respiratory Rate",  "breaths/min", 12,   20),
        "temperature_c":    ("Temperature",       "°C",          36.0, 38.3),
        "gcs_total":        ("GCS",               "/15",         14,   None),
        "pain_score":       ("Pain Score",        "/10",         None, 6),
    }
    flags = []
    for key, (label, unit, low, high) in checks.items():
        val = vitals.get(key)
        if val is None:
            continue
        try:
            fval = float(val)
            if low  is not None and fval < low:
                flags.append(f"{label} {fval}{unit} [LOW — normal ≥{low}]")
            elif high is not None and fval > high:
                flags.append(f"{label} {fval}{unit} [HIGH — normal ≤{high}]")
        except (TypeError, ValueError):
            pass
    return flags


# ─────────────────────────────────────────────────────────────
#  PROMPT BUILDER
# ─────────────────────────────────────────────────────────────

def _format_history(r: dict) -> str:
    """
    Extract patient history / comorbidities from the record.

    Supports two formats:
      - Flat hx_* keys at top level:   {"hx_diabetes": 1, "hx_hypertension": 0, ...}
      - Nested "history" dict:          {"history": {"diabetes": true, ...}}
      - Free-text "medical_history":    {"medical_history": "DM2, HTN, ..."}
    """
    lines = []

    # Free-text history string
    med_hx = r.get("medical_history") or r.get("past_medical_history") or ""
    if med_hx:
        lines.append(f"  Medical History : {med_hx}")

    # Nested history dict
    nested = r.get("history", {})
    if isinstance(nested, dict) and nested:
        active = [k.replace("_", " ").title() for k, v in nested.items() if v]
        if active:
            lines.append("  Comorbidities   : " + ", ".join(active))

    # Flat hx_* keys  (from history.csv merge)
    hx_keys = {k: v for k, v in r.items() if k.startswith("hx_") and v}
    if hx_keys:
        conds = [k[3:].replace("_", " ").title() for k in hx_keys]
        lines.append("  Comorbidities   : " + ", ".join(conds))

    # Arrival / visit metadata
    arrival_mode = r.get("arrival_mode") or r.get("arrival_transport")
    if arrival_mode:
        lines.append(f"  Arrival Mode    : {arrival_mode}")

    mental_status = r.get("mental_status") or r.get("mentation")
    if mental_status:
        lines.append(f"  Mental Status   : {mental_status}")

    injury_type = r.get("injury_type") or r.get("mechanism_of_injury")
    if injury_type:
        lines.append(f"  Injury / Mech   : {injury_type}")

    return "\n".join(lines) if lines else "  (no history data available)"


def build_prompt(r: dict, escalation_info: Optional["EscalationInfo"] = None) -> str:
    """
    Structured clinical prompt — instructs the LLM to EXPLAIN
    the ML decision, not re-triage. Includes full patient data and history.
    Escalation context is injected so the LLM addresses the specific trigger.
    """
    level      = r.get("predicted_acuity", 5)
    true_ac    = r.get("true_acuity")
    label      = KTAS_LABEL.get(level, "Unknown")
    clinical   = KTAS_CLINICAL.get(level, "")
    confidence = r.get("confidence_pct", "?")
    complaint  = (r.get("chief_complaint") or
                  r.get("chief_complaint_raw") or
                  "Not recorded")
    age        = r.get("age", "?")
    sex        = r.get("sex", "?")
    model_name = r.get("model", "ML model")
    vitals     = r.get("vitals", {})
    probs      = r.get("probabilities", {})

    hr    = vitals.get("heart_rate",       "not recorded")
    sbp   = vitals.get("systolic_bp",      "not recorded")
    dbp   = vitals.get("diastolic_bp",     "not recorded")
    spo2  = vitals.get("spo2",             "not recorded")
    rr    = vitals.get("respiratory_rate", "not recorded")
    temp  = vitals.get("temperature_c",    "not recorded")
    gcs   = vitals.get("gcs_total",        "not recorded")
    pain  = vitals.get("pain_score",       "not recorded")
    news2 = vitals.get("news2_score",      "not recorded")

    abnormal     = _flag_vitals(vitals)
    abnormal_str = (
        "\n".join(f"  ⚠ {f}" for f in abnormal)
        if abnormal else "  (no obvious vital sign derangements detected)"
    )

    prob_lines = ""
    for lbl, prob in sorted(probs.items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 25)
        prob_lines += f"  {lbl:<38} {prob*100:5.1f}%  {bar}\n"

    history_block = _format_history(r)

    # Conflict context — if true_acuity is known and differs
    conflict_block = ""
    if true_ac is not None and int(true_ac) != int(level):
        conflict_label = KTAS_LABEL.get(int(true_ac), str(true_ac))
        conflict_block = (
            f"\n⚠ ACUITY CONFLICT DETECTED:\n"
            f"  ML predicted : KTAS {level} — {label}\n"
            f"  Ground truth : KTAS {true_ac} — {conflict_label}\n"
            f"  → Address this discrepancy directly in your ESCALATION_REASON.\n"
        )

    # Escalation trigger context
    escalation_block = ""
    if escalation_info and escalation_info.reason_code != "NONE":
        escalation_block = (
            f"\n🔔 REVIEW TRIGGERED BY: {escalation_info.reason_label}\n"
            f"   {escalation_info.reason_detail}\n"
        )

    return f"""You are a senior emergency medicine physician reviewing a machine learning triage decision.
{conflict_block}{escalation_block}
═══ ML MODEL DECISION ═══
Model     : {model_name}
Prediction: KTAS {level} — {label}
Definition: {clinical}
Confidence: {confidence}

═══ PATIENT DATA ═══
Chief Complaint : {complaint}
Age / Sex       : {age}y, {sex}

Vital Signs:
  Heart Rate       : {hr} bpm
  Blood Pressure   : {sbp}/{dbp} mmHg
  SpO2             : {spo2}%
  Respiratory Rate : {rr} breaths/min
  Temperature      : {temp}°C
  GCS Total        : {gcs}/15
  Pain Score       : {pain}/10
  NEWS2 Score      : {news2}

Pre-computed Abnormal Vitals:
{abnormal_str}

Model Probability Distribution:
{prob_lines}
═══ PATIENT HISTORY ═══
{history_block}

═══ YOUR TASK ═══
Explain WHY the ML model chose KTAS {level} for this patient.
{"Also explain the CONFLICT between predicted and ground-truth acuity." if conflict_block else ""}
Do NOT re-classify or second-guess the model unless there is a documented conflict above.
Focus on which clinical findings — vitals, history, complaint — justify KTAS {level}.
Be concise, direct, and clinical.

Respond in EXACTLY this format — no preamble, no extra commentary:

ESCALATION_REASON:
[2-3 sentences. Name the specific vitals or findings that are the strongest drivers
of KTAS {level}. Explain WHY each one matters clinically for this level.
{"Include one sentence on the acuity conflict and its likely cause." if conflict_block else ""}]

PATIENT_SUMMARY:
[1-2 sentences. Who is this patient, what is their acute picture, and what relevant history bears on this presentation?]

RISK_FLAGS:
- [Specific finding + clinical significance]
- [Specific finding + clinical significance]
- [Third flag only if genuinely relevant — include comorbidity interactions if applicable]

CONFIDENCE_NOTE:
[1 sentence. What does {confidence} model confidence mean for clinical decision-making here?]"""


# ─────────────────────────────────────────────────────────────
#  RESPONSE PARSER
# ─────────────────────────────────────────────────────────────

def parse_llm_response(raw: str, patient_id: str) -> ExplanationResult:
    """
    Parse the structured LLM response into an ExplanationResult.
    Gracefully handles cases where the LLM drifts from the format.
    """
    result  = ExplanationResult(patient_id=patient_id, full_text=raw, is_ready=True)
    sections = {
        "ESCALATION_REASON": "",
        "PATIENT_SUMMARY":   "",
        "RISK_FLAGS":        "",
        "CONFIDENCE_NOTE":   "",
    }

    current = None
    for line in raw.splitlines():
        stripped = line.strip()
        matched  = False
        for key in sections:
            if stripped.startswith(key + ":") or stripped == key:
                current = key
                inline  = stripped[len(key):].lstrip(":").strip()
                if inline:
                    sections[key] += inline + "\n"
                matched = True
                break
        if not matched and current:
            sections[current] += line + "\n"

    result.escalation_reason = sections["ESCALATION_REASON"].strip()
    result.patient_summary   = sections["PATIENT_SUMMARY"].strip()
    result.confidence_note   = sections["CONFIDENCE_NOTE"].strip()

    flags = []
    for line in sections["RISK_FLAGS"].splitlines():
        clean = line.strip().lstrip("-•*·").strip()
        if clean:
            flags.append(clean)
    result.risk_flags = flags

    # Fallback: if parsing failed, dump everything into escalation_reason
    if not result.escalation_reason and raw.strip():
        result.escalation_reason = raw.strip()

    return result


# ─────────────────────────────────────────────────────────────
#  GROQ API CALLER
# ─────────────────────────────────────────────────────────────

def call_groq(prompt: str, model: str, api_key: str) -> str:
    """
    Call Groq's API using the official groq-python SDK.
    Returns raw response text or a descriptive error string.

    Groq free tier: https://console.groq.com
    pip install groq
    """
    if not HAS_GROQ:
        return (
            "GROQ_NOT_INSTALLED: Run:  pip install groq\n"
            "Then set your key:        export GROQ_API_KEY='gsk_...'"
        )

    if not api_key:
        return (
            "GROQ_NO_KEY: Set your Groq API key:\n"
            "  export GROQ_API_KEY='gsk_...'\n"
            "  Get a free key at: https://console.groq.com/keys"
        )

    try:
        client   = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a senior emergency medicine physician. "
                        "You explain AI triage decisions concisely and clinically. "
                        "Always follow the exact output format requested."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.15,    # low = consistent clinical reasoning
            max_tokens=500,      # enough for all 4 sections
            top_p=0.9,
        )
        return response.choices[0].message.content.strip()

    except Exception as exc:
        err = str(exc)
        # Provide actionable messages for common errors
        if "invalid_api_key" in err.lower() or "401" in err:
            return "GROQ_INVALID_KEY: Your API key is invalid. Check https://console.groq.com/keys"
        if "rate_limit" in err.lower() or "429" in err:
            return (
                "GROQ_RATE_LIMIT: Rate limit hit. "
                "Switch to llama-3.1-8b-instant (highest free limits) "
                "or wait a moment."
            )
        if "model_not_found" in err.lower() or "404" in err:
            return f"GROQ_BAD_MODEL: Model '{model}' not found. Use: {list(GROQ_MODELS.keys())}"
        return f"GROQ_ERROR: {err}"


# ─────────────────────────────────────────────────────────────
#  MAIN ENGINE
# ─────────────────────────────────────────────────────────────

class TriageExplainer:
    """
    Thread-safe LLM explainability engine backed by Groq API.

    - Each patient is explained only once (cached by patient_id).
    - explain_async() is non-blocking — safe in Streamlit's refresh loop.
    - set_model() switches model and clears cache.

    """

    def __init__(
        self,
        model:   str           = DEFAULT_MODEL,
        api_key: Optional[str] = None,
    ):
        self.model   = model
        self.api_key = api_key or os.environ.get("GROQ_API_KEY", "")

        self._cache   : dict[str, ExplanationResult] = {}
        self._loading : set[str]                     = set()
        self._lock    = threading.Lock()

    # ── Public API ────────────────────────────────────────────

    def explain(
        self,
        record: dict,
        escalation_info: Optional[EscalationInfo] = None,
    ) -> ExplanationResult:
        """Synchronous — blocks until Groq responds (~0.5-2s on free tier)."""
        pid = record.get("patient_id", "UNKNOWN")
        with self._lock:
            if pid in self._cache:
                return self._cache[pid]

        raw    = call_groq(build_prompt(record, escalation_info), self.model, self.api_key)
        result = parse_llm_response(raw, pid)

        with self._lock:
            self._cache[pid] = result
            self._loading.discard(pid)
        return result

    def explain_async(
        self,
        record: dict,
        escalation_info: Optional[EscalationInfo] = None,
    ) -> None:
        """
        Non-blocking: starts a background thread.
        Call get(pid) on the next Streamlit render cycle to retrieve result.
        escalation_info is forwarded into the prompt for context-aware explanations.
        """
        pid = record.get("patient_id", "UNKNOWN")
        with self._lock:
            if pid in self._cache or pid in self._loading:
                return                          # already done or in flight
            self._loading.add(pid)

        def _worker():
            raw    = call_groq(build_prompt(record, escalation_info), self.model, self.api_key)
            result = parse_llm_response(raw, pid)
            with self._lock:
                self._cache[pid] = result
                self._loading.discard(pid)

        threading.Thread(target=_worker, daemon=True).start()

    def get(self, patient_id: str) -> Optional[ExplanationResult]:
        """Return cached result or None if not ready yet."""
        with self._lock:
            return self._cache.get(patient_id)

    def is_loading(self, patient_id: str) -> bool:
        with self._lock:
            return patient_id in self._loading

    def is_ready(self, patient_id: str) -> bool:
        with self._lock:
            r = self._cache.get(patient_id)
            return r is not None and r.is_ready

    def set_model(self, model: str) -> None:
        """Switch model — clears cache so new patients use the new model."""
        with self._lock:
            if model != self.model:
                self.model  = model
                self._cache.clear()
                self._loading.clear()

    def set_api_key(self, key: str) -> None:
        """Update API key at runtime (e.g. from sidebar input)."""
        with self._lock:
            self.api_key = key
            self._cache.clear()
            self._loading.clear()

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._loading.clear()

    def cache_size(self) -> int:
        with self._lock:
            return len(self._cache)

    @staticmethod
    def available_models() -> dict:
        return GROQ_MODELS


# ─────────────────────────────────────────────────────────────
#  STANDALONE TEST  (python triage_explainer.py)
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_record = {
        "patient_id":       "TEST01",
        "predicted_acuity": 2,
        "confidence_pct":   "71.3%",
        "chief_complaint":  "chest pain with diaphoresis",
        "age":              62,
        "sex":              "M",
        "model":            "XGBoost",
        "vitals": {
            "heart_rate":       118,
            "systolic_bp":      88,
            "diastolic_bp":     54,
            "spo2":             91,
            "respiratory_rate": 24,
            "temperature_c":    37.1,
            "gcs_total":        15,
            "pain_score":       8,
            "news2_score":      9,
        },
        "probabilities": {
            "KTAS 1 – Resuscitation": 0.18,
            "KTAS 2 – Emergent":      0.71,
            "KTAS 3 – Urgent":        0.09,
            "KTAS 4 – Less Urgent":   0.01,
            "KTAS 5 – Non-Urgent":    0.01,
        },
    }

    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        print("⚠  Set GROQ_API_KEY env var first:  export GROQ_API_KEY='gsk_...'")
        print("   Get a free key at: https://console.groq.com/keys")
        exit(1)

    print("=" * 60)
    print(f"  TRIAGE EXPLAINER — standalone test")
    print(f"  Model : {DEFAULT_MODEL}")
    print(f"  Key   : {api_key[:8]}...")
    print("=" * 60)

    explainer = TriageExplainer(model=DEFAULT_MODEL, api_key=api_key)
    result    = explainer.explain(test_record)

    print(f"\n{'─'*60}")
    print("ESCALATION REASON:")
    print(result.escalation_reason)

    print(f"\n{'─'*60}")
    print("PATIENT SUMMARY:")
    print(result.patient_summary)

    print(f"\n{'─'*60}")
    print("RISK FLAGS:")
    for flag in result.risk_flags:
        print(f"  • {flag}")

    print(f"\n{'─'*60}")
    print("CONFIDENCE NOTE:")
    print(result.confidence_note)

    if result.error:
        print(f"\n[ERROR] {result.error}")