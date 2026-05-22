"""
streamlit_app.py  —  KTAS Triage Live Dashboard
================================================
Auto-refreshes every 2s. LLM explanations display automatically
on each patient card — no button needed.

Usage:
    streamlit run streamlit_app.py
    streamlit run streamlit_app.py -- --broker localhost:9092 --topic triage-output
"""

import json
import time
import argparse
import threading
from datetime import datetime

import sys
import os
import importlib.util as _ilu

# ── Load latency_store by absolute path — no sys.path manipulation ────────────
# src/kafka/ has __init__.py so adding it to sys.path shadows the kafka-python
# package. Instead we load latency_store.py directly via importlib using its
# absolute path, which never touches sys.path at all.
def _load_latency_store(db_arg=None):
    """
    Find and load latency_store.py by walking up from this file.
    Search order:
      1. Same directory as streamlit_app.py          (src/streamlit/)
      2. ../kafka/  relative to this file            (src/kafka/)
      3. cwd/latency_store.py                        (project root, if run from there)
      4. Path passed via --db arg's directory        (last resort)
    """
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "latency_store.py"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kafka", "latency_store.py"),
        os.path.join(os.getcwd(), "latency_store.py"),
        os.path.join(os.getcwd(), "src", "kafka", "latency_store.py"),
    ]
    for p in candidates:
        p = os.path.normpath(p)
        if os.path.exists(p):
            spec = _ilu.spec_from_file_location("latency_store", p)
            mod  = _ilu.module_from_spec(spec)
            sys.modules["latency_store"] = mod
            spec.loader.exec_module(mod)
            print(f"[METRICS] latency_store loaded from: {p}")
            return mod
    return None

_latency_store_mod = _load_latency_store()

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from kafka import KafkaConsumer, KafkaProducer

# Latency store — loaded above via importlib, no sys.path pollution
try:
    from latency_store import LatencyStore
    HAS_LATENCY_STORE = True
except Exception:
    HAS_LATENCY_STORE = False
    print("[METRICS] latency_store not available — metrics tabs will be empty")
# Streamlit does NOT call Groq — the consumer handles all LLM work.
# We only need the escalation gate here to know whether to show
# "pending" vs "routine" badge for patients not yet enriched.

# ─────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────

MAX_PATIENTS  = 100
REFRESH_EVERY = 2

KTAS_COLORS = {1: "#E24B4A", 2: "#EF9F27", 3: "#378ADD", 4: "#639922", 5: "#888780"}
KTAS_ICONS  = {1: "🔴", 2: "🟠", 3: "🔵", 4: "🟢", 5: "⚪"}
KTAS_LABEL  = {1: "Resuscitation", 2: "Emergent", 3: "Urgent", 4: "Less Urgent", 5: "Non-Urgent"}

GLOBAL_CSS = """
<style>
body, .stApp                { background-color: #13131f; color: #e0e0e0; }

div[data-testid="stMetricValue"] { font-size: 2rem    !important; font-weight: 700; }
div[data-testid="stMetricLabel"] { font-size: 0.82rem !important; color: #aaa; }

.stTextArea textarea        { background: #1e1e2e; color: #e0e0e0; border: 1px solid #333; }
[data-testid="stSidebar"]   { background: #0f0f1a; }

div.stButton > button {
    background-color : #1a7a3c !important;
    color            : #ffffff !important;
    border           : 1px solid #2ecc71 !important;
    border-radius    : 6px !important;
    font-weight      : 600 !important;
    transition       : background 0.15s ease, color 0.15s ease;
}
div.stButton > button:hover {
    background-color : #2ecc71 !important;
    color            : #000000 !important;
}
</style>
"""


# ─────────────────────────────────────────────────────────────
#  PATIENT STORE
# ─────────────────────────────────────────────────────────────

class PatientStore:
    def __init__(self):
        self._lock    = threading.Lock()
        self._records = []
        self._seen    = set()    # O(1) dedup
        self._index   = {}       # patient_id → position in _records
        self._dirty   = False

    def add(self, record: dict) -> bool:
        pid = record.get("patient_id")
        if not pid:
            return False
        with self._lock:
            if pid in self._seen:
                return False
            self._index[pid] = len(self._records)
            self._seen.add(pid)
            self._records.append(record)
            if len(self._records) > MAX_PATIENTS:
                oldest = self._records.pop(0)
                old_pid = oldest.get("patient_id")
                self._seen.discard(old_pid)
                self._index.pop(old_pid, None)
                # Rebuild index offsets after pop(0)
                self._index = {r.get("patient_id"): i for i, r in enumerate(self._records)}
            self._dirty = True
            return True

    def merge_enrichment(self, enrichment: dict):
        """
        Called when consumer publishes to triage-enrichment.
        Merges llm_explanation into the matching patient record in-place.
        """
        pid = enrichment.get("patient_id")
        if not pid:
            print(f"[STORE] merge_enrichment: no patient_id, skipping")
            return
        with self._lock:
            idx = self._index.get(pid)
            if idx is not None:
                self._records[idx]["llm_explanation"] = enrichment.get("llm_explanation", {})
                self._dirty = True
                print(f"[STORE] merged LLM enrichment for patient {pid} ✓")
            else:
                print(f"[STORE] merge_enrichment: patient {pid} not in store yet — enrichment arrived before raw record")

    def all(self):
        with self._lock:
            return list(self._records)

    def consume_dirty(self) -> bool:
        with self._lock:
            if self._dirty:
                self._dirty = False
                return True
            return False


# ─────────────────────────────────────────────────────────────
#  KAFKA LISTENER
# ─────────────────────────────────────────────────────────────

def kafka_listener(broker: str, topic: str, enrich_topic: str, store: PatientStore, lat_store=None):
    """
    Subscribes to two topics:
      - triage-output      → new patient prediction records  → store.add()
      - triage-enrichment  → LLM explanation from consumer   → store.merge_enrichment()
    Routed by the 'enrichment': True sentinel field.
    """
    while True:
        try:
            consumer = KafkaConsumer(
                topic,
                enrich_topic,
                bootstrap_servers=[broker],
                group_id="streamlit-triage-ui",
                auto_offset_reset="latest",
                value_deserializer=lambda m: json.loads(m.decode("utf-8")),
                consumer_timeout_ms=500,
            )
            print(f"[KAFKA] Connected → topics='{topic}', '{enrich_topic}'")
            while True:
                for msg in consumer:
                    val = msg.value
                    if val.get("enrichment"):
                        print(f"[KAFKA] enrichment received for patient {val.get('patient_id')}")
                        store.merge_enrichment(val)
                        #_lat = st.session_state.get("lat_store")
                        _lat = lat_store
                        if _lat:
                            try:
                                _lat.write_feedback(
                                    patient_id             = val.get("patient_id"),
                                    feedback_type          = "llm_explained",
                                    model_predicted_acuity = val.get("predicted_acuity"),
                                    final_label            = val.get("predicted_acuity"),
                                    explained_at           = time.time(),
                                )
                            except Exception as _e:
                                print(f"[METRICS] explained_at write failed: {_e}")
                    else:
                        store.add(val)
                time.sleep(0.1)
        except Exception as exc:
            print(f"[KAFKA] {exc} — retrying in 3s")
            time.sleep(3)


# ─────────────────────────────────────────────────────────────
#  SESSION INIT
# ─────────────────────────────────────────────────────────────

def init(broker: str, topic: str, enrich_topic: str, db_path: str = 'triage_latency.db'):
    # ── 1. Latency store FIRST — must exist before thread starts ─────────────
    if "lat_store" not in st.session_state:
        if HAS_LATENCY_STORE:
            try:
                st.session_state.lat_store = LatencyStore(db_path)
            except Exception as _e:
                st.session_state.lat_store = None
                print(f"[METRICS] LatencyStore init failed: {_e}")
        else:
            st.session_state.lat_store = None

    # ── 2. Patient store + kafka listener thread ──────────────────────────────
    if "store" not in st.session_state:
        store = PatientStore()
        st.session_state.store = store
        threading.Thread(
            target=kafka_listener,
            args=(broker, topic, enrich_topic, store, st.session_state.lat_store),
            daemon=True,
        ).start()

    # ── 3. Feedback Kafka producer ────────────────────────────────────────────
    if "feedback_producer" not in st.session_state:
        try:
            st.session_state.feedback_producer = KafkaProducer(
                bootstrap_servers=[broker],
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                key_serializer=lambda k: k.encode("utf-8") if k else None,
            )
        except Exception as e:
            st.session_state.feedback_producer = None
            print(f"[FEEDBACK] Kafka producer failed: {e}")

    for key, default in [
        ("override_open",  {}),
        ("override_text",  {}),
        ("override_ktas",  {}),
        ("decisions",      {}),
        ("accepted_ids",   []),
        ("accepted_count", 0),
    ]:
        if key not in st.session_state:
            st.session_state[key] = default


# ─────────────────────────────────────────────────────────────
#  FEEDBACK PUBLISHER
# ─────────────────────────────────────────────────────────────

def publish_feedback(
    patient: dict,
    feedback_type: str,
    final_label: int,
    override_reason: str = None,
    override_ktas: int = None,
):
    """
    Publish a clinician feedback event to the triage-feedback Kafka topic.
    feedback_consumer.py persists this to SQLite for retraining.

    feedback_type:
      "accepted"           — model + guardrails were right; clinician agreed
      "guardrail_accepted" — guardrails escalated; clinician confirmed the escalation
      "override"           — clinician corrected the KTAS level
    """
    producer = st.session_state.get("feedback_producer")
    if producer is None:
        return  # Kafka unavailable — don't crash the UI

    event = {
        "event_type":      "clinician_feedback",
        "feedback_type":   feedback_type,
        "final_label":     final_label,
        "override_reason": override_reason,
        "override_ktas":   override_ktas,
        "patient":         patient,
        "submitted_at":    datetime.now().isoformat(),  # Bug fix: was datetime.now() with wrong import
    }

    try:
        pid = patient.get("patient_id", "UNKNOWN")
        producer.send("triage-feedback", key=pid, value=event)
        producer.flush()
    except Exception as e:
        print(f"[FEEDBACK] Failed to publish for {patient.get('patient_id')}: {e}")


# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────

def _v(val) -> str:
    if val is None:
        return "&mdash;"
    s = str(val).strip()
    return s if s not in ("", "nan", "None") else "&mdash;"


def _vital_color(val, low=None, high=None) -> str:
    try:
        fval = float(val)
        if (low  is not None and fval < low) or \
           (high is not None and fval > high):
            return "#E24B4A"
    except Exception:
        pass
    return "#ddd"


# ─────────────────────────────────────────────────────────────
#  LLM EXPLANATION RENDERER  (pre-computed by consumer)
# ─────────────────────────────────────────────────────────────

def render_llm_explanation(r: dict):
    """
    Renders the LLM explanation that was pre-computed by kafka_consumer.py
    and merged into this record via the triage-enrichment topic.

    Three states:
      1. llm_explanation present + is_ready → show full explanation
      2. patient is an escalation candidate but enrichment not yet arrived → "pending" badge
      3. routine patient (no escalation) → green "no review needed" badge
    """
    llm = r.get("llm_explanation")

    # ── State 1: enrichment arrived ────────────────────────────
    if llm and llm.get("is_ready"):
        esc_code   = llm.get("escalation_code", "")
        esc_label  = llm.get("escalation_label", "LLM Review")
        esc_detail = llm.get("escalation_detail", "")

        badge_colors = {
            "CONFLICT":   ("#E24B4A", "#2a0d0d"),
            "KTAS_RAIL":  ("#EF9F27", "#2a1a00"),
            "VITAL_RAIL": ("#EF9F27", "#2a1a00"),
            "OVERRIDE":   ("#e67e22", "#2a1500"),
        }
        fg, bg = badge_colors.get(esc_code, ("#378ADD", "#0d1b2a"))

        # Trigger badge
        st.markdown(
            f'<div style="background:{bg};border-left:4px solid {fg};'
            f'border-radius:6px;padding:8px 14px;margin-top:8px;margin-bottom:2px;">'
            f'<span style="color:{fg};font-size:0.72rem;font-weight:700;letter-spacing:1px;">'
            f'🔔 LLM REVIEW — {esc_label}</span><br>'
            f'<span style="color:#aaa;font-size:0.79rem;">{esc_detail}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # # Explanation card
        # st.markdown(
        #     '<div style="background:#0d1b2a;border-left:4px solid #378ADD;'
        #     'border-radius:6px;padding:14px 18px 12px 18px;margin-top:4px;">'
        #     '<div style="color:#378ADD;font-size:0.73rem;font-weight:700;'
        #     'letter-spacing:1.2px;margin-bottom:10px;">&#129504; AI CLINICAL EXPLANATION</div>',
        #     unsafe_allow_html=True,
        # )

        # Escalation reason
        reason = llm.get("escalation_reason", "")
        if reason:
            st.markdown(
                f'<span style="color:#EF9F27;font-size:0.72rem;font-weight:700;letter-spacing:0.8px;">ESCALATION REASON</span><br>'
                f'<span style="color:#ddd;font-size:0.84rem;">{reason}</span>',
                unsafe_allow_html=True,
            )

        # Patient summary
        summary = llm.get("patient_summary", "")
        if summary:
            st.markdown(
                f'<div style="margin-top:8px;">'
                f'<span style="color:#888;font-size:0.72rem;font-weight:700;letter-spacing:0.8px;">PATIENT SUMMARY</span><br>'
                f'<span style="color:#ccc;font-size:0.84rem;">{summary}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Risk flags
        flags = llm.get("risk_flags", [])
        if flags:
            flags_html = "".join(
                f'<div style="display:flex;gap:8px;align-items:flex-start;margin-bottom:4px;">'
                f'<span style="color:#EF9F27;margin-top:1px;">&#9679;</span>'
                f'<span style="color:#ddd;font-size:0.84rem;">{flag}</span></div>'
                for flag in flags
            )
            st.markdown(
                '<div style="margin-top:10px;">'
                '<span style="color:#EF9F27;font-size:0.72rem;font-weight:700;letter-spacing:0.8px;">KEY RISK FLAGS</span>'
                '<div style="margin-top:5px;">' + flags_html + '</div></div>',
                unsafe_allow_html=True,
            )

        # Confidence note
        conf_note = llm.get("confidence_note", "")
        if conf_note:
            st.markdown(
                '<div style="border-top:1px solid #1e3a50;padding-top:8px;margin-top:4px;">'
                '<span style="color:#639922;font-size:0.72rem;font-weight:700;letter-spacing:0.8px;">CONFIDENCE NOTE</span><br>'
                f'<span style="color:#aaa;font-size:0.82rem;font-style:italic;">{conf_note}</span>'
                '</div>',
                unsafe_allow_html=True,
            )

        # Error note if Groq returned an error string
        if llm.get("error"):
            st.markdown(
                f'<div style="color:#E24B4A;font-size:0.78rem;margin-top:6px;">⚠ {llm["error"]}</div>',
                unsafe_allow_html=True,
            )

        st.markdown('</div>', unsafe_allow_html=True)
        return

    # ── State 2: escalation patient, enrichment not yet arrived ─
    level       = r.get("predicted_acuity", 5)
    is_rail     = r.get("safety_rail_triggered", False)
    is_conflict = r.get("has_feature_conflict", False)
    true_ac     = r.get("true_acuity")
    pred_ac     = r.get("predicted_acuity")
    is_escalation = (
        level in (1, 2) or
        is_rail or
        is_conflict or
        (true_ac is not None and true_ac != pred_ac)
    )

    if is_escalation:
        st.markdown(
            '<div style="background:#1a1a0d;border-left:4px solid #EF9F27;'
            'border-radius:6px;padding:10px 16px;margin-top:10px;">'
            '<span style="color:#EF9F27;font-size:0.72rem;font-weight:700;letter-spacing:1px;">'
            '⏳ LLM REVIEW IN PROGRESS</span><br>'
            '<span style="color:#666;font-size:0.80rem;">'
            'Consumer is generating Groq explanation — will appear on next refresh.</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        return

    # ── State 3: routine patient ────────────────────────────────
    # st.markdown(
    #     '<div style="background:#0d1f0d;border-left:4px solid #2ecc71;'
    #     'border-radius:6px;padding:10px 16px;margin-top:10px;">'
    #     '<span style="color:#2ecc71;font-size:0.72rem;font-weight:700;letter-spacing:1px;">'
    #     '✅ ROUTINE — NO LLM REVIEW REQUIRED</span><br>'
    #     '<span style="color:#555;font-size:0.80rem;">'
    #     'KTAS 3–5, vitals within range, no acuity conflict — Groq API not called.</span>'
    #     '</div>',
    #     unsafe_allow_html=True,
    # )


# ─────────────────────────────────────────────────────────────
#  GUARDRAIL RENDERER
# ─────────────────────────────────────────────────────────────

def render_guardrails(r: dict):
    flags        = r.get("intervention_flags", [])
    rail_on      = r.get("safety_rail_triggered", False)
    rail_reasons = r.get("safety_rail_reasons", [])
    conf_state   = r.get("confidence_state", "UNKNOWN")
    conf_gap     = r.get("confidence_gap")
    conf_top2    = r.get("confidence_top2_acuity")
    conf_top2pct = r.get("confidence_top2_pct")
    conflicts    = r.get("conflict_reasons", [])
    needs_review = r.get("requires_review", False)
    orig_acuity  = r.get("original_acuity")
    pred_acuity  = r.get("predicted_acuity")
    safety_notes = r.get("safety_notes", [])

    if not flags and conf_state in ("HIGH", "MODERATE", "UNKNOWN"):
        st.markdown(
            '<div style="background:#0d1f0d;border-left:4px solid #639922;border-radius:0 6px 6px 0;'
            'padding:6px 14px;margin-top:6px;font-size:0.78rem;color:#639922;">'
            '&#10003; No guardrails triggered &mdash; '
            f'confidence {conf_state.lower()}'
            + (f' ({r.get("confidence_pct","")}, gap={conf_gap:.2f})' if conf_gap is not None else '')
            + '</div>',
            unsafe_allow_html=True,
        )
        return

    badge_html = ""
    if rail_on:
        badge_html += '<span style="background:#FAECE7;color:#993C1D;padding:2px 9px;border-radius:20px;font-size:0.71rem;font-weight:600;margin-right:5px;">&#9650; Rail</span>'
    if conf_state == "AMBIGUOUS":
        badge_html += '<span style="background:#FAEEDA;color:#854F0B;padding:2px 9px;border-radius:20px;font-size:0.71rem;font-weight:600;margin-right:5px;">&#8776; Ambiguous</span>'
    elif conf_state == "LOW":
        badge_html += '<span style="background:#FCEBEB;color:#A32D2D;padding:2px 9px;border-radius:20px;font-size:0.71rem;font-weight:600;margin-right:5px;">&#33; Low conf</span>'
    if conflicts:
        badge_html += '<span style="background:#EEEDFE;color:#3C3489;padding:2px 9px;border-radius:20px;font-size:0.71rem;font-weight:600;margin-right:5px;">&#9741; Conflict</span>'
    if needs_review:
        badge_html += '<span style="background:#FAEEDA;color:#633806;padding:2px 9px;border-radius:20px;font-size:0.71rem;font-weight:600;">&#128065; Review</span>'

    escalation_html = ""
    if orig_acuity and pred_acuity and orig_acuity != pred_acuity:
        escalation_html = (
            f'<span style="margin-left:auto;font-size:0.78rem;color:#E24B4A;font-weight:600;">'
            f'KTAS-{orig_acuity} &rarr; KTAS-{pred_acuity}</span>'
        )

    body_html = ""

    if rail_on and rail_reasons:
        body_html += (
            '<div style="margin-top:10px;padding-top:10px;border-top:1px solid #1a2a1a;">'
            '<div style="color:#E24B4A;font-size:0.70rem;font-weight:700;letter-spacing:.6px;margin-bottom:5px;">SAFETY RAILS</div>'
        )
        for reason in rail_reasons:
            body_html += (
                f'<div style="display:flex;gap:7px;align-items:flex-start;margin-bottom:3px;">'
                f'<span style="color:#D85A30;margin-top:4px;font-size:9px;">&#9679;</span>'
                f'<span style="color:#ccc;font-size:0.80rem;">{reason}</span>'
                f'</div>'
            )
        body_html += '</div>'

    if conf_state in ("AMBIGUOUS", "LOW"):
        conf_color = "#BA7517" if conf_state == "AMBIGUOUS" else "#A32D2D"
        body_html += (
            f'<div style="margin-top:10px;padding-top:10px;border-top:1px solid #1a2a1a;">'
            f'<div style="color:{conf_color};font-size:0.70rem;font-weight:700;letter-spacing:.6px;margin-bottom:5px;">'
            f'CONFIDENCE &mdash; {conf_state}</div>'
        )
        if conf_top2 and conf_top2pct:
            body_html += (
                f'<div style="display:flex;gap:7px;align-items:flex-start;">'
                f'<span style="color:{conf_color};margin-top:4px;font-size:9px;">&#9679;</span>'
                f'<span style="color:#ccc;font-size:0.80rem;">'
                f'Top prediction {r.get("confidence_pct","")} vs KTAS-{conf_top2} at {conf_top2pct}'
                + (f', gap={conf_gap:.3f}' if conf_gap is not None else '')
                + '</span></div>'
            )
        body_html += '</div>'

    if conflicts:
        body_html += (
            '<div style="margin-top:10px;padding-top:10px;border-top:1px solid #1a2a1a;">'
            '<div style="color:#7F77DD;font-size:0.70rem;font-weight:700;letter-spacing:.6px;margin-bottom:5px;">FEATURE CONFLICTS</div>'
        )
        for c in conflicts:
            body_html += (
                f'<div style="display:flex;gap:7px;align-items:flex-start;margin-bottom:3px;">'
                f'<span style="color:#7F77DD;margin-top:4px;font-size:9px;">&#9679;</span>'
                f'<span style="color:#ccc;font-size:0.80rem;">{c}</span>'
                f'</div>'
            )
        body_html += '</div>'

    if orig_acuity and pred_acuity and orig_acuity != pred_acuity:
        body_html += (
            f'<div style="margin-top:10px;background:#1a0a0a;border-radius:5px;padding:6px 10px;'
            f'font-size:0.78rem;color:#E24B4A;">'
            f'Acuity escalated: KTAS-{orig_acuity} &rarr; KTAS-{pred_acuity}</div>'
        )
    elif safety_notes:
        body_html += (
            f'<div style="margin-top:10px;background:#111;border-radius:5px;padding:6px 10px;'
            f'font-size:0.77rem;color:#666;">{safety_notes[-1]}</div>'
        )

    st.markdown(
        '<div style="background:#0d1a0d;border-left:4px solid #E24B4A;'
        'border-radius:0 6px 6px 0;padding:10px 14px;margin-top:6px;">'
        f'<div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">'
        f'<span style="font-size:0.72rem;font-weight:700;color:#E24B4A;letter-spacing:.8px;">&#9888; GUARDRAILS</span>'
        f'{badge_html}{escalation_html}</div>'
        f'{body_html}'
        f'</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
#  PATIENT PANEL  — single definition, all logic here
# ─────────────────────────────────────────────────────────────

def render_patient_panel(r: dict):
    pid = r.get("patient_id", "?")

    # Bug fix: accepted patients are removed from view entirely
    if pid in st.session_state.accepted_ids:
        return

    level     = r.get("predicted_acuity", 5)
    color     = KTAS_COLORS.get(level, "#888")
    icon      = KTAS_ICONS.get(level, "?")
    label     = KTAS_LABEL.get(level, "Unknown")
    conf      = _v(r.get("confidence_pct"))
    complaint = _v(r.get("chief_complaint") or r.get("chief_complaint_raw"))
    age       = _v(r.get("age"))
    sex       = _v(r.get("sex"))
    ts        = r.get("timestamp", "")[:19].replace("T", " ")
    decision  = st.session_state.decisions.get(pid)

    vitals = r.get("vitals", {})
    hr     = _v(vitals.get("heart_rate"))
    sbp    = _v(vitals.get("systolic_bp"))
    dbp    = _v(vitals.get("diastolic_bp"))
    spo2   = _v(vitals.get("spo2"))
    rr     = _v(vitals.get("respiratory_rate"))
    pain   = _v(vitals.get("pain_score"))
    news2  = _v(vitals.get("news2_score"))

    hr_c   = _vital_color(vitals.get("heart_rate"),       60,  100)
    sbp_c  = _vital_color(vitals.get("systolic_bp"),      90,  180)
    spo2_c = _vital_color(vitals.get("spo2"),             94,  None)
    rr_c   = _vital_color(vitals.get("respiratory_rate"), 12,  20)

    # Overridden cards stay visible with orange border
    if decision == "overridden":
        border = "#e67e22"
        badge  = ('<span style="background:#e67e22;color:#000;padding:2px 10px;'
                  'border-radius:10px;font-size:0.77rem;font-weight:700;">&#9998; Overridden</span>')
    else:
        border = color
        badge  = ""

    st.markdown(
        f'<div style="border-left:5px solid {border};background:#1a1a2e;'
        f'border-radius:8px;padding:14px 18px 10px 18px;margin-bottom:4px;">'

        f'<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;">'
        f'<div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;">'
        f'<span style="font-size:1.05rem;font-weight:700;color:white;">Patient {pid}</span>'
        f'<span style="background:{color};color:white;padding:3px 12px;border-radius:12px;'
        f'font-weight:700;font-size:0.82rem;">{icon} KTAS {level} &mdash; {label}</span>'
        f'{badge}'
        f'</div>'
        f'<span style="color:#888;font-size:0.78rem;">{ts}</span>'
        f'</div>'

        f'<div style="margin-top:10px;">'
        f'<span style="color:#888;font-size:0.72rem;">CHIEF COMPLAINT</span><br>'
        f'<span style="color:#fff;font-size:0.95rem;font-weight:600;">{complaint}</span>'
        f'</div>'

        f'<div style="margin-top:10px;display:flex;flex-wrap:wrap;gap:20px;">'
        f'<div><span style="color:#888;font-size:0.72rem;">PATIENT</span><br>'
        f'<span style="color:#ddd;font-size:0.86rem;">{age}y &middot; {sex}</span></div>'
        f'<div><span style="color:#888;font-size:0.72rem;">HEART RATE</span><br>'
        f'<span style="color:{hr_c};font-size:0.86rem;font-weight:600;">{hr} bpm</span></div>'
        f'<div><span style="color:#888;font-size:0.72rem;">BLOOD PRESSURE</span><br>'
        f'<span style="color:{sbp_c};font-size:0.86rem;font-weight:600;">{sbp}/{dbp} mmHg</span></div>'
        f'<div><span style="color:#888;font-size:0.72rem;">SpO2</span><br>'
        f'<span style="color:{spo2_c};font-size:0.86rem;font-weight:600;">{spo2}%</span></div>'
        f'<div><span style="color:#888;font-size:0.72rem;">RESP RATE</span><br>'
        f'<span style="color:{rr_c};font-size:0.86rem;font-weight:600;">{rr}/min</span></div>'
        f'<div><span style="color:#888;font-size:0.72rem;">PAIN</span><br>'
        f'<span style="color:#ddd;font-size:0.86rem;">{pain}/10</span></div>'
        f'<div><span style="color:#888;font-size:0.72rem;">NEWS2</span><br>'
        f'<span style="color:#ddd;font-size:0.86rem;">{news2}</span></div>'
        f'<div><span style="color:#888;font-size:0.72rem;">CONFIDENCE</span><br>'
        f'<span style="color:#ddd;font-size:0.86rem;">{conf}</span></div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Guardrail summary ──────────────────────────────────────
    render_guardrails(r)

    # ── LLM explanation — pre-computed by consumer ─────────────
    render_llm_explanation(r)

    # ── Action buttons — only shown while pending ──────────────
    if decision is None:
        b1, b2, _ = st.columns([1, 1, 6])
        with b1:
            if st.button("✅ Accept", key=f"accept_{pid}"):
                rails_fired       = r.get("safety_rail_triggered", False)
                conf_issue        = r.get("confidence_state") in ("AMBIGUOUS", "LOW")
                conflict          = r.get("has_feature_conflict", False)
                guardrail_changed = r.get("original_acuity") != r.get("predicted_acuity")

                fb_type = (
                    "guardrail_accepted"
                    if (guardrail_changed or rails_fired or conf_issue or conflict)
                    else "accepted"
                )

                publish_feedback(
                    patient=r,
                    feedback_type=fb_type,
                    final_label=r.get("predicted_acuity"),
                )

                if pid not in st.session_state.accepted_ids:
                    st.session_state.accepted_ids.append(pid)
                st.session_state.accepted_count += 1
                st.rerun()

        with b2:
            if st.button("✏️ Override", key=f"override_{pid}"):
                st.session_state.override_open[pid] = True
                st.rerun()

    # ── Override form ──────────────────────────────────────────
    if st.session_state.override_open.get(pid) and decision is None:
        st.markdown(
            '<div style="background:#1e1200;border-left:3px solid #e67e22;'
            'border-radius:0 6px 6px 0;padding:12px 16px;margin-top:8px;">',
            unsafe_allow_html=True,
        )

        ktas_options = {
            1: "KTAS 1 – Resuscitation",
            2: "KTAS 2 – Emergent",
            3: "KTAS 3 – Urgent",
            4: "KTAS 4 – Less Urgent",
            5: "KTAS 5 – Non-Urgent",
        }
        col_sel, col_note = st.columns([1, 3])
        with col_sel:
            override_ktas = st.selectbox(
                "Correct KTAS level",
                options=list(ktas_options.keys()),
                format_func=lambda k: ktas_options[k],
                index=level - 1,
                key=f"ktas_sel_{pid}",
            )
            st.session_state.override_ktas[pid] = override_ktas

        with col_note:
            note = st.text_area(
                "Override reason (required)",
                key=f"override_area_{pid}",
                placeholder=(
                    "e.g. Patient visibly deteriorating — escalating to KTAS 2.\n"
                    "Include any clinical observations not captured in vitals."
                ),
                height=90,
            )

        st.markdown('</div>', unsafe_allow_html=True)

        s1, s2, _ = st.columns([1, 1, 6])
        with s1:
            if st.button("✅ Submit override", key=f"submit_{pid}"):
                if note.strip():
                    chosen_ktas = st.session_state.override_ktas.get(pid, level)
                    publish_feedback(
                        patient=r,
                        feedback_type="override",
                        final_label=chosen_ktas,
                        override_reason=note.strip(),
                        override_ktas=chosen_ktas,
                    )
                    st.session_state.override_text[pid] = note.strip()
                    st.session_state.decisions[pid]     = "overridden"
                    st.session_state.override_open[pid] = False
                    st.rerun()
                else:
                    st.warning("Please enter a reason before submitting.")

        with s2:
            if st.button("Cancel", key=f"cancel_{pid}"):
                st.session_state.override_open[pid] = False
                st.rerun()

    # ── Override note (shown after submission) ─────────────────
    if decision == "overridden" and pid in st.session_state.override_text:
        chosen_ktas = st.session_state.override_ktas.get(pid, "?")
        st.markdown(
            f'<div style="background:#2a1f0e;border-left:3px solid #e67e22;'
            f'padding:8px 14px;border-radius:0 4px 4px 0;margin-top:6px;">'
            f'<span style="color:#EF9F27;font-size:0.72rem;font-weight:700;letter-spacing:.6px;">OVERRIDE</span><br>'
            f'<span style="color:#f0a500;font-size:0.84rem;">'
            f'Corrected to KTAS {chosen_ktas} &mdash; '
            + st.session_state.override_text[pid] +
            f'</span><br>'
            f'<span style="color:#666;font-size:0.75rem;">Saved to retraining queue</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<div style='margin-bottom:16px;'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  DASHBOARD
# ─────────────────────────────────────────────────────────────

def render_dashboard(enrich_topic: str):
    st.markdown(
        '<h1 style="color:#E24B4A;margin-bottom:2px;">&#127973; KTAS Emergency Triage</h1>'
        '<p style="color:#666;font-size:0.85rem;margin-top:0;">'
        'Live predictions via Kafka &mdash; auto-refreshes &mdash; '
        '&#129504; AI explanations pre-computed by consumer (escalations only)</p>',
        unsafe_allow_html=True,
    )

    # Fix: st.components.v1.html deprecated → use st.html
    st.html(
        "<script>window.parent.document.querySelector('section.main').scrollTo("
        "0, window.parent.document.querySelector('section.main').scrollHeight);</script>"
    )

    store   = st.session_state.store
    records = store.all()
    total   = len(records)

    decisions  = st.session_state.decisions
    accepted   = st.session_state.accepted_count
    overridden = sum(1 for d in decisions.values() if d == "overridden")
    pending    = total - accepted - overridden
    emergency  = sum(1 for r in records if r.get("is_emergency"))
    evaluated  = [(r["true_acuity"], r["predicted_acuity"])
                  for r in records if r.get("true_acuity") is not None]
    accuracy   = (sum(1 for t, p in evaluated if t == p) / len(evaluated) * 100
                  if evaluated else 0)
    enriched   = sum(1 for r in records if r.get("llm_explanation", {}).get("is_ready"))

    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    m1.metric("Patients Seen",  total)
    m2.metric("Emergencies",    emergency)
    m3.metric("Pending Review", pending)
    m4.metric("Accepted",       accepted)
    m5.metric("Overridden",     overridden)
    m6.metric("Model Accuracy", f"{accuracy:.1f}%")
    m7.metric("LLM Enriched",   enriched)

    st.markdown("---")

    with st.sidebar:
        st.markdown("### 🔍 Filters")
        emergency_only = st.checkbox("Emergencies only", value=False)
        pending_only   = st.checkbox("Pending only",     value=False)

        st.markdown("---")
        st.markdown("### 🧠 LLM Enrichment")
        st.markdown(
            '<div style="color:#888;font-size:0.78rem;">'
            'Groq is called by the <b>consumer</b>, not the UI.<br><br>'
            'Only escalation patients receive LLM review:<br>'
            '&bull; KTAS 1 or 2 prediction<br>'
            '&bull; Acuity conflict (predicted ≠ true)<br>'
            '&bull; Critical vital sign breach<br>'
            '&bull; Clinician override<br><br>'
            'Set <code>$env:GROQ_API_KEY</code> and run the consumer — '
            'explanations appear here automatically.'
            '</div>',
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown("### ℹ️ KTAS Scale")
        for lvl, lbl in KTAS_LABEL.items():
            st.markdown(
                f'<span style="color:{KTAS_COLORS[lvl]};font-weight:700;">'
                f'{KTAS_ICONS[lvl]} KTAS {lvl}</span> — {lbl}',
                unsafe_allow_html=True,
            )

    if total == 0:
        st.info("⏳ Waiting for patients. Start kafka_producer.py and kafka_consumer.py.")
    else:
        filtered = [r for r in records
                    if r.get("patient_id") not in st.session_state.accepted_ids]

        if emergency_only:
            filtered = [r for r in filtered if r.get("is_emergency")]
        if pending_only:
            filtered = [r for r in filtered if decisions.get(r.get("patient_id")) is None]

        st.markdown(
            f'<p style="color:#888;font-size:0.85rem;">'
            f'Showing {len(filtered)} of {total} patients '
            f'({st.session_state.accepted_count} accepted and cleared)</p>',
            unsafe_allow_html=True,
        )

        for r in filtered:
            render_patient_panel(r)

    # Polling and rerun handled in __main__ triage tab block


# ─────────────────────────────────────────────────────────────
#  METRICS TABS
# ─────────────────────────────────────────────────────────────

KTAS_COLORS_HEX = {1: "#E24B4A", 2: "#EF9F27", 3: "#378ADD", 4: "#639922", 5: "#888780"}

def _no_store_warning():
    st.warning("Metrics DB not connected. Make sure latency_store.py is on the path and the consumer has run.")


def render_latency_tab():
    """
    Tab 2 — Latency
    ───────────────
    Run selector  →  segment bar chart (P50 / P95)  →  per-patient scatter
    """
    store = st.session_state.get("lat_store")
    if store is None:
        _no_store_warning()
        return

    runs = store.get_runs()
    if not runs:
        st.info("No runs recorded yet. Start the producer and consumer.")
        return

    # ── Run selector ──────────────────────────────────────────
    run_labels = {
        r["run_id"]: f"{r['run_id']}  ·  {r['message_count']} msgs"
                     + (f"  ·  P50={r['p50_e2e_ms']:.0f}ms" if r.get("p50_e2e_ms") else "  ·  (live)")
        for r in runs
    }
    selected_id = st.selectbox(
        "Select run",
        options=list(run_labels.keys()),
        format_func=lambda k: run_labels[k],
        key="latency_run_sel",
    )
    run = next(r for r in runs if r["run_id"] == selected_id)

    msgs = store.get_messages(selected_id)
    if not msgs:
        st.info("No messages recorded for this run yet. Make sure kafka_consumer.py is running with --db pointing to the same file.")
        return

    # ── Live aggregation from raw message rows ────────────────
    # This works whether or not finalize_run() has been called,
    # so data shows immediately while the consumer is still running.
    import numpy as _np
    def _p(col, pct):
        vals = [m[col] for m in msgs if m.get(col) is not None]
        return float(_np.percentile(vals, pct)) if vals else None

    live = {
        "message_count":      len(msgs),
        "p50_e2e_ms":         _p("e2e_ms",            50),
        "p95_e2e_ms":         _p("e2e_ms",            95),
        "p50_kafka_ms":       _p("kafka_transit_ms",  50),
        "p95_kafka_ms":       _p("kafka_transit_ms",  95),
        "p50_ml_ms":          _p("ml_inference_ms",   50),
        "p95_ml_ms":          _p("ml_inference_ms",   95),
        "p50_safety_ms":      _p("safety_pipeline_ms",50),
        "p95_safety_ms":      _p("safety_pipeline_ms",95),
        "p50_llm_ms":         _p("llm_ms",            50),
    }

    # Duration and throughput from raw timestamps
    t_vals = [m["produced_at"] for m in msgs if m.get("produced_at")]
    s_vals = [m["safety_done_at"] for m in msgs if m.get("safety_done_at")]
    if t_vals and s_vals:
        _dur = max(s_vals) - min(t_vals)
        live["duration_secs"]       = _dur
        live["throughput_msg_sec"]  = len(msgs) / _dur if _dur > 0 else None
    else:
        live["duration_secs"]       = run.get("duration_secs")
        live["throughput_msg_sec"]  = run.get("throughput_msg_sec")

    # ── Summary chips ─────────────────────────────────────────
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Messages",        live["message_count"])
    c2.metric("Duration",        f"{live['duration_secs']:.1f}s"       if live.get("duration_secs")      else "—")
    c3.metric("Throughput",      f"{live['throughput_msg_sec']:.2f} msg/s" if live.get("throughput_msg_sec") else "—")
    c4.metric("P50 e2e",         f"{live['p50_e2e_ms']:.0f} ms"        if live.get("p50_e2e_ms")         else "—")
    c5.metric("P95 e2e",         f"{live['p95_e2e_ms']:.0f} ms"        if live.get("p95_e2e_ms")         else "—")
    c6.metric("LLM P50 (async)", f"{live['p50_llm_ms']:.0f} ms"        if live.get("p50_llm_ms")         else "—")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    # ── LEFT: Segment breakdown bar chart ─────────────────────
    with col_left:
        st.markdown(
            '<p style="color:#aaa;font-size:0.78rem;font-weight:700;letter-spacing:.8px;">PIPELINE SEGMENT LATENCY</p>',
            unsafe_allow_html=True,
        )

        segments = ["Kafka Transit", "ML Inference", "Safety Pipeline"]
        colors   = ["#378ADD", "#EF9F27", "#E24B4A"]

        p50_vals = [live.get("p50_kafka_ms") or 0, live.get("p50_ml_ms") or 0, live.get("p50_safety_ms") or 0]
        p95_vals = [live.get("p95_kafka_ms") or 0, live.get("p95_ml_ms") or 0, live.get("p95_safety_ms") or 0]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="P50 (median)",
            x=segments, y=p50_vals,
            marker_color=colors,
            opacity=0.9,
            text=[f"{v:.0f}ms" for v in p50_vals],
            textposition="outside",
        ))
        fig.add_trace(go.Bar(
            name="P95",
            x=segments, y=p95_vals,
            marker_color=colors,
            opacity=0.45,
            text=[f"{v:.0f}ms" for v in p95_vals],
            textposition="outside",
        ))
        fig.update_layout(
            barmode="group",
            paper_bgcolor="#13131f",
            plot_bgcolor="#13131f",
            font=dict(color="#e0e0e0", size=12),
            legend=dict(orientation="h", y=1.12, x=0),
            yaxis=dict(title="ms", gridcolor="#222"),
            xaxis=dict(gridcolor="#222"),
            margin=dict(t=30, b=20, l=0, r=0),
            height=320,
        )

        # LLM on secondary axis note
        llm_p50 = live.get("p50_llm_ms")
        llm_p95 = live.get("p95_llm_ms")
        if llm_p50 is not None and llm_p95 is not None:
            st.markdown(
                f'<p style="color:#555;font-size:0.75rem;margin-bottom:4px;">'
                f'LLM explanation (async, excluded from e2e) — P50: {llm_p50:.0f}ms · P95: {llm_p95:.0f}ms</p>',
                unsafe_allow_html=True,
            )

        st.plotly_chart(fig, width='stretch')

    # ── RIGHT: Per-patient scatter ────────────────────────────
    with col_right:
        st.markdown(
            '<p style="color:#aaa;font-size:0.78rem;font-weight:700;letter-spacing:.8px;">E2E LATENCY PER PATIENT</p>',
            unsafe_allow_html=True,
        )

        df = pd.DataFrame(msgs)
        df["seq"]   = range(1, len(df) + 1)
        df["rail"]  = df["safety_rail_triggered"].apply(lambda x: "Rail fired" if x else "No rail")
        df["ktas"]  = df["predicted_acuity"].apply(lambda x: f"KTAS-{x}" if x else "Unknown")
        df["color"] = df["predicted_acuity"].apply(
            lambda x: KTAS_COLORS_HEX.get(x, "#888") if x else "#888"
        )
        df["label"] = df.apply(
            lambda row: (
                f"Patient: {row['patient_id']}<br>"
                f"KTAS: {row.get('predicted_acuity','?')}<br>"
                f"e2e: {row['e2e_ms']:.0f}ms<br>"
                f"Rail: {'Yes' if row['safety_rail_triggered'] else 'No'}"
            ), axis=1
        )

        fig2 = go.Figure()
        for ktas_val, grp in df.groupby("predicted_acuity", dropna=False):
            color = KTAS_COLORS_HEX.get(ktas_val, "#888") if ktas_val else "#888"
            fig2.add_trace(go.Scatter(
                x=grp["seq"],
                y=grp["e2e_ms"],
                mode="markers",
                name=f"KTAS-{ktas_val}" if ktas_val else "Unknown",
                marker=dict(color=color, size=8, opacity=0.85,
                            symbol=["diamond" if r else "circle"
                                    for r in grp["safety_rail_triggered"]]),
                hovertemplate="%{text}<extra></extra>",
                text=grp["label"],
            ))

        # Reference lines
        if run.get("p50_e2e_ms"):
            fig2.add_hline(y=run["p50_e2e_ms"], line_dash="dot",
                          line_color="#639922", opacity=0.6,
                          annotation_text="P50", annotation_font_color="#639922")
        if run.get("p95_e2e_ms"):
            fig2.add_hline(y=run["p95_e2e_ms"], line_dash="dot",
                          line_color="#E24B4A", opacity=0.6,
                          annotation_text="P95", annotation_font_color="#E24B4A")

        fig2.update_layout(
            paper_bgcolor="#13131f",
            plot_bgcolor="#13131f",
            font=dict(color="#e0e0e0", size=12),
            legend=dict(orientation="h", y=1.12, x=0),
            yaxis=dict(title="e2e latency (ms)", gridcolor="#222"),
            xaxis=dict(title="Patient sequence", gridcolor="#222"),
            margin=dict(t=30, b=20, l=0, r=0),
            height=320,
        )
        st.markdown(
            '<p style="color:#555;font-size:0.75rem;margin-bottom:4px;">'
            'Diamond markers = safety rail fired · Hover for patient detail</p>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig2, width='stretch')


def render_throughput_tab():
    """
    Tab 3 — Throughput
    ──────────────────
    One row per run — ablation table comparing runs side by side.
    """
    store = st.session_state.get("lat_store")
    if store is None:
        _no_store_warning()
        return

    runs = store.get_runs()
    if not runs:
        st.info("No runs recorded yet. Run the producer and consumer.")
        return

    st.markdown(
        '<p style="color:#aaa;font-size:0.78rem;font-weight:700;letter-spacing:.8px;">RUN COMPARISON TABLE</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#555;font-size:0.75rem;margin-top:-8px;">'
        'Each row is one producer run. Use --run-id to label ablation experiments.</p>',
        unsafe_allow_html=True,
    )

    import numpy as _np2

    def _fmt(val, suffix="", decimals=1):
        if val is None:
            return "—"
        return f"{val:.{decimals}f}{suffix}"

    def _live_agg(run_id):
        """Compute live aggregates from raw message rows for any run."""
        msgs = store.get_messages(run_id)
        if not msgs:
            return {}
        def _p(col, pct):
            vals = [m[col] for m in msgs if m.get(col) is not None]
            return float(_np2.percentile(vals, pct)) if vals else None
        t_vals = [m["produced_at"] for m in msgs if m.get("produced_at")]
        s_vals = [m["safety_done_at"] for m in msgs if m.get("safety_done_at")]
        dur    = (max(s_vals) - min(t_vals)) if t_vals and s_vals else None
        rail_hits = sum(1 for m in msgs if m.get("safety_rail_triggered"))
        return {
            "n":          len(msgs),
            "duration":   dur,
            "throughput": len(msgs) / dur if dur else None,
            "p50_e2e":    _p("e2e_ms",             50),
            "p95_e2e":    _p("e2e_ms",             95),
            "p50_kafka":  _p("kafka_transit_ms",   50),
            "p50_ml":     _p("ml_inference_ms",    50),
            "p50_safety": _p("safety_pipeline_ms", 50),
            "rail_rate":  rail_hits / len(msgs) * 100 if msgs else None,
        }

    rows = []
    for r in runs:
        a = _live_agg(r["run_id"])
        rows.append({
            "Run ID":          r["run_id"],
            "Started":         r.get("started_at", "")[:16].replace("T", " "),
            "Messages":        a.get("n", r.get("message_count", 0)),
            "Duration (s)":    _fmt(a.get("duration"), "s"),
            "Throughput":      _fmt(a.get("throughput"), " msg/s", 2),
            "P50 e2e (ms)":    _fmt(a.get("p50_e2e"), "ms", 0),
            "P95 e2e (ms)":    _fmt(a.get("p95_e2e"), "ms", 0),
            "P50 Kafka (ms)":  _fmt(a.get("p50_kafka"), "ms", 0),
            "P50 ML (ms)":     _fmt(a.get("p50_ml"), "ms", 0),
            "P50 Safety (ms)": _fmt(a.get("p50_safety"), "ms", 0),
            "Rail hit rate":   _fmt(a.get("rail_rate"), "%"),
            "Status":          "✓ Finalized" if r.get("finalized") else "⏳ Live",
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        width='stretch',
        hide_index=True,
        column_config={
            "Run ID":      st.column_config.TextColumn(width="medium"),
            "Started":     st.column_config.TextColumn(width="small"),
            "Status":      st.column_config.TextColumn(width="small"),
        },
    )

    # Throughput sparkline across runs (if >1 run)
    # Build sparkline data from live aggregates (already computed in rows above)
    spark_data = [(r["Run ID"], r["Throughput"]) for r in rows if r["Throughput"] != "—"]
    if len(spark_data) > 1:
        spark_x  = [d[0] for d in spark_data]
        spark_y  = [float(d[1].replace(" msg/s","")) for d in spark_data]
        st.markdown("---")
        st.markdown(
            '<p style="color:#aaa;font-size:0.78rem;font-weight:700;letter-spacing:.8px;">THROUGHPUT ACROSS RUNS</p>',
            unsafe_allow_html=True,
        )
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=spark_x,
            y=spark_y,
            marker_color="#378ADD",
            text=[f"{v:.2f}" for v in spark_y],
            textposition="outside",
        ))
        fig.update_layout(
            paper_bgcolor="#13131f",
            plot_bgcolor="#13131f",
            font=dict(color="#e0e0e0", size=11),
            yaxis=dict(title="msg/sec", gridcolor="#222"),
            xaxis=dict(tickangle=-30),
            margin=dict(t=20, b=60, l=0, r=0),
            height=260,
            showlegend=False,
        )
        st.plotly_chart(fig, width='stretch')
    elif len(runs) == 1:
        st.markdown(
            '<p style="color:#555;font-size:0.78rem;">Run more producer runs to compare throughput across experiments.</p>',
            unsafe_allow_html=True,
        )


def render_auditability_tab():
    """
    Tab 4 — Auditability
    ─────────────────────
    Override confusion matrix  +  safety rail reason ranked list.
    """
    store = st.session_state.get("lat_store")
    if store is None:
        _no_store_warning()
        return

    matrix  = store.get_override_matrix()
    reasons = store.get_rail_reasons()

    if not matrix and not reasons:
        st.info("No clinician feedback recorded yet. Accept or override patients in the Triage tab.")
        return

    col_left, col_right = st.columns([3, 2])

    # ── LEFT: Override confusion matrix ──────────────────────
    with col_left:
        st.markdown(
            '<p style="color:#aaa;font-size:0.78rem;font-weight:700;letter-spacing:.8px;">CLINICIAN OVERRIDE MATRIX</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="color:#555;font-size:0.75rem;margin-top:-8px;">'
            'Rows = model prediction · Columns = clinician final label · '
            'Diagonal = accepted · Off-diagonal = corrected</p>',
            unsafe_allow_html=True,
        )

        ktas_levels = [1, 2, 3, 4, 5]
        z = [[matrix.get(pred, {}).get(label, 0) for label in ktas_levels]
             for pred in ktas_levels]

        # Custom text — show count, blank for zeros
        text = [[str(val) if val > 0 else "" for val in row] for row in z]

        # Color scale: diagonal (correct) green, off-diagonal red
        # Build a custom colorscale
        fig = go.Figure(go.Heatmap(
            z=z,
            x=[f"Label KTAS-{l}" for l in ktas_levels],
            y=[f"Pred KTAS-{p}" for p in ktas_levels],
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=16, color="white"),
            colorscale=[
                [0.0, "#13131f"],
                [0.5, "#1a4a2e"],
                [1.0, "#2ecc71"],
            ],
            showscale=False,
            hoverongaps=False,
            hovertemplate="Model predicted: %{y}<br>Clinician label: %{x}<br>Count: %{z}<extra></extra>",
        ))

        # Overlay diagonal cells with a different color to highlight correct predictions
        for i, ktas in enumerate(ktas_levels):
            val = matrix.get(ktas, {}).get(ktas, 0)
            if val > 0:
                fig.add_shape(
                    type="rect",
                    x0=i - 0.5, x1=i + 0.5,
                    y0=i - 0.5, y1=i + 0.5,
                    line=dict(color="#2ecc71", width=2),
                    fillcolor="rgba(46,204,113,0.15)",
                )

        fig.update_layout(
            paper_bgcolor="#13131f",
            plot_bgcolor="#13131f",
            font=dict(color="#e0e0e0", size=12),
            margin=dict(t=10, b=10, l=0, r=0),
            height=340,
            xaxis=dict(side="bottom"),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, width='stretch')

        # Summary line
        total_decisions = sum(sum(row.values()) for row in matrix.values())
        diagonal        = sum(matrix.get(k, {}).get(k, 0) for k in ktas_levels)
        off_diag        = total_decisions - diagonal
        if total_decisions > 0:
            st.markdown(
                f'<p style="color:#555;font-size:0.78rem;">'
                f'Total decisions: {total_decisions} · '
                f'Accepted (diagonal): {diagonal} ({diagonal/total_decisions*100:.0f}%) · '
                f'Corrected (off-diagonal): {off_diag} ({off_diag/total_decisions*100:.0f}%)</p>',
                unsafe_allow_html=True,
            )

    # ── RIGHT: Safety rail reason ranked list ────────────────
    with col_right:
        st.markdown(
            '<p style="color:#aaa;font-size:0.78rem;font-weight:700;letter-spacing:.8px;">SAFETY RAIL TRIGGERS</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p style="color:#555;font-size:0.75rem;margin-top:-8px;">'
            'Ranked by frequency across all overridden/accepted patients</p>',
            unsafe_allow_html=True,
        )

        if not reasons:
            st.markdown('<p style="color:#555;font-size:0.84rem;">No rail triggers recorded yet.</p>',
                        unsafe_allow_html=True)
        else:
            max_count = reasons[0][1] if reasons else 1
            for reason, count in reasons:
                pct_bar = int(count / max_count * 100)
                st.markdown(
                    f'<div style="margin-bottom:10px;">'
                    f'<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
                    f'<span style="color:#ddd;font-size:0.82rem;">{reason}</span>'
                    f'<span style="color:#EF9F27;font-size:0.82rem;font-weight:700;">{count}</span>'
                    f'</div>'
                    f'<div style="background:#1a1a2e;border-radius:3px;height:6px;">'
                    f'<div style="background:#EF9F27;width:{pct_bar}%;height:6px;border-radius:3px;"></div>'
                    f'</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--broker",       default="localhost:9092")
    p.add_argument("--topic",        default="triage-output")
    p.add_argument("--enrich-topic", default="triage-enrichment",
                   help="Topic consumer publishes LLM enrichment to (must match --enrich-topic on consumer)")
    p.add_argument("--db",           default="triage_latency.db",
                   help="Path to shared latency/metrics SQLite DB (default: triage_latency.db)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    st.set_page_config(
        page_title="KTAS Triage",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

    init(args.broker, args.topic, args.enrich_topic, args.db)

    tab_triage, tab_latency, tab_throughput, tab_audit = st.tabs([
        "🏥 Triage",
        "⏱ Latency",
        "📊 Throughput",
        "🔍 Auditability",
    ])

    with tab_triage:
        render_dashboard(args.enrich_topic)

    with tab_latency:
        render_latency_tab()

    with tab_throughput:
        render_throughput_tab()

    with tab_audit:
        render_auditability_tab()

    # ── Polling loop — outside all tab blocks so it always runs ──────────────
    # Streamlit executes __main__ top-to-bottom on every rerun regardless of
    # which tab is active. Placing st.rerun() here means the page refreshes
    # on new Kafka data no matter which tab the user is viewing.
    _store    = st.session_state.store
    _deadline = time.time() + 10
    while time.time() < _deadline:
        time.sleep(0.3)
        _has_new = _store.consume_dirty()
        _has_pending = any(
            not r.get("llm_explanation", {}).get("is_ready")
            and (
                r.get("predicted_acuity") in (1, 2)
                or r.get("safety_rail_triggered")
                or r.get("has_feature_conflict")
                or (r.get("true_acuity") is not None
                    and r.get("true_acuity") != r.get("predicted_acuity"))
            )
            for r in _store.all()
            if r.get("patient_id") not in st.session_state.accepted_ids
        )
        if _has_new or _has_pending:
            break
    st.rerun()