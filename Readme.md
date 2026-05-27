# 🏥 KTAS Emergency Triage System

A real-time emergency department triage pipeline that streams synthetic patient records through Kafka, classifies acuity with XGBoost, applies a three-layer safety guardrail system, and generates LLM explanations for high-risk cases — all visualised on a live Streamlit dashboard.

---

## Architecture

```
kafka_producer.py          →   Kafka (triage-input)
                                    ↓
kafka_consumer.py          →   XGBoost + 3-layer safety pipeline
                                    ↓                        ↓
                           Kafka (triage-output)   Kafka (triage-enrichment)
                                    ↓                        ↓
                           streamlit_app.py  ←←←←←←←←←←←←←←
                                    ↓
                           Clinician accepts / overrides
                                    ↓
                           Kafka (triage-feedback)
                                    ↓
                           feedback_consumer.py  →  SQLite (retraining queue)
```

---

## Safety Pipeline (3 Layers)

Every patient prediction passes through three layers before reaching the dashboard:

**Layer 1 — Deterministic Safety Rails**
Hard vital-sign and keyword rules that can only raise acuity, never lower it. Examples: SpO2 < 90% → KTAS-1 floor, "cardiac arrest" keyword → KTAS-1 forced, GCS ≤ 8 → KTAS-2 floor.

**Layer 2 — Confidence Classification**
Top-1 probability and gap-to-second-best classify predictions as HIGH / MODERATE / AMBIGUOUS / LOW. Low-confidence cases are flagged for review.

**Layer 3 — Feature Conflict Detection**
Cross-checks vitals, chief complaint, and medical history for inconsistencies — e.g. a benign complaint paired with critical vitals, or high-risk comorbidities with a low acuity prediction.

---

## Model Results

Three models were evaluated on merged vitals, TF-IDF chief complaint, and 25 binary comorbidity flags:

| Model | Accuracy |
|---|---|
| Logistic Regression | 0.72 |
| Random Forest | 0.79 |
| **XGBoost** | **0.83** |

- **Emergency detection** (KTAS 1–2 vs 3–5): ~0.94 accuracy
- **Macro AUC** (one-vs-rest): ~0.91

---

## Pipeline Performance

Measured across 10 runs (20–200 patients, 100ms–2s intervals):

| Metric | Baseline (2s) | Stress (100ms) |
|---|---|---|
| Throughput | 0.50 msg/s | 3.94 msg/s |
| P50 end-to-end latency | 82ms | 199ms |
| P95 end-to-end latency | 136ms | 309ms |
| P50 Kafka transit | 55ms | 140ms |
| P50 ML inference | 23ms | 52ms |
| P50 safety pipeline | <1ms | <1ms |
| Safety rail hit rate | 10–23% | 10–23% |

Safety pipeline adds negligible latency (<1ms) regardless of throughput — deterministic rules run in-process with no I/O.

---

## Live Demo

**[View on Streamlit →](YOUR_STREAMLIT_URL)**
Latency, Throughput, and Auditability tabs are populated from a pre-recorded run. The Triage tab requires a live Kafka connection.

📹 **[Full system demo (YouTube) →](https://youtu.be/MDKxKKQBEN4?si=eE72GuH_LdN1Iz3l)**

📝 **[Medium writeup →](https://medium.com/@chaudharysaumya847/ktas-triage-detection-with-explainer-ai-f581918ab1ee)**

📝 **[Streamlit URL →](https://triageaiexplainer.streamlit.app/)**