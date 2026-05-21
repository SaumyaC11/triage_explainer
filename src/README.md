# KTAS Emergency Triage — Kafka + Streamlit Pipeline

Real-time triage prediction pipeline:

```
[kafka_producer.py]
      │  synthetic patient data
      ▼
  Kafka topic: triage-input
      │
[kafka_consumer.py]  ←── triage_model.pkl
      │  KTAS predictions + probabilities
      ▼
  Kafka topic: triage-output
      │
[streamlit_app.py]
      │  live dashboard
      ▼
  Browser: http://localhost:8501
```

---

## 1. Prerequisites

### Kafka (Docker — easiest)
```bash
docker run -d --name kafka \
  -p 9092:9092 \
  -e KAFKA_CFG_ADVERTISED_LISTENERS=PLAINTEXT://localhost:9092 \
  -e KAFKA_CFG_ZOOKEEPER_CONNECT="" \
  -e ALLOW_PLAINTEXT_LISTENER=yes \
  bitnami/kafka:latest
```

Or with docker-compose (Kafka + ZooKeeper):
```yaml
# docker-compose.yml
version: "3"
services:
  zookeeper:
    image: bitnami/zookeeper:latest
    environment: { ALLOW_ANONYMOUS_LOGIN: "yes" }
    ports: ["2181:2181"]

  kafka:
    image: bitnami/kafka:latest
    depends_on: [zookeeper]
    ports: ["9092:9092"]
    environment:
      KAFKA_CFG_ZOOKEEPER_CONNECT: zookeeper:2181
      KAFKA_CFG_ADVERTISED_LISTENERS: PLAINTEXT://localhost:9092
      ALLOW_PLAINTEXT_LISTENER: "yes"
```
```bash
docker-compose up -d
```

### Python packages
```bash
pip install kafka-python streamlit plotly pandas numpy scikit-learn xgboost scipy
```

---

## 2. Train and save the model (if not done yet)
```bash
python triage_system.py \
  --csv data.csv \
  --complaints complaints.csv \
  --history history.csv \
  --save-model triage_model.pkl
```
The model bundle saved to `triage_model.pkl` includes the preprocessor,
TF-IDF vectorizer, numeric/categorical column lists, and the trained classifier.

---

## 3. Create Kafka topics (optional — auto-created by default)
```bash
kafka-topics.sh --create --topic triage-input  --bootstrap-server localhost:9092 --partitions 3
kafka-topics.sh --create --topic triage-output --bootstrap-server localhost:9092 --partitions 3
```
```
bin\windows\zookeeper-server-start.bat config\zookeeper.properties

bin/kafka-server-start.sh config/server.properties

bin/kafka-topics.sh --create --topic triage-input  --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
bin/kafka-topics.sh --create --topic triage-output --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1
```
---

## 4. Start all three services (3 separate terminals)

**Terminal 1 — Consumer (loads model, makes predictions)**
```bash
python kafka_consumer.py \
  --broker localhost:9092 \
  --in-topic triage-input \
  --out-topic triage-output \
  --model triage_model.pkl
```

**Terminal 2 — Producer (generates synthetic patients)**
```bash
python kafka_producer.py \
  --broker localhost:9092 \
  --topic triage-input \
  --interval 2        # one patient every 2 seconds
```

**Terminal 3 — Streamlit Dashboard**
```bash
streamlit run streamlit_app.py -- \
  --broker localhost:9092 \
  --topic triage-output
```
Open `http://localhost:8501` in your browser.

---

## 5. Dashboard features

| Section | Description |
|---|---|
| **Top metrics** | Total patients, emergency count, model accuracy, avg confidence, non-urgent count |
| **Acuity Distribution** | Bar chart of KTAS 1–5 predictions (colour-coded) |
| **Running Accuracy** | Line chart of prediction accuracy vs ground-truth over time |
| **Live Patient Feed** | Per-patient cards with vitals, probability bars, KTAS badge |
| **Emergency filter** | Checkbox to show only KTAS 1–2 patients |

---

## 6. File summary

| File | Purpose |
|---|---|
| `triage_system.py` | Original model training code |
| `triage_model.pkl` | Saved model bundle (produced by training) |
| `kafka_producer.py` | Synthetic patient generator → `triage-input` |
| `kafka_consumer.py` | Model inference → `triage-output` |
| `streamlit_app.py` | Real-time dashboard reading `triage-output` |

---

## 7. Customisation

- **Real data**: In `kafka_producer.py` replace `generate_patient()` with a function
  that reads from your actual database / EHR system and publishes real patient rows.
- **Prediction interval**: Change `--interval` in the producer.
- **Broker address**: All three scripts accept `--broker HOST:PORT`.
- **Rolling window**: Edit `MAX_PATIENTS = 200` in `streamlit_app.py`.
