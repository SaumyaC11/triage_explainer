import pandas as pd
import json

df = pd.read_csv("src/data/train.csv")

stats = {}
for acuity in [1, 2, 3, 4, 5]:
    subset = df[df["triage_acuity"] == acuity]
    stats[acuity] = {
        col: {
            "mean": subset[col].mean(),
            "std":  subset[col].std()
        }
        for col in ["heart_rate","systolic_bp","diastolic_bp",
                    "spo2","respiratory_rate","temperature_c",
                    "gcs_total","pain_score","news2_score"]
        if col in subset.columns
    }

with open("src/results/acuity_stats.json", "w") as f:
    json.dump(stats, f, indent=2)

print("Saved acuity_stats.json")