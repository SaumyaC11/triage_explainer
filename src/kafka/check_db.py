"""
check_db.py
===========
Diagnostic script — run this while (or after) kafka_consumer.py is running.
Tells you exactly what's in triage_latency.db and whether Streamlit would see it.

Usage:
    python check_db.py
    python check_db.py --db /path/to/triage_latency.db
"""

import sqlite3
import argparse
import os
import sys

def hr(): print("-" * 60)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="triage_latency.db",
                        help="Path to SQLite DB (default: triage_latency.db)")
    args = parser.parse_args()

    # ── 1. Does the file exist? ───────────────────────────────
    hr()
    print(f"DB PATH : {os.path.abspath(args.db)}")
    if not os.path.exists(args.db):
        print("❌  File does not exist.")
        print()
        print("Possible reasons:")
        print("  1. kafka_consumer.py hasn't run yet")
        print("  2. kafka_consumer.py is running from a different directory")
        print("     → check where you launched it; the DB is created in that cwd")
        print("  3. --db flag wasn't passed to kafka_consumer.py")
        print("     → default is 'triage_latency.db' in cwd")
        sys.exit(1)

    size_kb = os.path.getsize(args.db) / 1024
    print(f"✓  File exists  ({size_kb:.1f} KB)")

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row

    # ── 2. Tables present? ───────────────────────────────────
    hr()
    print("TABLES")
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()]
    for t in ["runs", "messages", "overrides"]:
        exists = t in tables
        print(f"  {'✓' if exists else '❌'} {t}")
    if not tables:
        print("  No tables found — schema never created.")
        print("  → latency_store.py may not be importable by kafka_consumer.py")
        sys.exit(1)

    # ── 3. runs table ─────────────────────────────────────────
    hr()
    print("RUNS TABLE")
    runs = conn.execute("SELECT * FROM runs ORDER BY started_at DESC").fetchall()
    if not runs:
        print("  ❌  No runs recorded.")
        print("  → kafka_consumer.py is running but write_message() isn't being called")
        print("     Check consumer logs for:  [METRICS] write_message failed: ...")
    else:
        print(f"  ✓  {len(runs)} run(s) found\n")
        for r in runs:
            fin = "✓ finalized" if r["finalized"] else "⏳ in progress"
            p50 = f"{r['p50_e2e_ms']:.0f}ms" if r["p50_e2e_ms"] else "not yet (finalize_run pending)"
            print(f"  run_id       : {r['run_id']}")
            print(f"  started_at   : {r['started_at']}")
            print(f"  messages     : {r['message_count']}")
            print(f"  status       : {fin}")
            print(f"  P50 e2e      : {p50}")
            print()

    # ── 4. messages table ─────────────────────────────────────
    hr()
    print("MESSAGES TABLE")
    total_msgs = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    if total_msgs == 0:
        print("  ❌  No messages recorded.")
        print("  → write_message() never succeeded — check consumer logs")
    else:
        print(f"  ✓  {total_msgs} message(s) recorded\n")
        # Sample the last 5
        rows = conn.execute("""
            SELECT patient_id, run_id,
                   kafka_transit_ms, ml_inference_ms,
                   safety_pipeline_ms, e2e_ms,
                   safety_rail_triggered, predicted_acuity
            FROM messages
            ORDER BY produced_at DESC
            LIMIT 5
        """).fetchall()
        print(f"  {'patient_id':<12} {'run_id':<18} {'kafka':>8} {'ml':>8} {'safety':>8} {'e2e':>8} {'rail':>5} {'ktas':>5}")
        print(f"  {'─'*12} {'─'*18} {'─'*8} {'─'*8} {'─'*8} {'─'*8} {'─'*5} {'─'*5}")
        for m in rows:
            def _ms(v): return f"{v:.0f}ms" if v is not None else "—"
            print(f"  {str(m['patient_id']):<12} {str(m['run_id']):<18} "
                  f"{_ms(m['kafka_transit_ms']):>8} {_ms(m['ml_inference_ms']):>8} "
                  f"{_ms(m['safety_pipeline_ms']):>8} {_ms(m['e2e_ms']):>8} "
                  f"{'Y' if m['safety_rail_triggered'] else 'N':>5} "
                  f"{str(m['predicted_acuity'] or '?'):>5}")

        # Check for any NULL segment columns (sign of bad timestamps)
        nulls = conn.execute("""
            SELECT COUNT(*) FROM messages
            WHERE e2e_ms IS NULL
               OR kafka_transit_ms IS NULL
               OR ml_inference_ms IS NULL
               OR safety_pipeline_ms IS NULL
        """).fetchone()[0]
        if nulls:
            print(f"\n  ⚠  {nulls} row(s) have NULL segment values")
            print("     → produced_at may be missing from Kafka messages")
            print("       (check kafka_producer.py stamps 'produced_at' and 'run_id')")
        else:
            print(f"\n  ✓  All segment columns populated")

    # ── 5. overrides table ────────────────────────────────────
    hr()
    print("OVERRIDES TABLE (auditability)")
    total_ovr = conn.execute("SELECT COUNT(*) FROM overrides").fetchone()[0]
    if total_ovr == 0:
        print("  ℹ  No overrides yet — accept or override a patient in the Triage tab")
        print("     feedback_consumer.py must also be running with --metrics-db pointing here")
    else:
        print(f"  ✓  {total_ovr} feedback event(s)\n")
        breakdown = conn.execute("""
            SELECT feedback_type, COUNT(*) as n
            FROM overrides GROUP BY feedback_type
        """).fetchall()
        for b in breakdown:
            print(f"    {b['feedback_type']:<22} : {b['n']}")

    # ── 6. What Streamlit would see ───────────────────────────
    hr()
    print("WHAT STREAMLIT SEES  (simulating LatencyStore read methods)\n")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from latency_store import LatencyStore
        store = LatencyStore(args.db)

        runs_out = store.get_runs()
        print(f"  get_runs()        → {len(runs_out)} run(s)")

        if runs_out:
            first_run = runs_out[0]["run_id"]
            msgs_out  = store.get_messages(first_run)
            print(f"  get_messages()    → {len(msgs_out)} message(s) for run '{first_run}'")

        matrix = store.get_override_matrix()
        print(f"  get_override_matrix() → {matrix if matrix else 'empty (no feedback yet)'}")

        reasons = store.get_rail_reasons()
        print(f"  get_rail_reasons()    → {reasons[:3]}{'...' if len(reasons) > 3 else ''}")

        print("\n  ✓  LatencyStore reads working correctly")

    except ImportError:
        print("  ❌  latency_store.py not importable from this directory")
        print(f"     cwd: {os.getcwd()}")
        print("     → run check_db.py from the same directory as latency_store.py")

    hr()
    conn.close()

if __name__ == "__main__":
    main()