#!/usr/bin/env python3
"""
Quick script to approve MLflow runs for migration.
Usage:
    python approve_run.py <run_id>
    python approve_run.py <run_id1> <run_id2> <run_id3>
"""

import os
import sys

import mlflow

# Setup MLflow
project_root = os.getenv("root_folder")
mlflow_tracking_uri = (
    os.path.join(project_root, "mlruns") if project_root else "./mlruns"
)
mlflow.set_tracking_uri(f"file://{mlflow_tracking_uri}")

if len(sys.argv) < 2:
    print("Usage: python approve_run.py <run_id1> [run_id2] [run_id3] ...")
    print("Examples:")
    print("  python approve_run.py abc123")
    print("  python approve_run.py abc123 def456 ghi789")
    sys.exit(1)

run_ids = sys.argv[1:]

client = mlflow.tracking.MlflowClient()
success_count = 0
error_count = 0

print(f"Approving {len(run_ids)} run(s) for migration...")
print("=" * 50)

for run_id in run_ids:
    try:
        # Check if run exists
        run = client.get_run(run_id)
        current_status = run.data.tags.get("data_status", "not set")

        print(f"Run: {run_id[:8]}...")
        print(f"  Current status: {current_status}")

        # Set approval tag
        client.set_tag(run_id, "data_status", "approved")
        print("  ✅ Approved for migration")
        success_count += 1

    except Exception as e:
        print(f"Run: {run_id[:8]}...")
        print(f"  ❌ Error: {e}")
        error_count += 1

    print()

print("=" * 50)
print(f"Summary: {success_count} approved, {error_count} errors")

if error_count > 0:
    sys.exit(1)
