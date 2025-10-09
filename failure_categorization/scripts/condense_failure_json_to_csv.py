#!/usr/bin/env python3
import sys, json, csv
from pathlib import Path

def collect_rows(results_dir: Path):
    rows = []
    for jf in results_dir.glob("*_analysis_*actions.json"):
        try:
            data = json.loads(jf.read_text())
        except Exception:
            continue

        instances = data.get("results") or data.get("failed_instances") or []
        for inst in instances:
            instance_id = inst.get("instance_id") or inst.get("id") or "unknown"
            failure_type = inst.get("category") or inst.get("failure_type") or "other"
            rationale = inst.get("description") or inst.get("reasoning") or ""
            rows.append((instance_id, failure_type, rationale))
    return rows

def main():
    if len(sys.argv) < 2:
        print("Usage: python condense_failure_csv.py <results_dir> [output_csv]")
        sys.exit(1)

    results_dir = Path(sys.argv[1]).resolve()
    out_csv = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else (results_dir / "condensed_failure_annotations.csv")

    rows = collect_rows(results_dir)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["instance_id", "category", "rationale"])
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_csv}")

if __name__ == "__main__":
    main()