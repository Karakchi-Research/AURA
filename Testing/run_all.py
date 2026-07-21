#!/usr/bin/env python3

import sys
import subprocess
from pathlib import Path
from datetime import datetime
import shutil
import re

# ===== FULL LIST OF SCRIPTS TO RUN =====
SCRIPTS = [
    "only_optical_detector_ML.py",
    "only_optical_detector_threshold.py",
    "only_optical_timing_detector_ml.py",
    "only_optical_timing_detector_threshold.py",
    "only_thermal_ml.py",
    "only_thermal_timing_ml.py",
    "only_thermal_timing_threshold.py",
    "optical_thermal_ml.py",
    "optical_thermal_threshold.py",
    "thermal_adapter_machine_learning_final.py",
    "thermal_adapter_unlimitVideo.py",
    "multi_fused_ml.py",
    "multi_fused_threshold.py",
    "multi_fused_ml_ablation.py",
    "multi_fused_hybrid_optical_thermal_timing.py",
]

METRIC_RE = re.compile(
    r"(precision|recall|f1|accuracy|auc|threshold|throughput|latency|tp:|tn:|fp:|fn:|avg latency|file size|video size)",
    re.IGNORECASE,
)


def timestamp():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def find_python():
    if shutil.which("python3"):
        return "python3"
    if shutil.which("python"):
        return "python"
    print("ERROR: Python not found.")
    sys.exit(1)


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 run_all.py <file_path> [run_label]")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    if not input_file.exists():
        print(f"ERROR: File not found: {input_file}")
        sys.exit(1)

    run_label = (
        sys.argv[2]
        if len(sys.argv) > 2
        else f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    )

    output_dir = Path("reports") / run_label
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_file = output_dir / "report.txt"
    python_cmd = find_python()
    exit_codes = {}

    with summary_file.open("w") as summary:
        summary.write(f"Run label: {run_label}\n")
        summary.write(f"Input file: {input_file}\n")
        summary.write(f"Started: {timestamp()}\n\n")

        for script in SCRIPTS:
            script_path = Path(script)
            log_file = output_dir / f"{script_path.stem}.log"

            summary.write(f"Running: {script}\n")

            if not script_path.exists():
                summary.write("  ERROR: Script not found\n\n")
                exit_codes[script] = 127
                continue

            cmd = [python_cmd, str(script_path), str(input_file)]

            with log_file.open("w") as log:
                process = subprocess.run(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )

            exit_codes[script] = process.returncode

            summary.write(f"  Log: {log_file}\n")
            summary.write(f"  Exit code: {process.returncode}\n\n")

        summary.write("\n=== CONSOLIDATED METRICS ===\n\n")

        for script in SCRIPTS:
            log_file = output_dir / f"{Path(script).stem}.log"
            summary.write(f"{script}:\n")

            if not log_file.exists():
                summary.write("  (no log)\n\n")
                continue

            content = log_file.read_text(errors="ignore")
            lines = [
                line.strip()
                for line in content.splitlines()
                if METRIC_RE.search(line)
            ]

            if lines:
                for line in lines:
                    summary.write(f"  {line}\n")
            else:
                summary.write("  (no metrics found)\n")

            summary.write("\n")

        summary.write("Finished: " + timestamp() + "\n")

    print("\nDone.")
    print(f"Report folder: {output_dir}")
    print(f"Summary file: {summary_file}")

    failed = [script for script, code in exit_codes.items() if code != 0]
    if failed:
        print("\nWARNING: Some scripts returned non-zero exit codes:")
        for script in failed:
            print(f"  - {script} -> {exit_codes[script]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
