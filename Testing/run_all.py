#!/usr/bin/env python3
"""
AURA Testing Orchestrator
Runs all model experiments (timing-only, optical-only, timing+optical fused)
with both threshold and ML detection methods on a directory of optical camera images.
Aggregates reports and metrics.
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime
import shutil
import re
import json

# ===== NEW SIMPLIFIED TEST SCRIPTS =====
SCRIPTS = [
    "timing_threshold.py",
    "timing_ml.py",
    "optical_threshold.py",
    "optical_ml.py",
    "timing_optical_threshold.py",
    "timing_optical_ml.py",
]

METRIC_RE = re.compile(
    r"(precision|recall|f1-score|accuracy|true positives|false positives|false negatives|actual anomalies|detected anomalies|Images Processed|Blocks to Process|Avg Latency|Memory Used|threshold:|thresholds:)",
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


def extract_metrics(log_content):
    """Extract key metrics from script log output."""
    metrics = {}
    lines = log_content.split('\n')
    
    for line in lines:
        if 'Accuracy:' in line:
            try:
                metrics['accuracy'] = float(line.split(':')[1].strip().rstrip('%'))
            except:
                pass
        elif 'Precision:' in line:
            try:
                metrics['precision'] = float(line.split(':')[1].strip().rstrip('%'))
            except:
                pass
        elif 'Recall:' in line:
            try:
                metrics['recall'] = float(line.split(':')[1].strip().rstrip('%'))
            except:
                pass
        elif 'F1-Score:' in line:
            try:
                metrics['f1'] = float(line.split(':')[1].strip().rstrip('%'))
            except:
                pass
        elif 'True Positives:' in line:
            try:
                metrics['tp'] = int(line.split(':')[1].strip())
            except:
                pass
        elif 'False Positives:' in line:
            try:
                metrics['fp'] = int(line.split(':')[1].strip())
            except:
                pass
        elif 'False Negatives:' in line:
            try:
                metrics['fn'] = int(line.split(':')[1].strip())
            except:
                pass
        elif 'Actual Anomalies:' in line:
            try:
                metrics['actual_anomalies'] = int(line.split(':')[1].strip())
            except:
                pass
        elif 'Detected Anomalies:' in line:
            try:
                metrics['detected_anomalies'] = int(line.split(':')[1].strip())
            except:
                pass
        elif 'Avg Latency:' in line:
            try:
                metrics['latency_us'] = float(line.split(':')[1].strip().split()[0])
            except:
                pass
        elif 'Memory Used:' in line:
            try:
                metrics['memory_mb'] = float(line.split(':')[1].strip().split()[0])
            except:
                pass
    
    return metrics


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 run_all.py <image_directory> [anomaly_percent] [run_label]")
        print("\nExample:")
        print("  python3 run_all.py /path/to/optical_images 20 my_experiment")
        sys.exit(1)

    image_dir = Path(sys.argv[1])
    if not image_dir.exists() or not image_dir.is_dir():
        print(f"ERROR: Image directory not found or not a directory: {image_dir}")
        sys.exit(1)

    anomaly_percent = sys.argv[2] if len(sys.argv) > 2 else "20"
    run_label = (
        sys.argv[3]
        if len(sys.argv) > 3
        else f"run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    )

    output_dir = Path("reports") / run_label
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_file = output_dir / "experiment_summary.txt"
    metrics_file = output_dir / "metrics.json"
    python_cmd = find_python()
    exit_codes = {}
    all_metrics = {}

    print(f"\n{'='*80}")
    print(f"AURA Testing Orchestrator")
    print(f"{'='*80}")
    print(f"Image Directory: {image_dir}")
    print(f"Anomaly Rate: {anomaly_percent}%")
    print(f"Run Label: {run_label}")
    print(f"Output Directory: {output_dir}")
    print(f"Python Command: {python_cmd}\n")

    with summary_file.open("w") as summary:
        summary.write(f"AURA Testing Experiment Report\n")
        summary.write(f"{'='*80}\n")
        summary.write(f"Run label: {run_label}\n")
        summary.write(f"Image directory: {image_dir}\n")
        summary.write(f"Anomaly rate: {anomaly_percent}%\n")
        summary.write(f"Started: {timestamp()}\n\n")

        for script in SCRIPTS:
            script_path = Path(script)
            log_file = output_dir / f"{script_path.stem}.log"

            print(f"[{len(exit_codes)+1}/{len(SCRIPTS)}] Running: {script}")
            summary.write(f"\nRunning: {script}\n")
            summary.write(f"{'-'*80}\n")

            if not script_path.exists():
                print(f"  ❌ Script not found")
                summary.write("  ERROR: Script not found\n\n")
                exit_codes[script] = 127
                continue

            cmd = [python_cmd, str(script_path), str(image_dir), anomaly_percent]

            with log_file.open("w") as log:
                process = subprocess.run(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=Path(__file__).parent
                )

            exit_codes[script] = process.returncode
            
            if process.returncode == 0:
                print(f"  ✓ Exit code: 0")
            else:
                print(f"  ❌ Exit code: {process.returncode}")

            summary.write(f"  Exit code: {process.returncode}\n")
            summary.write(f"  Log file: {log_file}\n")

            # Extract metrics
            if log_file.exists():
                log_content = log_file.read_text(errors="ignore")
                metrics = extract_metrics(log_content)
                all_metrics[script_path.stem] = metrics
                
                if metrics:
                    summary.write(f"\n  Metrics:\n")
                    for key, value in metrics.items():
                        if isinstance(value, float):
                            summary.write(f"    - {key}: {value:.2f}\n")
                        else:
                            summary.write(f"    - {key}: {value}\n")

            summary.write("\n")

        # Write consolidated comparison
        summary.write(f"\n\n{'='*80}\n")
        summary.write("CONSOLIDATED METRICS COMPARISON\n")
        summary.write(f"{'='*80}\n\n")

        # Accuracy comparison
        summary.write("Accuracy Comparison:\n")
        summary.write(f"{'  Model':<35} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
        summary.write(f"  {'-'*75}\n")
        
        for script_name, metrics in sorted(all_metrics.items()):
            accuracy = metrics.get('accuracy', 0)
            precision = metrics.get('precision', 0)
            recall = metrics.get('recall', 0)
            f1 = metrics.get('f1', 0)
            summary.write(f"  {script_name:<35} {accuracy:>10.2f}% {precision:>10.2f}% {recall:>10.2f}% {f1:>10.2f}%\n")

        # Detection comparison
        summary.write(f"\n\nDetection Statistics:\n")
        summary.write(f"{'  Model':<35} {'TP':<8} {'FP':<8} {'FN':<8} {'Total':<8}\n")
        summary.write(f"  {'-'*63}\n")
        
        for script_name, metrics in sorted(all_metrics.items()):
            tp = metrics.get('tp', 0)
            fp = metrics.get('fp', 0)
            fn = metrics.get('fn', 0)
            total = tp + fp + fn
            summary.write(f"  {script_name:<35} {tp:<8} {fp:<8} {fn:<8} {total:<8}\n")

        # Performance comparison
        summary.write(f"\n\nPerformance Metrics:\n")
        summary.write(f"{'  Model':<35} {'Latency (µs)':<15} {'Memory (MB)':<15}\n")
        summary.write(f"  {'-'*64}\n")
        
        for script_name, metrics in sorted(all_metrics.items()):
            latency = metrics.get('latency_us', 0)
            memory = metrics.get('memory_mb', 0)
            summary.write(f"  {script_name:<35} {latency:>13.2f} {memory:>13.2f}\n")

        summary.write(f"\n\nFinished: {timestamp()}\n")

    # Write metrics JSON
    with metrics_file.open("w") as mf:
        json.dump(all_metrics, mf, indent=2)

    print(f"\n{'='*80}")
    print(f"Experiment Complete")
    print(f"{'='*80}")
    print(f"📁 Reports saved to: {output_dir}")
    print(f"📄 Summary: {summary_file}")
    print(f"📊 Metrics (JSON): {metrics_file}\n")

    failed = [script for script, code in exit_codes.items() if code != 0]
    if failed:
        print("⚠️  Some scripts returned non-zero exit codes:")
        for script in failed:
            print(f"   - {script} -> {exit_codes[script]}")
        print()


if __name__ == "__main__":
    main()
