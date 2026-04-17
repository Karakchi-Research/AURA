#!/usr/bin/env python3
"""
AURA LPBF Geometric Monitoring Orchestrator
Runs all model experiments (timing-only, optical-only, timing+optical hybrid)
with both threshold-based and ML-based detection methods on a directory of optical camera images.
Analyzes spatial geometry (wall thickness, surface roughness, center drift) and temporal 
dynamics (layer-to-layer changes) to detect LPBF thin-wall instabilities.
Aggregates reports and metrics across all 6 detection approaches.
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime
import shutil
import re
import json

# ===== LPBF GEOMETRIC MONITORING TESTS =====
# Test models grouped by detection approach:
#   Spatial (Optical): wall thickness, drift, roughness from static images
#   Temporal (Timing): layer-to-layer changes and cumulative instability evolution
#   Hybrid: combined spatial + temporal features for robust anomaly detection
SCRIPTS = [
    "optical_threshold.py",        # Spatial: threshold-based anomaly detection
    "optical_ml.py",               # Spatial: ML-based anomaly detection (Random Forest)
    "timing_threshold.py",         # Temporal: threshold-based anomaly detection
    "timing_ml.py",                # Temporal: ML-based anomaly detection (Random Forest)
    "timing_optical_threshold.py", # Hybrid: spatial + temporal threshold fusion
    "timing_optical_ml.py",        # Hybrid: spatial + temporal ML fusion
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
        print("  python3 run_all.py <image_directory> [thickness_threshold] [run_label]")
        print("\nExample:")
        print("  python3 run_all.py /path/to/optical_images 5 lpbf_test_0.6mm")
        print("\nParameters:")
        print("  image_directory: Path to directory with LPBF optical camera images")
        print("  thickness_threshold: Max acceptable thickness variance in % (default: 5)")
        print("  run_label: Experiment label for output (default: auto-generated timestamp)")
        sys.exit(1)

    image_dir = Path(sys.argv[1])
    if not image_dir.exists() or not image_dir.is_dir():
        print(f"ERROR: Image directory not found or not a directory: {image_dir}")
        sys.exit(1)

    thickness_threshold = sys.argv[2] if len(sys.argv) > 2 else "5"
    run_label = (
        sys.argv[3]
        if len(sys.argv) > 3
        else f"lpbf_run_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    )

    output_dir = Path("reports") / run_label
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_file = output_dir / "experiment_summary.txt"
    metrics_file = output_dir / "metrics.json"
    python_cmd = find_python()
    exit_codes = {}
    all_metrics = {}

    print(f"\n{'='*80}")
    print(f"AURA LPBF Geometric Monitoring Orchestrator")
    print(f"{'='*80}")
    print(f"Image Directory: {image_dir}")
    print(f"Thickness Threshold: ±{thickness_threshold}%")
    print(f"Run Label: {run_label}")
    print(f"Output Directory: {output_dir}")
    print(f"Python Command: {python_cmd}\n")

    with summary_file.open("w") as summary:
        summary.write(f"AURA LPBF Geometric Monitoring Experiment Report\n")
        summary.write(f"{'='*80}\n")
        summary.write(f"Run label: {run_label}\n")
        summary.write(f"Image directory: {image_dir}\n")
        summary.write(f"Thickness threshold: ±{thickness_threshold}%\n")
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

            cmd = [python_cmd, str(script_path), str(image_dir), thickness_threshold]

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
        summary.write("Detection Performance:\n")
        summary.write(f"{'  Model':<35} {'Accuracy':<12} {'Precision':<12} {'Recall':<12}\n")
        summary.write(f"  {'-'*71}\n")
        
        for script_name, metrics in sorted(all_metrics.items()):
            accuracy = metrics.get('accuracy', 0)
            precision = metrics.get('precision', 0)
            recall = metrics.get('recall', 0)
            summary.write(f"  {script_name:<35} {accuracy:>10.2f}% {precision:>10.2f}% {recall:>10.2f}%\n")

        # Detection comparison
        summary.write(f"\n\nAnomalies Detected:\n")
        summary.write(f"{'  Model':<35} {'TP':<8} {'FP':<8} {'FN':<8} {'Total':<8}\n")
        summary.write(f"  {'-'*63}\n")
        
        for script_name, metrics in sorted(all_metrics.items()):
            tp = metrics.get('tp', 0)
            fp = metrics.get('fp', 0)
            fn = metrics.get('fn', 0)
            total = tp + fp + fn if (tp + fp + fn > 0) else metrics.get('detected_anomalies', 0)
            summary.write(f"  {script_name:<35} {tp:<8} {fp:<8} {fn:<8} {total:<8}\n")

        # Performance comparison
        summary.write(f"\n\nPerformance Metrics:\n")
        summary.write(f"{'  Model':<35} {'Memory (MB)':<15}\n")
        summary.write(f"  {'-'*50}\n")
        
        for script_name, metrics in sorted(all_metrics.items()):
            memory = metrics.get('memory_mb', 0)
            summary.write(f"  {script_name:<35} {memory:>13.2f}\n")

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
