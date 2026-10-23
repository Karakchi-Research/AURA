# optical_timing_ml.py
# Combines optical and timing sensors with machine learning for anomaly detection.

# Import necessary libraries
import os
import random
import sys
import time

import pandas as pd
from Crypto.Cipher import AES
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from summary_helper import save_summary, print_report

# Configuration
BLOCK_SIZE = 16
KEY = b"ThisIsA16ByteKey"
MAX_BLOCKS = 10000


# process_block()
# Process a single block, extracting optical and timing features.
#
# @param args - A tuple containing a block, its index, and an anomaly flag.
# @return A dictionary containing optical, timing, and label features.
def process_block(args):
    block, index, inject_anomaly = args
    start_time = time.perf_counter()
    modified_block = list(block)

    # Simulated optical signal: base + Hamming Weight (HW)
    base_optical = 100
    hw = sum(bin(b).count("1") for b in block)

    # Optical signature = base + data-dependent emission + noise
    current_optical = base_optical + (hw * 0.5) + random.uniform(-1, 1)

    if inject_anomaly:
        # Optical signature of a fault attack (burst)
        current_optical += random.uniform(30, 60)
        modified_block[0] ^= 0xFF

    cipher = AES.new(KEY, AES.MODE_ECB)
    cipher.encrypt(bytes(modified_block))
    end_time = time.perf_counter()

    return {
        "optical_signal": current_optical,
        "hamming_weight": hw,
        "latency": end_time - start_time,
        "is_malicious": 1 if inject_anomaly else 0,
    }


# run_on_file()
# Process a file, train the detector, and export anomaly predictions and metrics.
#
# @param file_path - The path to the file to be processed.
def run_on_file(file_path):
    if not os.path.exists(file_path):
        print("Error: File '{}' not found.".format(file_path))
        return

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

    # End-to-end timer: file read -> feature extraction -> training -> inference
    e2e_start = time.perf_counter()

    # 1. Collect raw data from the file
    results = []
    print("Analyzing {}...".format(os.path.basename(file_path)))

    with open(file_path, "rb") as f:
        idx = 0
        while idx < MAX_BLOCKS:
            chunk = f.read(BLOCK_SIZE)
            if not chunk:
                break
            if len(chunk) < BLOCK_SIZE:
                chunk += b"\0" * (BLOCK_SIZE - len(chunk))
            results.append(process_block((chunk, idx, random.random() < 0.1)))
            idx += 1
            if idx % 2000 == 0:
                print(f" Processed {idx} blocks...", end="\r")

    # 2. Prepare the dataset
    df = pd.DataFrame(results)

    # Features: optical intensity and per-block latency
    X = df[["optical_signal", "latency"]]
    y = df["is_malicious"]

    # Split: 30% training, 70% testing
    split_idx = int(len(df) * 0.3)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # 3. Train a Random Forest classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # 4. Detect anomalies in the test set
    y_pred = rf.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

    # 5. Compute confusion matrix and metrics
    e2e_latency = time.perf_counter() - e2e_start
    avg_latency = df["latency"].mean()
    total_latency = df["latency"].sum()
    throughput = file_size_mb / (total_latency if total_latency > 0 else 1)

    # 6. Export results and summary
    out_dir = os.path.join(os.getcwd(), "results")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    base = os.path.splitext(os.path.basename(file_path))[0]
    preds = X_test.copy()
    preds["predicted_label"] = y_pred
    preds["actual_label"] = y_test.values
    preds.to_csv(
        os.path.join(out_dir, "optical_timing_ml_{}_predictions.csv".format(base)),
        index=False,
    )

    summary = {
        "file": os.path.basename(file_path),
        "file_size_mb": "{:.2f}".format(file_size_mb),
        "avg_latency_sec": "{:.6f}".format(avg_latency),
        "end_to_end_latency_sec": "{:.4f}".format(e2e_latency),
        "throughput_mb_s": "{:.2f}".format(throughput),
        "memory_usage_mb": "{:.2f}".format(
            df.memory_usage(deep=True).sum() / (1024 * 1024)
        ),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": "{:.4f}".format(precision_score(y_test, y_pred)),
        "recall": "{:.4f}".format(recall_score(y_test, y_pred)),
        "f1": "{:.4f}".format(f1_score(y_test, y_pred)),
    }

    save_summary(out_dir, base, "Optical Timing ML", summary)
    print_report("OPTICAL TIMING ML", base, summary)

    # Excel report (predictions + summary)
    try:
        report_path = os.path.join(
            out_dir, "optical_timing_ml_{}_report.xlsx".format(base)
        )
        summary_df = pd.DataFrame(
            {"metric": list(summary.keys()), "value": list(summary.values())}
        )
        with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
            preds.to_excel(writer, sheet_name="predictions", index=False)
            summary_df.to_excel(writer, sheet_name="summary", index=False)
    except Exception:
        pass


# main()
# Parse the command-line file path and run the detector.
def main():
    if len(sys.argv) < 2:
        print("Usage: python3 optical_timing_ml.py <video_file>")
        sys.exit(1)
    run_on_file(sys.argv[1])


if __name__ == "__main__":
    main()
