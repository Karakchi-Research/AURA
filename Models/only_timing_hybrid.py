# only_timing_hybrid.py
# Uses only timing sensor data with both thresholding and machine learning for anomaly detection.

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
# Process a single block of data, simulating timing anomalies and measuring latency.
#
# @param args - A tuple containing a block of data and a boolean indicating whether to inject an anomaly.
# @return A dictionary containing a timing feature and a label indicating if the block is malicious.
def process_block(args):
    block, inject_anomaly = args
    start_time = time.perf_counter()

    if inject_anomaly:
        attack_type = random.choice(["spike", "stealth"])
        if attack_type == "spike":
            time.sleep(random.uniform(0.01, 0.02))  # Obvious delay
        else:
            time.sleep(random.uniform(0.001, 0.003))  # Subtle shift

    cipher = AES.new(KEY, AES.MODE_ECB)
    cipher.encrypt(block)
    end_time = time.perf_counter()

    return {
        "timing": end_time - start_time,
        "is_malicious": 1 if inject_anomaly else 0,
    }


# calculate_thresholds()
# Calculate thresholds for timing anomalies based on the training data.
#
# @param results - A list of dictionaries containing timing features and a label indicating if the block is malicious.
# @return A dictionary containing the calculated threshold for timing
def calculate_thresholds(results):
    normal_results = [result for result in results if result["is_malicious"] == 0]
    values = [result["timing"] for result in normal_results]
    mean = sum(values) / len(values)
    sigma = (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5
    return {"timing": mean + (3.0 * sigma)}


# run_on_file()
# Process a file, extracting timing features and detecting anomalies using both thresholding and machine learning.
#
# @param file_path - The path to the file to be processed.
def run_on_file(file_path):
    if not os.path.exists(file_path):
        print("Error: File '{}' not found.".format(file_path))
        return

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

    # End-to-end timer: file read -> feature extraction -> training -> detection
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
            results.append(process_block((chunk, random.random() < 0.1)))
            idx += 1
            if idx % 2000 == 0:
                print(f" Processed {idx} blocks...", end="\r")

    df = pd.DataFrame(results)
    split = int(len(df) * 0.3)
    train_df, test_df = df.iloc[:split].copy(), df.iloc[split:].copy()

    # 2. Calculate thresholds from the training data
    threshold = calculate_thresholds(train_df.to_dict("records"))["timing"]

    # 3. Train a Random Forest classifier on the timing feature
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(train_df[["timing"]], train_df["is_malicious"])

    # 4. Detect anomalies in the test set using both methods
    start_bench = time.perf_counter()
    thresh_hits = test_df["timing"] > threshold
    ml_hits = rf.predict(test_df[["timing"]])
    end_bench = time.perf_counter()

    y_pred_hybrid = (thresh_hits.values | ml_hits.astype(bool)).astype(int)
    y_true = test_df["is_malicious"].values

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_hybrid, labels=[0, 1]).ravel()

    # 5. Compute confusion matrix and metrics
    e2e_latency = time.perf_counter() - e2e_start
    avg_latency = df["timing"].mean()
    total_latency = df["timing"].sum() + (end_bench - start_bench)
    throughput = file_size_mb / total_latency if total_latency > 0 else 0

    # 6. Export results and summary
    out_dir = os.path.join(os.getcwd(), "results")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    base = os.path.splitext(os.path.basename(file_path))[0]
    preds = test_df.copy()
    preds["pred"] = y_pred_hybrid
    preds["actual"] = y_true
    preds.to_csv(
        os.path.join(out_dir, "timing_only_{}_predictions.csv".format(base)),
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
        "precision": "{:.4f}".format(precision_score(y_true, y_pred_hybrid)),
        "recall": "{:.4f}".format(recall_score(y_true, y_pred_hybrid)),
        "f1": "{:.4f}".format(f1_score(y_true, y_pred_hybrid)),
    }

    save_summary(out_dir, base, "Timing Only", summary)
    print_report("TIMING ONLY", base, summary)

    # Excel report (predictions + summary)
    try:
        report_path = os.path.join(out_dir, "timing_only_{}_report.xlsx".format(base))
        summary_df = pd.DataFrame(
            {"metric": list(summary.keys()), "value": list(summary.values())}
        )
        with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
            preds.to_excel(writer, sheet_name="predictions", index=False)
            summary_df.to_excel(writer, sheet_name="summary", index=False)
    except Exception:
        pass


# Main Execution
def main():
    if len(sys.argv) < 2:
        print("Usage: python3 only_timing_hybrid.py <video_file>")
        return
    run_on_file(sys.argv[1])


if __name__ == "__main__":
    main()
