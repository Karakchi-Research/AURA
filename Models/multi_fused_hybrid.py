# multi_fused_hybrid.py
# Combines timing, thermal, and optical sensors with both thresholding and machine learning for anomaly detection.

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
# Process a single block, extracting timing, thermal, and optical features.
#
# @param args - A tuple containing a block, its index, and an anomaly flag.
# @return A dictionary containing sensor features and the malicious label.
def process_block(args):
    block, index, inject_anomaly = args
    start_time = time.perf_counter()
    modified_block = list(block)

    # Base sensors (simulated hardware baselines)
    base_temp = 35.0  # Celsius
    base_optical = 100  # Intensity units

    # Hamming Weight (count of '1' bits in the block)
    hw = sum(bin(b).count("1") for b in block)

    if inject_anomaly:
        anomaly_type = random.choice(["delay", "fault", "stealth"])
        if anomaly_type == "delay":
            time.sleep(random.uniform(0.005, 0.01))
            base_temp -= 2.0  # Throttling/idle cooling
        elif anomaly_type == "fault":
            base_temp += 8.0  # High thermal spike
            base_optical += 50  # Optical burst
            modified_block[0] ^= 0xFF
        else:  # Subtle anomalies that stay near thresholds
            base_temp += 1.5
            base_optical += 5.0

    cipher = AES.new(KEY, AES.MODE_ECB)
    cipher.encrypt(bytes(modified_block))
    end_time = time.perf_counter()

    return {
        "index": index,
        "timing": end_time - start_time,
        "thermal": base_temp + random.uniform(-0.5, 0.5),
        "optical": base_optical + (hw * 0.5) + random.uniform(-2, 2),
        "is_malicious": 1 if inject_anomaly else 0,
    }


# calculate_thresholds()
# Calculate sensor thresholds from baseline measurements.
#
# @param results - A list of sensor-feature dictionaries.
# @return A dictionary containing one threshold per sensor.
def calculate_thresholds(results):
    thresholds = {}
    for sensor in ["timing", "thermal", "optical"]:
        values = [result[sensor] for result in results]
        mean = sum(values) / len(values)
        sigma = (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5
        thresholds[sensor] = mean + (3.0 * sigma)
    return thresholds


# run_on_file()
# Process a file, detect anomalies with thresholds and ML, and export metrics.
#
# @param file_path - The path to the file to be processed.
def run_on_file(file_path):
    if not os.path.exists(file_path):
        print("Error: File '{}' not found.".format(file_path))
        return

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

    # End-to-end timer (file read -> feature extraction -> training -> detection)
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

    df = pd.DataFrame(results)
    split = int(len(df) * 0.3)
    train_df, test_df = df.iloc[:split].copy(), df.iloc[split:].copy()

    # 2. Calculate thresholds for each sensor based on training data
    metrics = ["timing", "thermal", "optical"]
    threshold_values = calculate_thresholds(train_df.to_dict("records"))

    # 3. Train a Random Forest classifier on the training data
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(train_df[metrics], train_df["is_malicious"])

    # 4. Detect anomalies in the test set using both thresholding and ML
    thresh_hits = (
        (test_df["timing"] > threshold_values["timing"])
        | (test_df["thermal"] > threshold_values["thermal"])
        | (test_df["optical"] > threshold_values["optical"])
    )
    ml_hits = rf.predict(test_df[metrics])

    # Generate final predictions (a block is malicious if either method flags it)
    y_pred_hybrid = (thresh_hits.values | ml_hits.astype(bool)).astype(int)
    y_true = test_df["is_malicious"].values

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_hybrid, labels=[0, 1]).ravel()

    # 5. Compute confusion matrix and metrics
    e2e_latency = time.perf_counter() - e2e_start
    avg_latency = df["timing"].mean()
    total_latency = df["timing"].sum()
    throughput = file_size_mb / total_latency if total_latency > 0 else 0

    # 6. Export results and summary
    out_dir = os.path.join(os.getcwd(), "results")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    base = os.path.splitext(os.path.basename(file_path))[0]
    test_df["hybrid_pred"] = y_pred_hybrid
    test_df.to_csv(
        os.path.join(out_dir, "multi_fused_hybrid_{}_detection.csv".format(base)),
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
        "threshold_timing": "{:.6f}".format(threshold_values["timing"]),
        "threshold_thermal": "{:.2f}".format(threshold_values["thermal"]),
        "threshold_optical": "{:.2f}".format(threshold_values["optical"]),
    }
    save_summary(out_dir, base, "Multi Fused Hybrid", summary)
    print_report("MULTI FUSED HYBRID", base, summary)

    # Excel report with predictions + summary
    try:
        report_path = os.path.join(
            out_dir, "multi_fused_hybrid_{}_report.xlsx".format(base)
        )
        summary_df = pd.DataFrame(
            {"metric": list(summary.keys()), "value": list(summary.values())}
        )
        with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
            test_df.to_excel(writer, sheet_name="predictions", index=False)
            summary_df.to_excel(writer, sheet_name="summary", index=False)
    except Exception:
        pass


# main()
# Parse the command-line file path and run the detector.
def main():
    if len(sys.argv) < 2:
        print("Usage: python3 multi_fused_hybrid.py <video_file>")
        return
    run_on_file(sys.argv[1])


if __name__ == "__main__":
    main()
