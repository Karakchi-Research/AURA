# optical_thermal_threshold.py
# Combines optical and thermal sensors with thresholding for anomaly detection.

# Import necessary libraries
import os
import random
import sys
import time

import pandas as pd
from Crypto.Cipher import AES

from summary_helper import print_report, save_summary

# Configuration
BLOCK_SIZE = 16
KEY = b"ThisIsA16ByteKey"
MAX_BLOCKS = 10000


# process_block()
# Process a single block, extracting optical and thermal features.
#
# @param args - A tuple containing a block, its index, and an anomaly flag.
# @return A dictionary containing optical, thermal, and label features.
def process_block(args):
    block, index, inject_anomaly = args
    start_time = time.perf_counter()
    modified_block = list(block)

    # 1. SIMULATE OPTICAL SENSOR (Based on Hamming Weight)
    # HW represents light intensity emitted during data processing
    base_optical = 100
    hw = sum(bin(b).count("1") for b in block)
    current_optical = base_optical + (hw * 0.5) + random.uniform(-1, 1)

    # 2. SIMULATE THERMAL SENSOR (Celsius)
    base_temp = 38.0
    current_temp = base_temp + random.uniform(-0.3, 0.3)

    if inject_anomaly:
        # Randomly choose between a Thermal spike or an Optical burst
        anomaly_choice = random.choice(["thermal_spike", "optical_burst"])
        if anomaly_choice == "thermal_spike":
            current_temp += random.uniform(8.0, 15.0)  # Thermal spike (e.g. fault)
            modified_block[0] ^= 0xFF
        else:
            current_optical += random.uniform(40, 70)  # Optical burst (e.g. laser)

    # Standard AES Encryption
    cipher = AES.new(KEY, AES.MODE_ECB)
    cipher.encrypt(bytes(modified_block))
    end_time = time.perf_counter()

    return {
        "index": index,
        "optical": current_optical,
        "thermal": current_temp,
        "time": end_time - start_time,
        "is_malicious": 1 if inject_anomaly else 0,
    }


# calculate_thresholds()
# Calculate sensor thresholds from baseline measurements.
#
# @param results - A list of sensor-feature dictionaries.
# @return A dictionary containing one threshold per sensor.
def calculate_thresholds(results):
    thresholds = {}
    for sensor in ["optical", "thermal"]:
        vals = [r[sensor] for r in results]
        mu = sum(vals) / len(vals)
        sigma = (sum((v - mu) ** 2 for v in vals) / len(vals)) ** 0.5
        thresholds[sensor] = mu + (3.0 * sigma)  # Standard Z-score threshold
    return thresholds


# run_on_file()
# Process a file, detect threshold-based anomalies, and export metrics.
#
# @param file_path - The path to the file to be processed.
def run_on_file(file_path):
    if not os.path.exists(file_path):
        print("Error: File '{}' not found.".format(file_path))
        return

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
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

    # 2. Calculate thresholds from the baseline data
    split = int(len(results) * 0.2)
    thresholds = calculate_thresholds(results[:split])

    # 3. Detect anomalies in the test set
    tp, fp, fn, tn = 0, 0, 0, 0
    for r in results[split:]:
        # Detect if EITHER Optical OR Thermal sensor exceeds threshold
        detected = (
            r["optical"] > thresholds["optical"] or r["thermal"] > thresholds["thermal"]
        )

        if r["is_malicious"] and detected:
            tp += 1
        elif not r["is_malicious"] and detected:
            fp += 1
        elif r["is_malicious"] and not detected:
            fn += 1
        else:
            tn += 1

    # 4. Compute confusion matrix and metrics
    e2e_latency = time.perf_counter() - e2e_start
    avg_latency = sum(r["time"] for r in results) / len(results)
    total_latency = sum(r["time"] for r in results)
    throughput = file_size_mb / total_latency if total_latency > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    # 5. Export results and summary
    out_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(file_path))[0]
    predictions = results[split:]
    for row in predictions:
        row["predicted_label"] = int(
            row["optical"] > thresholds["optical"]
            or row["thermal"] > thresholds["thermal"]
        )
    pd.DataFrame(predictions).to_csv(
        os.path.join(
            out_dir, "optical_thermal_threshold_{}_predictions.csv".format(base)
        ),
        index=False,
    )
    summary = {
        "file": os.path.basename(file_path),
        "file_size_mb": "{:.2f}".format(file_size_mb),
        "avg_latency_sec": "{:.6f}".format(avg_latency),
        "end_to_end_latency_sec": "{:.4f}".format(e2e_latency),
        "throughput_mb_s": "{:.2f}".format(throughput),
        "memory_usage_mb": "{:.2f}".format(
            pd.DataFrame(results).memory_usage(deep=True).sum() / (1024 * 1024)
        ),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": "{:.4f}".format(prec),
        "recall": "{:.4f}".format(rec),
        "f1": "{:.4f}".format(f1),
        "threshold_optical": "{:.2f}".format(thresholds["optical"]),
        "threshold_thermal": "{:.2f}".format(thresholds["thermal"]),
    }
    save_summary(out_dir, base, "Optical Thermal Threshold", summary)
    print_report("OPTICAL THERMAL THRESHOLD", base, summary)

    # Excel report (predictions + summary)
    try:
        report_path = os.path.join(
            out_dir, "optical_thermal_threshold_{}_report.xlsx".format(base)
        )
        summary_df = pd.DataFrame(
            {"metric": list(summary.keys()), "value": list(summary.values())}
        )
        with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
            predictions.to_excel(writer, sheet_name="predictions", index=False)
            summary_df.to_excel(writer, sheet_name="summary", index=False)
    except Exception:
        pass


# main()
# Parse the command-line file path and run the detector.
def main():
    if len(sys.argv) < 2:
        print("Usage: python optical_thermal_threshold.py <video_file>")
        return
    run_on_file(sys.argv[1])


if __name__ == "__main__":
    main()
