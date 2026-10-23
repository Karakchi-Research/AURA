# optical_thermal_ml.py
# Combines optical and thermal sensors with machine learning for anomaly detection.

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
# Process a single block, extracting optical and thermal features.
#
# @param args - A tuple containing a block, its index, and an anomaly flag.
# @return A dictionary containing optical, thermal, and label features.
def process_block(args):
    block, index, inject_anomaly = args
    start_time = time.perf_counter()
    modified_block = list(block)

    # 1. Optical sensor simulation: base + Hamming Weight (HW) + noise
    base_opt = 100
    hw = sum(bin(b).count("1") for b in block)
    current_opt = base_opt + (hw * 0.5) + random.uniform(-1, 1)

    # 2. Thermal sensor simulation: base temperature + fluctuation
    base_temp = 38.0
    current_temp = base_temp + random.uniform(-0.3, 0.3)

    if inject_anomaly:
        # Choose which sensor the attack targets
        attack_type = random.choice(["thermal_glitch", "optical_laser"])
        if attack_type == "thermal_glitch":
            current_temp += random.uniform(7.0, 15.0)  # Heat spike
        else:
            current_opt += random.uniform(40, 80)  # Light burst
        modified_block[0] ^= 0xFF  # Fault injection

    cipher = AES.new(KEY, AES.MODE_ECB)
    cipher.encrypt(bytes(modified_block))
    end_time = time.perf_counter()

    return {
        "thermal_sig": current_temp,
        "optical_sig": current_opt,
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

    # Features: heat, light, and data pattern
    features = ["thermal_sig", "optical_sig", "hamming_weight"]
    X = df[features]
    y = df["is_malicious"]

    # 30% training / 70% testing
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
    preds_df = df.iloc[split_idx:].copy()
    preds_df["predicted_label"] = y_pred
    preds_df.to_csv(
        os.path.join(out_dir, "optical_thermal_ml_{}_predictions.csv".format(base)),
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

    save_summary(out_dir, base, "Optical Thermal ML", summary)
    print_report("OPTICAL THERMAL ML", base, summary)

    # Excel report (predictions + summary)
    try:
        report_path = os.path.join(
            out_dir, "optical_thermal_ml_{}_report.xlsx".format(base)
        )
        summary_df = pd.DataFrame(
            {"metric": list(summary.keys()), "value": list(summary.values())}
        )
        with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
            preds_df.to_excel(writer, sheet_name="predictions", index=False)
            summary_df.to_excel(writer, sheet_name="summary", index=False)
    except Exception:
        pass


# main()
# Parse the command-line file path and run the detector.
def main():
    if len(sys.argv) < 2:
        print("Usage: python3 optical_thermal_ml.py <video_file>")
        sys.exit(1)
    run_on_file(sys.argv[1])


if __name__ == "__main__":
    main()
