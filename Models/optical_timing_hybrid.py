# Combines optical and timing sensors with thresholding and machine learning.

# Import necessary libraries
import os
import random
import sys
import time

import pandas as pd
from Crypto.Cipher import AES
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from summary_helper import print_report, save_summary

BLOCK_SIZE = 16
KEY = b"ThisIsA16ByteKey"
MAX_BLOCKS = 10000


# process_block()
# Process a single block, extracting optical and timing features.
#
# @param args - A tuple containing a block and an anomaly flag.
# @return A dictionary containing optical, timing, and label features.
def process_block(args):
    block, _, inject_anomaly = args

    start_time = time.perf_counter()
    modified_block = list(block)

    # Base sensor (simulated hardware baselines)
    base_optical = 100  # Intensity units

    # Hamming Weight (count of '1' bits in the block)
    hw = sum(bin(b).count("1") for b in block)

    if inject_anomaly:
        anomaly_type = random.choice(["timing_delay", "optical_burst"])
        if anomaly_type == "timing_delay":
            time.sleep(random.uniform(0.005, 0.01))
        else:
            base_optical += 50  # Optical burst

    cipher = AES.new(KEY, AES.MODE_ECB)
    cipher.encrypt(bytes(modified_block))
    end_time = time.perf_counter()

    return {
        "timing": end_time - start_time,
        "thermal": base_thermal + random.uniform(-0.5, 0.5),
        "optical": base_optical + (hw * 0.5) + random.uniform(-2, 2),
        "hamming_weight": hw,
        "is_malicious": 1 if inject_anomaly else 0,
    }


# calculate_thresholds()
# Calculate thresholds from baseline measurements.
#
# @param results - A list of sensor-feature dictionaries.
# @return A dictionary containing one threshold per sensor.
def calculate_thresholds(results):
    thresholds = {}
    for sensor in ["optical", "timing", "hamming_weight"]:
        values = [result[sensor] for result in results]
        mean = sum(values) / len(values)
        sigma = (sum((value - mean) ** 2 for value in values) / len(values)) ** 0.5
        thresholds[sensor] = mean + (3.0 * sigma)
    return thresholds


# run_on_file()
# Process a file, detect anomalies using thresholding and ML, and export metrics.
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
    with open(file_path, "rb") as file:
        index = 0
        while index < MAX_BLOCKS:
            chunk = file.read(BLOCK_SIZE)
            if not chunk:
                break
            if len(chunk) < BLOCK_SIZE:
                chunk += b"\0" * (BLOCK_SIZE - len(chunk))
            results.append(process_block((chunk, index, random.random() < 0.1)))
            index += 1
    if not results:
        print("No data blocks found in {}.".format(file_path))
        return
    # 2. Prepare the dataset
    df = pd.DataFrame(results)
    split = max(1, int(len(df) * 0.3))
    train = df.iloc[:split].copy()
    test = df.iloc[split:].copy()
    if test.empty:
        test = train.copy()
    # 3. Calculate thresholds and train the Random Forest classifier.
    threshold_values = calculate_thresholds(train.to_dict("records"))
    feature_names = ["optical", "timing", "hamming_weight"]
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(train[feature_names], train["is_malicious"])
    ml_hits = classifier.predict(test[feature_names]).astype(bool)
    threshold_hits = (
        test[feature_names].gt(pd.Series(threshold_values)).any(axis=1).values
    )
    predictions = (ml_hits | threshold_hits).astype(int)
    # 4. Detect anomalies in the test set using both methods.
    actual = test["is_malicious"].values
    tn, fp, fn, tp = confusion_matrix(actual, predictions, labels=[0, 1]).ravel()
    precision = precision_score(actual, predictions, zero_division=0)
    recall = recall_score(actual, predictions, zero_division=0)
    f1 = f1_score(actual, predictions, zero_division=0)
    # 5. Export results and summary.
    output_dir = os.path.join(os.getcwd(), "results")
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(file_path))[0]
    predictions_df = test.copy()
    predictions_df["predicted_label"] = predictions
    predictions_df.to_csv(
        os.path.join(
            output_dir, "{}_{}_predictions.csv".format("optical_timing_hybrid", base)
        ),
        index=False,
    )
    avg_latency = df["timing"].mean()
    total_latency = df["timing"].sum()
    throughput = file_size_mb / total_latency if total_latency > 0 else 0
    summary = {
        "file": os.path.basename(file_path),
        "file_size_mb": "{:.2f}".format(file_size_mb),
        "avg_latency_sec": "{:.6f}".format(avg_latency),
        "end_to_end_latency_sec": "{:.4f}".format(time.perf_counter() - e2e_start),
        "throughput_mb_s": "{:.2f}".format(throughput),
        "memory_usage_mb": "{:.2f}".format(
            df.memory_usage(deep=True).sum() / (1024 * 1024)
        ),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": "{:.4f}".format(precision),
        "recall": "{:.4f}".format(recall),
        "f1": "{:.4f}".format(f1),
        "threshold_optical": "{:.2f}".format(threshold_values["optical"]),
        "threshold_timing": "{:.6f}".format(threshold_values["timing"]),
        "threshold_hamming_weight": "{:.2f}".format(
            threshold_values["hamming_weight"]
        ),
    }
    save_summary(output_dir, base, "Optical Timing Hybrid", summary)
    print_report("OPTICAL TIMING HYBRID", base, summary)


# main()
# Parse the command-line file path and run the detector.
def main():
    if len(sys.argv) < 2:
        print("Usage: python3 optical_timing_hybrid.py <video_file>")
        return
    run_on_file(sys.argv[1])


if __name__ == "__main__":
    main()
