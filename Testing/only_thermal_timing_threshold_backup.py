import os
import sys
import time
import random
import pandas as pd
from Crypto.Cipher import AES

BLOCK_SIZE = 16
KEY = b'ThisIsA16ByteKey'

def process_block(args):
    block, index, inject_anomaly = args

    start_time = time.perf_counter()
    modified_block = list(block)

    base_temp = 38.0
    current_temp = base_temp + random.uniform(-0.3, 0.3)

    if inject_anomaly:
        anomaly = random.choice(["timing_delay", "thermal_spike"])

        if anomaly == "timing_delay":
            time.

#time.sleep(random.uniform(0.005, 0.02))
        else:
            current_temp += random.uniform(8.0, 15.0)
            modified_block[0] ^= 0xFF

    cipher = AES.new(KEY, AES.MODE_ECB)
    cipher.encrypt(bytes(modified_block))

    end_time = time.perf_counter()

    return {
        "index": index,
        "latency": end_time - start_time,
        "thermal": current_temp,
        "is_malicious": 1 if inject_anomaly else 0
    }

def compute_metrics(tp, fp, fn, tn):
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 only_thermal_timing_threshold.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print("File not found.")
        sys.exit(1)

    print(f"Analyzing {os.path.basename(file_path)} (TIMING + THERMAL THRESHOLD MODE)...")

    with open(file_path, "rb") as f:
        raw = f.read()

    blocks = [raw[i:i+BLOCK_SIZE] for i in range(0, len(raw), BLOCK_SIZE) if len(raw[i:i+BLOCK_SIZE]) == BLOCK_SIZE]

    print(f"Processed {len(blocks)} blocks...")

    results = []

    start_total = time.perf_counter()

    for idx, block in enumerate(blocks):
        inject = random.random() < 0.08
        results.append(process_block((block, idx, inject)))

    end_total = time.perf_counter()

    df = pd.DataFrame(results)

    train_size = int(len(df) * 0.2)
    train = df.iloc[:train_size]
    test = df.iloc[train_size:].copy()

    time_thresh = train["latency"].mean() + 3 * train["latency"].std()
    thermal_thresh = train["thermal"].mean() + 3 * train["thermal"].std()

    test["pred"] = (
        (test["latency"] > time_thresh) |
        (test["thermal"] > thermal_thresh)
    ).astype(int)

    tp = int(((test["pred"] == 1) & (test["is_malicious"] == 1)).sum())
    fp = int(((test["pred"] == 1) & (test["is_malicious"] == 0)).sum())
    fn = int(((test["pred"] == 0) & (test["is_malicious"] == 1)).sum())
    tn = int(((test["pred"] == 0) & (test["is_malicious"] == 0)).sum())

    precision, recall, f1 = compute_metrics(tp, fp, fn, tn)

    file_size = os.path.getsize(file_path) / (1024 * 1024)
    avg_latency = df["latency"].mean()
    total_time = end_total - start_total
    throughput = file_size / total_time if total_time else 0

    print("======================================================")
    print(f"TIMING + THERMAL THRESHOLD REPORT: {os.path.basename(file_path)}")
    print(f"FILE SIZE: {file_size:.2f} MB")
    print(f"AVG LATENCY: {avg_latency:.6f} sec/block")
    print(f"THROUGHPUT: {throughput:.2f} MB/s")
    print("------------------------------------------------------")
    print(f"TIME THRESHOLD: {time_thresh:.4f}")
    print(f"THERMAL THRESHOLD: {thermal_thresh:.2f}")
    print("------------------------------------------------------")
    print(f"TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")
    print(f"PRECISION: {precision:.4f} | RECALL: {recall:.4f} | F1: {f1:.4f}")
    print("======================================================")

if __name__ == "__main__":
    main()
