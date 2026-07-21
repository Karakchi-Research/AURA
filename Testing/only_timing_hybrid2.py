import pandas as pd
import numpy as np
import time
import os
import sys
import random
from Crypto.Cipher import AES
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

# Configuration
BLOCK_SIZE = 16
KEY = b'ThisIsA16ByteKey'

def get_timing_data(args):
    """Measures encryption latency and injects hybrid anomalies."""
    block, inject_anomaly = args
    start_time = time.perf_counter()
    
    if inject_anomaly:
        attack_type = random.choice(["spike", "stealth"])
        if attack_type == "spike":
            time.sleep(random.uniform(0.01, 0.02)) # Obvious delay
        else:
            time.sleep(random.uniform(0.001, 0.003)) # Subtle shift

    cipher = AES.new(KEY, AES.MODE_ECB)
    cipher.encrypt(block)
    
    latency = time.perf_counter() - start_time
    return {"latency": latency, "is_malicious": 1 if inject_anomaly else 0}

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 only_timing_hybrid.py <video_file>")
        return

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    # 1. Data Collection
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    raw_data = []
    print(f"Analyzing: {os.path.basename(file_path)} ({file_size_mb:.2f} MB)")

    with open(file_path, "rb") as f:
        for idx in range(10000):
            chunk = f.read(BLOCK_SIZE)
            if not chunk: break
            if len(chunk) < BLOCK_SIZE: chunk += b'\0' * (BLOCK_SIZE - len(chunk))
            raw_data.append(get_timing_data((chunk, random.random() < 0.1)))

    df = pd.DataFrame(raw_data)
    split = int(len(df) * 0.3)
    train_df, test_df = df.iloc[:split].copy(), df.iloc[split:].copy()

    # 2. Layer 1: Threshold
    normal_train = train_df[train_df['is_malicious'] == 0]['latency']
    threshold = normal_train.mean() + (3 * normal_train.std())

    # 3. Layer 2: ML
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(train_df[['latency']], train_df['is_malicious'])

    # 4. Detection
    start_bench = time.perf_counter()
    thresh_hits = test_df['latency'] > threshold
    ml_hits = rf.predict(test_df[['latency']])
    end_bench = time.perf_counter()

    y_pred_hybrid = (thresh_hits | ml_hits).astype(int)
    y_true = test_df['is_malicious']

    # 5. Metric Calculations (Including Confusion Matrix)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_hybrid).ravel()
    
    avg_latency_s = df['latency'].mean()
    total_proc_time = df['latency'].sum() + (end_bench - start_bench)
    throughput_mb_s = file_size_mb / total_proc_time if total_proc_time > 0 else 0

    # 6. Final Clean Output
    print("\n" + "="*60)
    print(f"HYBRID PERFORMANCE REPORT: {os.path.basename(file_path)}")
    print("="*60)
    print(f"FILE SIZE:          {file_size_mb:.2f} MB")
    print(f"THROUGHPUT:         {throughput_mb_s:.2f} MB/s")
    print(f"AVERAGE LATENCY:    {avg_latency_s:.6f} s/block")
    print("-" * 60)
    print(f"TRUE POSITIVES (TP):  {tp}")
    print(f"TRUE NEGATIVES (TN):  {tn}")
    print(f"FALSE POSITIVES (FP): {fp}")
    print(f"FALSE NEGATIVES (FN): {fn}")
    print("-" * 60)
    print(f"ACCURACY:           {accuracy_score(y_true, y_pred_hybrid):.4f}")
    print(f"RECALL:             {recall_score(y_true, y_pred_hybrid):.4f}")
    print(f"PRECISION:          {precision_score(y_true, y_pred_hybrid):.4f}")
    print(f"F1-SCORE:           {f1_score(y_true, y_pred_hybrid):.4f}")
    print("="*60)

if __name__ == "__main__":
    main()
