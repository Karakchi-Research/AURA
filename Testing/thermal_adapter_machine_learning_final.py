import pandas as pd
import numpy as np
import random
import time
import multiprocessing
import os
import sys
from Crypto.Cipher import AES
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Configuration
BLOCK_SIZE = 16
KEY = b'ThisIsA16ByteKey'

def process_block(args):
    """Simulates AES encryption with potential hardware anomalies."""
    block, index, inject_anomaly = args
    start_time = time.perf_counter()
    anomaly_type = None
    modified_block = list(block)

    if inject_anomaly:
        anomaly_type = random.choice(["delay", "fault"])
        if anomaly_type == "delay":
            time.sleep(random.uniform(0.005, 0.02))
        elif anomaly_type == "fault":
            modified_block[0] ^= 0xFF

    cipher = AES.new(KEY, AES.MODE_ECB)
    ciphertext = list(cipher.encrypt(bytes(modified_block)))
    end_time = time.perf_counter()

    return {
        "time": end_time - start_time,
        "byte_sum": sum(ciphertext),
        "is_malicious": 1 if inject_anomaly else 0
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <video.mp4_or_avi>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)

    # 1. Total Video Size
    file_size_bytes = os.path.getsize(file_path)
    file_size_mb = file_size_bytes / (1024 * 1024)

    # 2. Streaming Data Generation (Unlimited Size Support)
    print(f"Streaming {os.path.basename(file_path)}...")
    results = []
    start_wall = time.perf_counter()
    
    with open(file_path, "rb") as f:
        idx = 0
        while True:
            chunk = f.read(BLOCK_SIZE)
            if not chunk: break
            if len(chunk) < BLOCK_SIZE:
                chunk += b'\0' * (BLOCK_SIZE - len(chunk))
            
            # Simulate real-time stream processing
            res = process_block((chunk, idx, random.random() < 0.1))
            results.append(res)
            idx += 1
            if idx % 1000 == 0: print(f" Processed {idx} blocks...", end='\r')

    end_wall = time.perf_counter()
    df = pd.DataFrame(results)

    # 3. Machine Learning (Random Forest)
    # Using first 30% for training, rest for testing
    split = int(len(df) * 0.3)
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(train_df[['time', 'byte_sum']], train_df['is_malicious'])
    
    # Inference
    y_pred = rf.predict(test_df[['time', 'byte_sum']])
    y_true = test_df['is_malicious']

    # 4. Final Console Output
    avg_latency = df['time'].mean()
    total_time = end_wall - start_wall
    throughput = file_size_mb / total_time

 # Calculate Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print("\n" + "="*50)
    print(f"VIDEO: {os.path.basename(file_path)}")
    print(f"TOTAL SIZE:      {file_size_mb:.2f} MB")
    print(f"AVERAGE LATENCY:  {avg_latency:.6f} sec/block")
    print(f"THROUGHPUT:       {throughput:.2f} MB/s")
    print("-" * 50)
    print(f"TP: {tp} | FP: {fp}")
    print(f"FN: {fn} | TN: {tn}")
    print("-" * 50)
    print(f"PRECISION: {precision_score(y_true, y_pred):.4f}")
    print(f"RECALL:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1-SCORE:  {f1_score(y_true, y_pred):.4f}")
    print("="*50)


if __name__ == "__main__":
    main()
