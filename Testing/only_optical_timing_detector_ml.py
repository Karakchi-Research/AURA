import pandas as pd
import numpy as np
import random
import time
import os
import sys
from Crypto.Cipher import AES
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

BLOCK_SIZE = 16
KEY = b'ThisIsA16ByteKey'

def process_block_optical_ml(args):
    block, index, inject_anomaly = args
    start_time = time.perf_counter()
    modified_block = list(block)

    # Simulated Optical Signal: Base + Hamming Weight (HW)
    base_optical = 100 
    hw = sum(bin(b).count('1') for b in block)
    
    # Optical signature = Base + Data-dependent emission + Noise
    current_optical = base_optical + (hw * 0.5) + random.uniform(-1, 1)

    if inject_anomaly:
        # Simulate Optical Signature of a Fault Attack (Burst)
        current_optical += random.uniform(30, 60)  
        modified_block[0] ^= 0xFF

    cipher = AES.new(KEY, AES.MODE_ECB)
    cipher.encrypt(bytes(modified_block))
    end_time = time.perf_counter()

    return {
        "optical_signal": current_optical,
        "hamming_weight": hw,
        "latency": end_time - start_time,
        "is_malicious": 1 if inject_anomaly else 0
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <video_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

    # 1. STREAMING DATA COLLECTION
    results = []
    print(f"Analyzing {os.path.basename(file_path)} (OPTICAL ML MODE)...")
    
    with open(file_path, "rb") as f:
        idx = 0
        while True:
            chunk = f.read(BLOCK_SIZE)
            if not chunk: break
            if len(chunk) < BLOCK_SIZE: chunk += b'\0' * (BLOCK_SIZE - len(chunk))
            
            results.append(process_block_optical_ml((chunk, idx, random.random() < 0.1)))
            idx += 1
            if idx % 5000 == 0: print(f" Processed {idx} blocks...", end='\r')

    # 2. DATASET PREPARATION
    df = pd.DataFrame(results)
    
    # Features: Only Optical Intensity and Data Hamming Weight
    X = df[['optical_signal', 'latency']]
    y = df['is_malicious']

    # Split: 30% for Training, 70% for Testing
    split_idx = int(len(df) * 0.3)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # 3. RANDOM FOREST TRAINING
    print(f"\nTraining Optical Random Forest on {len(X_train)} samples...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # 4. INFERENCE & METRICS
    start_inf = time.perf_counter()
    y_pred = rf.predict(X_test)
    end_inf = time.perf_counter()
    
    # Calculations
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    avg_latency = df['latency'].mean()
    throughput = file_size_mb / (df['latency'].sum())
    
    # Console Output
    print("\n" + "="*55)
    print(f"OPTICAL ML REPORT: {os.path.basename(file_path)}")
    print(f"VIDEO SIZE:    {file_size_mb:.2f} MB")
    print(f"AVG LATENCY:   {avg_latency:.6f} sec/block")
    print(f"THROUGHPUT:    {throughput:.2f} MB/s")
    print("-" * 55)
    print(f"TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")
    print(f"PRECISION: {precision_score(y_test, y_pred):.4f}")
    print(f"RECALL:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-SCORE:  {f1_score(y_test, y_pred):.4f}")
    print("="*55)

if __name__ == "__main__":
    main()
