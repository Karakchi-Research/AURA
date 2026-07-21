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

def process_block_thermal_ml(args):
    block, index, inject_anomaly = args
    start_time = time.perf_counter()
    modified_block = list(block)

    # Simulated Thermal Feature (Celsius)
    # Base temp fluctuates slightly with background noise
    base_temp = 38.0 
    current_temp = base_temp + random.uniform(-0.3, 0.3)

    if inject_anomaly:
        # Simulate a thermal spike caused by a hardware fault/glitch
        # Attacks like voltage glitching or EM injection create localized heat
        current_temp += random.uniform(5.0, 12.0)  
        modified_block[0] ^= 0xFF

    cipher = AES.new(KEY, AES.MODE_ECB)
    cipher.encrypt(bytes(modified_block))
    end_time = time.perf_counter()

    return {
        "thermal_signature": current_temp,
        "latency": end_time - start_time,
        "is_malicious": 1 if inject_anomaly else 0
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <video_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

    # 1. STREAMING DATA COLLECTION
    results = []
    print(f"Analyzing {os.path.basename(file_path)} (THERMAL ML MODE)...")
    
    with open(file_path, "rb") as f:
        idx = 0
        while True:
            chunk = f.read(BLOCK_SIZE)
            if not chunk: break
            if len(chunk) < BLOCK_SIZE: chunk += b'\0' * (BLOCK_SIZE - len(chunk))
            
            results.append(process_block_thermal_ml((chunk, idx, random.random() < 0.1)))
            idx += 1
            if idx % 5000 == 0: print(f" Processed {idx} blocks...", end='\r')

    # 2. DATASET PREPARATION
    df = pd.DataFrame(results)
    
    # Feature: Only Thermal Signature
    X = df[['thermal_signature']]
    y = df['is_malicious']

    # 30% Train (to learn thermal baseline) / 70% Test
    split_idx = int(len(df) * 0.3)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # 3. RANDOM FOREST TRAINING
    print(f"\nTraining Thermal Random Forest on {len(X_train)} samples...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # 4. INFERENCE & METRICS
    y_pred = rf.predict(X_test)
    
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    avg_latency = df['latency'].mean()
    throughput = file_size_mb / (df['latency'].sum() if df['latency'].sum() > 0 else 1)
    
    # 5. FINAL CONSOLE REPORT
    print("\n" + "="*55)
    print(f"THERMAL ML REPORT: {os.path.basename(file_path)}")
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
