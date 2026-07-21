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

def process_block_thermal_optical_ml(args):
    """Simulates AES with Thermal and Optical side-channels."""
    block, index, inject_anomaly = args
    start_time = time.perf_counter()
    modified_block = list(block)

    # 1. OPTICAL SENSOR SIMULATION
    # Base + Hamming Weight (HW) + Noise
    base_opt = 100 
    hw = sum(bin(b).count('1') for b in block)
    current_opt = base_opt + (hw * 0.5) + random.uniform(-1, 1)

    # 2. THERMAL SENSOR SIMULATION
    # Base temp + standard fluctuation
    base_temp = 38.0 
    current_temp = base_temp + random.uniform(-0.3, 0.3)

    if inject_anomaly:
        # Choose which sensor the attack targets
        attack_type = random.choice(["thermal_glitch", "optical_laser"])
        if attack_type == "thermal_glitch":
            current_temp += random.uniform(7.0, 15.0) # Heat spike
        else:
            current_opt += random.uniform(40, 80) # Light burst
        modified_block[0] ^= 0xFF # Fault injection

    # Standard Encryption
    cipher = AES.new(KEY, AES.MODE_ECB)
    cipher.encrypt(bytes(modified_block))
    end_time = time.perf_counter()

    return {
        "thermal_sig": current_temp,
        "optical_sig": current_opt,
        "hamming_weight": hw,
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
    print(f"Analyzing {os.path.basename(file_path)} (THERMAL + OPTICAL ML)...")
    
    with open(file_path, "rb") as f:
        idx = 0
        while True:
            chunk = f.read(BLOCK_SIZE)
            if not chunk: break
            if len(chunk) < BLOCK_SIZE: chunk += b'\0' * (BLOCK_SIZE - len(chunk))
            
            results.append(process_block_thermal_optical_ml((chunk, idx, random.random() < 0.1)))
            idx += 1
            if idx % 5000 == 0: print(f" Processed {idx} blocks...", end='\r')

    # 2. DATASET PREPARATION
    df = pd.DataFrame(results)
    
    # Features: Heat, Light, and Data Pattern
    features = ['thermal_sig', 'optical_sig', 'hamming_weight']
    X = df[features]
    y = df['is_malicious']

    # 30% Training / 70% Testing
    split_idx = int(len(df) * 0.3)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # 3. RANDOM FOREST TRAINING
    print(f"\nTraining Multi-Sensor ML Model on {len(X_train)} samples...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # 4. INFERENCE & METRICS
    y_pred = rf.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    avg_lat = df['latency'].mean()
    throughput = file_size_mb / (df['latency'].sum())

    # 5. FINAL REPORT
    print("\n" + "="*55)
    print(f"THERMAL + OPTICAL ML REPORT: {os.path.basename(file_path)}")
    print(f"VIDEO SIZE:    {file_size_mb:.2f} MB")
    print(f"AVG LATENCY:   {avg_lat:.6f} sec/block")
    print(f"THROUGHPUT:    {throughput:.2f} MB/s")
    print("-" * 55)
    print(f"TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")
    print(f"PRECISION: {precision_score(y_test, y_pred):.4f}")
    print(f"RECALL:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-SCORE:  {f1_score(y_test, y_pred):.4f}")
    print("-" * 55)
    
    # Scientific Insight
    importances = rf.feature_importances_
    print("SENSOR IMPORTANCE:")
    for feat, imp in zip(features, importances):
        print(f" - {feat}: {imp:.4f}")
    print("="*55)

if __name__ == "__main__":
    main()
