import pandas as pd
import numpy as np
import random
import time
import os
import sys
from Crypto.Cipher import AES
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# Configuration
BLOCK_SIZE = 16
KEY = b'ThisIsA16ByteKey'

def get_hybrid_features(args):
    """Simulates multimodal sensor data for a 16-byte block."""
    block, index, inject_anomaly = args
    start_time = time.perf_counter()
    modified_block = list(block)
    
    # Base sensors (simulated hardware baselines)
    base_temp, base_opt = 38.0, 100 
    # Hamming Weight: count of '1' bits in the block
    hw = bin(int.from_bytes(block, byteorder='big')).count('1')

    if inject_anomaly:
        attack = random.choice(["delay", "fault", "stealth"])
        if attack == "delay":
            time.sleep(random.uniform(0.005, 0.01))
        elif attack == "fault":
            base_temp += random.uniform(8, 15) # High thermal spike
            base_opt += random.uniform(40, 70) # Optical burst
            modified_block[0] ^= 0xFF
        else: # Stealth: Subtle anomalies that stay near thresholds
            base_temp += 1.5 
            base_opt += 5.0 

    cipher = AES.new(KEY, AES.MODE_ECB)
    cipher.encrypt(bytes(modified_block))
    latency = time.perf_counter() - start_time
    
    return {
        "timing": latency,
        "thermal": base_temp + random.uniform(-0.2, 0.2),
        "optical": base_opt + (hw * 0.5) + random.uniform(-1, 1),
        "is_malicious": 1 if inject_anomaly else 0
    }

def main():
    # Fix: Correctly grab the file path string from CLI
    if len(sys.argv) < 2:
        print("Usage: python3 script.py <video_file>")
        return

    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

    # 1. DATA COLLECTION (Streaming)
    raw_results = []
    print(f"Analyzing {os.path.basename(file_path)}...")
    
    with open(file_path, "rb") as f:
        idx = 0
        while idx < 15000: # Increased block limit for better ML training
            chunk = f.read(BLOCK_SIZE)
            if not chunk: break
            if len(chunk) < BLOCK_SIZE: chunk += b'\0' * (BLOCK_SIZE - len(chunk))
            raw_results.append(get_hybrid_features((chunk, idx, random.random() < 0.1)))
            idx += 1
            if idx % 2000 == 0: print(f" Read {idx} blocks...", end='\r')

    df = pd.DataFrame(raw_results)
    split = int(len(df) * 0.3)
    train_df, test_df = df.iloc[:split].copy(), df.iloc[split:].copy()

    # 2. LAYER 1: CALIBRATE THRESHOLDS (Statistical Baseline)
    metrics = ['timing', 'thermal', 'optical']
    # Calculate thresholds as Mean + 3*StdDev
    threshold_values = {m: train_df[m].mean() + (3 * train_df[m].std()) for m in metrics}

    # 3. LAYER 2: TRAIN ML MODEL (Contextual Intelligence)
    print("\nTraining Multi-Modal Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(train_df[metrics], train_df['is_malicious'])

    # 4. HYBRID DETECTION (Vectorized Speed Optimization)
    print("Executing Vectorized Hybrid Inference...")
    
    # Layer 1: Threshold-based logic (Vectorized)
    # Check if ANY of the three sensors exceed their specific thresholds
    thresh_hits = (test_df['timing'] > threshold_values['timing']) | \
                  (test_df['thermal'] > threshold_values['thermal']) | \
                  (test_df['optical'] > threshold_values['optical'])
    
    # Layer 2: ML-based logic (Predict entire batch at once)
    ml_hits = rf.predict(test_df[metrics])

    # Final Hybrid Decision: Logical OR (Flag if Threshold OR ML detects it)
    y_pred_hybrid = (thresh_hits.values | ml_hits).astype(int)
    y_true = test_df['is_malicious'].values

    # 5. METRICS & CONSOLE REPORT
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_hybrid).ravel()
    avg_lat = df['timing'].mean()
    total_time_enc = df['timing'].sum()
    throughput = file_size_mb / total_time_enc if total_time_enc > 0 else 0

    print("\n" + "="*65)
    print(f"HYBRID PERFORMANCE REPORT: {os.path.basename(file_path)}")
    print("="*65)
    print(f"FILE SIZE:      {file_size_mb:.2f} MB")
    print(f"AVG LATENCY:    {avg_lat:.6f} sec/block")
    print(f"THROUGHPUT:     {throughput:.2f} MB/s")
    print("-" * 65)
    print(f"TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")
    print(f"PRECISION:      {precision_score(y_true, y_pred_hybrid):.4f}")
    print(f"RECALL:         {recall_score(y_true, y_pred_hybrid):.4f}")
    print(f"F1-SCORE:       {f1_score(y_true, y_pred_hybrid):.4f}")
    print("-" * 65)
    print(f"THRESHOLD (T):  {threshold_values['timing']:.4f} s")
    print(f"THRESHOLD (H):  {threshold_values['thermal']:.1f} °C")
    print(f"THRESHOLD (O):  {threshold_values['optical']:.1f} units")
    print("="*65)

    # Optional: Save results to CSV for plotting
    test_df['hybrid_pred'] = y_pred_hybrid
    test_df.to_csv("hybrid_detection_results.csv", index=False)
    print("\n[✔] Detailed detection log saved to 'hybrid_detection_results.csv'")

if __name__ == "__main__":
    main()
