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

def process_block_with_sensors(args):
    block, index, inject_anomaly = args
    start_time = time.perf_counter()
    anomaly_type = None
    modified_block = list(block)

    # Base sensor levels
    base_temp = 35.0  # Celsius
    base_optical = 100 # Intensity units
    
    # Hamming Weight for optical simulation
    hw = sum(bin(b).count('1') for b in block)

    if inject_anomaly:
        anomaly_type = random.choice(["delay", "fault"])
        if anomaly_type == "delay":
            time.sleep(random.uniform(0.005, 0.02))
            base_temp -= 2.0 # Throttling/Idle cooling
        elif anomaly_type == "fault":
            modified_block[0] ^= 0xFF
            base_temp += 8.0 # High energy spike from glitch
            base_optical += 50 # Optical flash from fault injection

    cipher = AES.new(KEY, AES.MODE_ECB)
    ciphertext = list(cipher.encrypt(bytes(modified_block)))
    end_time = time.perf_counter()

    return {
        "time": end_time - start_time,
        "byte_sum": sum(ciphertext),
        "thermal": base_temp + random.uniform(-0.5, 0.5), # Add noise
        "optical": base_optical + (hw * 0.5) + random.uniform(-2, 2),
        "is_malicious": 1 if inject_anomaly else 0
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <video.mp4_or_avi>")
        sys.exit(1)

    file_path = sys.argv[1]
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

    results = []
    start_wall = time.perf_counter()
    
    print(f"Analyzing {os.path.basename(file_path)} with Multi-Modal Sensors...")
    with open(file_path, "rb") as f:
        idx = 0
        while True:
            chunk = f.read(BLOCK_SIZE)
            if not chunk: break
            if len(chunk) < BLOCK_SIZE:
                chunk += b'\0' * (BLOCK_SIZE - len(chunk))
            
            results.append(process_block_with_sensors((chunk, idx, random.random() < 0.1)))
            idx += 1
            if idx % 1000 == 0: print(f" Processed {idx} blocks...", end='\r')

    df = pd.DataFrame(results)
    split = int(len(df) * 0.3)
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    
    # Train using ALL features: Time, Data, Thermal, Optical
    features = ['time', 'byte_sum', 'thermal', 'optical']
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(train_df[features], train_df['is_malicious'])
    
    y_pred = rf.predict(test_df[features])
    tn, fp, fn, tp = confusion_matrix(test_df['is_malicious'], y_pred).ravel()
    
    # Console Output
    avg_lat = df['time'].mean()
    throughput = file_size_mb / (time.perf_counter() - start_wall)

    print("\n" + "="*50)
    print(f"MULTI-MODAL REPORT: {os.path.basename(file_path)}")
    print(f"VIDEO SIZE: {file_size_mb:.2f} MB | THROUGHPUT: {throughput:.2f} MB/s")
    print(f"AVG LATENCY: {avg_lat:.6f} sec/block")
    print("-" * 50)
    print(f"TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")
    print(f"PRECISION: {precision_score(test_df['is_malicious'], y_pred):.4f}")
    print(f"RECALL:    {recall_score(test_df['is_malicious'], y_pred):.4f}")
    print(f"F1-SCORE:  {f1_score(test_df['is_malicious'], y_pred):.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
