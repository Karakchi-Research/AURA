import pandas as pd
import numpy as np
import random
import time
import os
import sys
from Crypto.Cipher import AES
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score

# --- CORE SIMULATION FUNCTIONS ---
BLOCK_SIZE = 16
KEY = b'ThisIsA16ByteKey'

def process_block_full(args):
    block, index, inject_anomaly = args
    start_time = time.perf_counter()
    modified_block = list(block)
    
    # Base Sensor Levels
    base_opt, base_temp = 100, 38.0
    hw = sum(bin(b).count('1') for b in block)

    if inject_anomaly:
        attack = random.choice(["delay", "fault"])
        if attack == "delay":
            time.sleep(random.uniform(0.005, 0.01))
        else:
            base_temp += random.uniform(5, 10)
            base_opt += random.uniform(30, 60)
            modified_block[0] ^= 0xFF

    cipher = AES.new(KEY, AES.MODE_ECB)
    cipher.encrypt(bytes(modified_block))
    
    return {
        "timing": time.perf_counter() - start_time,
        "optical": base_opt + (hw * 0.5) + random.uniform(-1, 1),
        "thermal": base_temp + random.uniform(-0.2, 0.2),
        "is_malicious": 1 if inject_anomaly else 0
    }

def run_ablation_study(df, file_size_mb):
    """Iterates through sensor combinations and records performance + overhead."""
    cases = [
        ("Timing Only", ["timing"]),
        ("Optical Only", ["optical"]),
        ("Thermal Only", ["thermal"]),
        ("Dual (Timing+Opt)", ["timing", "optical"]),
        ("Dual (Opt+Therm)", ["optical", "thermal"]),
        ("FULL SYSTEM (All)", ["timing", "optical", "thermal"])
    ]
    
    ablation_results = []
    split = int(len(df) * 0.3)
    train_df, test_df = df.iloc[:split], df.iloc[split:]
    
    print("\nRunning Ablation Trials...")
    for name, features in cases:
        # Measure Training + Inference overhead
        start_bench = time.perf_counter()
        
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(train_df[features], train_df['is_malicious'])
        y_pred = rf.predict(test_df[features])
        
        end_bench = time.perf_counter()
        total_bench_time = end_bench - start_bench
        
        # Calculate Metrics
        y_true = test_df['is_malicious']
        avg_lat_us = (total_bench_time / len(test_df)) * 1_000_000 # microseconds
        throughput = file_size_mb / (df['timing'].sum() + total_bench_time)

        ablation_results.append({
            "Configuration": name,
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, zero_division=0),
            "Lat (us/blk)": round(avg_lat_us, 2),
            "TPput (MB/s)": round(throughput, 2)
        })
    
    return pd.DataFrame(ablation_results)

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <video.mp4>")
        return

    file_path = sys.argv[1]
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    
    raw_data = []
    print(f"Streaming {os.path.basename(file_path)}...")
    with open(file_path, "rb") as f:
        idx = 0
        while True:
            chunk = f.read(BLOCK_SIZE)
            if not chunk: break
            if len(chunk) < BLOCK_SIZE: chunk += b'\0' * (BLOCK_SIZE - len(chunk))
            raw_data.append(process_block_full((chunk, idx, random.random() < 0.1)))
            idx += 1
            if idx >= 10000: break # Safety limit

    df = pd.DataFrame(raw_data)
    comparison_table = run_ablation_study(df, file_size_mb)

    print("\n" + "="*85)
    print(f"      ABLATION STUDY SUMMARY: {os.path.basename(file_path)}")
    print("="*85)
    print(comparison_table.to_string(index=False))
    print("="*85)
    
    comparison_table.to_csv("ablation_metrics_summary.csv", index=False)
    print("\nFull metrics saved to 'ablation_metrics_summary.csv'")

if __name__ == "__main__":
    main()
