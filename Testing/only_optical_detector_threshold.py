import pandas as pd
import random
import time
import os
import sys
from Crypto.Cipher import AES

BLOCK_SIZE = 16
KEY = b'ThisIsA16ByteKey'

def process_block_optical_only(args):
    block, index, inject_anomaly = args
    start_time = time.perf_counter()
    modified_block = list(block)

    # Base Optical Level + Hamming Weight (HW) simulation
    # HW represents the number of '1' bits in the data block
    base_optical = 100 
    hw = sum(bin(b).count('1') for b in block)
    
    # Current optical signature = Base + Data-dependent emission + Noise
    current_optical = base_optical + (hw * 0.5) + random.uniform(-1, 1)

    if inject_anomaly:
        # Simulate an Optical Flash/Burst caused by a Fault Injection (e.g., Laser)
        current_optical += 50  
        modified_block[0] ^= 0xFF

    cipher = AES.new(KEY, AES.MODE_ECB)
    cipher.encrypt(bytes(modified_block))
    end_time = time.perf_counter()

    return {
        "index": index,
        "optical": current_optical,
        "time": end_time - start_time,
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
    print(f"Analyzing {os.path.basename(file_path)} (OPTICAL THRESHOLD MODE)...")
    
    with open(file_path, "rb") as f:
        idx = 0
        while True:
            chunk = f.read(BLOCK_SIZE)
            if not chunk: break
            if len(chunk) < BLOCK_SIZE: chunk += b'\0' * (BLOCK_SIZE - len(chunk))
            
            results.append(process_block_optical_only((chunk, idx, random.random() < 0.1)))
            idx += 1
            if idx % 2000 == 0: print(f" Processed {idx} blocks...", end='\r')

    # 2. OPTICAL BASELINE CALIBRATION (First 20% of video)
    split = int(len(results) * 0.2)
    baseline_vals = [r['optical'] for r in results[:split]]
    
    mu_opt = sum(baseline_vals) / len(baseline_vals)
    sigma_opt = (sum((v - mu_opt)**2 for v in baseline_vals) / len(baseline_vals))**0.5
    
    # Optical Threshold (Mean + 3 * StdDev)
    threshold_opt = mu_opt + (3.0 * sigma_opt)

    # 3. DETECTION PHASE (Remaining 80% of video)
    test_data = results[split:]
    tp, fp, fn, tn = 0, 0, 0, 0
    
    for r in test_data:
        # Binary Decision based ONLY on Optical Sensor
        detected = r['optical'] > threshold_opt
        
        if r['is_malicious'] and detected: tp += 1
        elif not r['is_malicious'] and detected: fp += 1
        elif r['is_malicious'] and not detected: fn += 1
        else: tn += 1

    # 4. FINAL CONSOLE OUTPUT
    avg_latency = sum(r['time'] for r in results) / len(results)
    throughput = file_size_mb / sum(r['time'] for r in results)
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

    print("\n" + "="*55)
    print(f"OPTICAL SENSOR REPORT: {os.path.basename(file_path)}")
    print(f"FILE SIZE:     {file_size_mb:.2f} MB")
    print(f"AVG LATENCY:   {avg_latency:.6f} sec/block")
    print(f"THROUGHPUT:    {throughput:.2f} MB/s")
    print("-" * 55)
    print(f"OPTICAL THRESHOLD: {threshold_opt:.2f} units")
    print("-" * 55)
    print(f"TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")
    print(f"PRECISION: {prec:.4f} | RECALL: {rec:.4f} | F1: {f1:.4f}")
    print("="*55)

if __name__ == "__main__":
    main()
