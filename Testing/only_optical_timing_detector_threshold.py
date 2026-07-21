import pandas as pd
import random
import time
import os
import sys
from Crypto.Cipher import AES

BLOCK_SIZE = 16
KEY = b'ThisIsA16ByteKey'

def process_block_dual_sensor(args):
    block, index, inject_anomaly = args
    start_time = time.perf_counter()
    modified_block = list(block)

    # Base Optical Level + Data-dependent Hamming Weight (HW)
    base_optical = 100 
    hw = sum(bin(b).count('1') for b in block)
    current_optical = base_optical + (hw * 0.5) + random.uniform(-1, 1)

    if inject_anomaly:
        anomaly_choice = random.choice(["delay", "fault"])
        if anomaly_choice == "delay":
            time.sleep(random.uniform(0.005, 0.02)) # Timing Anomaly
        else:
            current_optical += random.uniform(30, 60) # Optical Anomaly
            modified_block = [b ^ 0xFF for b in modified_block]

    cipher = AES.new(KEY, AES.MODE_ECB)
    cipher.encrypt(bytes(modified_block))
    end_time = time.perf_counter()

    return {
        "index": index,
        "time": end_time - start_time,
        "optical": current_optical,
        "is_malicious": 1 if inject_anomaly else 0
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <video_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

    # 1. STREAMING DATA
    results = []
    print(f"Streaming {os.path.basename(file_path)} (TIMING + OPTICAL THRESHOLD)...")
    with open(file_path, "rb") as f:
        idx = 0
        while True:
            chunk = f.read(BLOCK_SIZE)
            if not chunk: break
            if len(chunk) < BLOCK_SIZE: chunk += b'\0' * (BLOCK_SIZE - len(chunk))
            results.append(process_block_dual_sensor((chunk, idx, random.random() < 0.1)))
            idx += 1
            if idx % 2000 == 0: print(f" Processed {idx} blocks...", end='\r')

    # 2. DUAL-CHANNEL BASELINE CALIBRATION (First 20% of video)
    split = int(len(results) * 0.2)
    baseline = results[:split]
    
    # Calculate Mean & Sigma for both sensors
    metrics = ['time', 'optical']
    thresh = {}
    for m in metrics:
        vals = [r[m] for r in baseline]
        mu = sum(vals) / len(vals)
        sigma = (sum((v - mu)**2 for v in vals) / len(vals))**0.5
        thresh[m] = mu + (2.5 * sigma)

    # 3. MULTIMODAL DETECTION (Remaining 80%)
    tp, fp, fn, tn = 0, 0, 0, 0
    for r in results[split:]:
        # Detect if EITHER channel triggers
        detected = (r['time'] > thresh['time'] or r['optical'] > thresh['optical'])
        
        if r['is_malicious'] and detected: tp += 1
        elif not r['is_malicious'] and detected: fp += 1
        elif r['is_malicious'] and not detected: fn += 1
        else: tn += 1

    # 4. FINAL REPORT
    avg_lat = sum(r['time'] for r in results) / len(results)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0

    print("\n" + "="*55)
    print(f"DUAL-MODAL THRESHOLD REPORT: {os.path.basename(file_path)}")
    print(f"SIZE: {file_size_mb:.2f} MB | AVG LATENCY: {avg_lat:.6f} s")
    print("-" * 55)
    print(f"TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")
    print(f"PRECISION: {prec:.4f} | RECALL: {rec:.4f} | F1: {2*(prec*rec)/(prec+rec) if (prec+rec)>0 else 0:.4f}")
    print("="*55)

if __name__ == "__main__":
    main()
