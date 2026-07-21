import pandas as pd
import random
import time
import os
import sys
from Crypto.Cipher import AES

BLOCK_SIZE = 16
KEY = b'ThisIsA16ByteKey'

def process_block_with_sensors(args):
    block, index, inject_anomaly = args
    start_time = time.perf_counter()
    anomaly_type = None
    modified_block = list(block)

    # Base levels + Hamming Weight for optical
    base_temp, base_optical = 35.0, 100 
    hw = sum(bin(b).count('1') for b in block)

    if inject_anomaly:
        anomaly_type = random.choice(["delay", "fault"])
        if anomaly_type == "delay":
            time.sleep(random.uniform(0.005, 0.02))
            base_temp -= 2.0
        elif anomaly_type == "fault":
            modified_block[0] ^= 0xFF
            base_temp += 8.0 # Thermal spike
            base_optical += 50 # Optical burst

    cipher = AES.new(KEY, AES.MODE_ECB)
    ciphertext = list(cipher.encrypt(bytes(modified_block)))
    end_time = time.perf_counter()

    return {
        "index": index,
        "time": end_time - start_time,
        "thermal": base_temp + random.uniform(-0.5, 0.5),
        "optical": base_optical + (hw * 0.5) + random.uniform(-2, 2),
        "is_malicious": 1 if inject_anomaly else 0
    }

def calculate_thresholds(results):
    """Establishes baseline mu and sigma for all 3 sensors."""
    metrics = ['time', 'thermal', 'optical']
    thresholds = {}
    for m in metrics:
        values = [r[m] for r in results]
        mu = sum(values) / len(values)
        sigma = (sum((v - mu)**2 for v in values) / len(values))**0.5
        thresholds[m] = mu + (2.5 * sigma) # Use 2.5 for balanced sensitivity
    return thresholds

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <video.mp4_or_avi>")
        sys.exit(1)

    file_path = sys.argv[1]
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

    # 1. STREAMING PROCESS
    results = []
    start_wall = time.perf_counter()
    print(f"Streaming {os.path.basename(file_path)}...")
    
    with open(file_path, "rb") as f:
        idx = 0
        while True:
            chunk = f.read(BLOCK_SIZE)
            if not chunk: break
            if len(chunk) < BLOCK_SIZE: chunk += b'\0' * (BLOCK_SIZE - len(chunk))
            results.append(process_block_with_sensors((chunk, idx, random.random() < 0.1)))
            idx += 1
            if idx % 1000 == 0: print(f" Processed {idx} blocks...", end='\r')

    # 2. SPLIT & THRESHOLDING
    split = int(len(results) * 0.2) # Use first 20% as Baseline
    train_res = results[:split]
    test_res = results[split:]

    thresholds = calculate_thresholds(train_res)

    # 3. MULTI-MODAL DETECTION LOGIC
    # Flag as anomaly if ANY sensor exceeds its threshold
    tp, fp, fn, tn = 0, 0, 0, 0
    for r in test_res:
        detected = (r['time'] > thresholds['time'] or 
                    r['thermal'] > thresholds['thermal'] or 
                    r['optical'] > thresholds['optical'])
        
        if r['is_malicious'] and detected: tp += 1
        elif not r['is_malicious'] and detected: fp += 1
        elif r['is_malicious'] and not detected: fn += 1
        else: tn += 1

    # 4. FINAL REPORT
    avg_lat = sum(r['time'] for r in results) / len(results)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0

    print("\n" + "="*50)
    print(f"THRESHOLD REPORT: {os.path.basename(file_path)}")
    print(f"SIZE: {file_size_mb:.2f} MB | AVG LATENCY: {avg_lat:.6f} s")
    print("-" * 50)
    print(f"THRESHOLDS: Time={thresholds['time']:.4f}, Therm={thresholds['thermal']:.1f}, Opt={thresholds['optical']:.1f}")
    print("-" * 50)
    print(f"TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")
    print(f"PRECISION: {prec:.4f} | RECALL: {rec:.4f} | F1: {2*(prec*rec)/(prec+rec) if (prec+rec)>0 else 0:.4f}")
    print("="*50)

if __name__ == "__main__":
    main()
