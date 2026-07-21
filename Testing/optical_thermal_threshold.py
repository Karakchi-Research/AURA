import pandas as pd
import random
import time
import os
import sys
from Crypto.Cipher import AES

# Configuration
BLOCK_SIZE = 16
KEY = b'ThisIsA16ByteKey'

def process_block_multi_modal(args):
    """Simulates AES encryption with Optical and Thermal side-channels."""
    block, index, inject_anomaly = args
    start_time = time.perf_counter()
    modified_block = list(block)

    # 1. SIMULATE OPTICAL SENSOR (Based on Hamming Weight)
    # HW represents light intensity emitted during data processing
    base_optical = 100 
    hw = sum(bin(b).count('1') for b in block)
    current_optical = base_optical + (hw * 0.5) + random.uniform(-1, 1)

    # 2. SIMULATE THERMAL SENSOR (Celsius)
    base_temp = 38.0 
    current_temp = base_temp + random.uniform(-0.3, 0.3)

    if inject_anomaly:
        # Randomly choose between a Thermal spike or an Optical burst
        anomaly_choice = random.choice(["thermal_spike", "optical_burst"])
        if anomaly_choice == "thermal_spike":
            current_temp += random.uniform(8.0, 15.0) # Thermal spike (e.g. fault)
            modified_block[0] ^= 0xFF
        else:
            current_optical += random.uniform(40, 70) # Optical burst (e.g. laser)

    # Standard AES Encryption
    cipher = AES.new(KEY, AES.MODE_ECB)
    cipher.encrypt(bytes(modified_block))
    end_time = time.perf_counter()

    return {
        "index": index,
        "optical": current_optical,
        "thermal": current_temp,
        "time": end_time - start_time,
        "is_malicious": 1 if inject_anomaly else 0
    }

def calculate_multi_thresholds(results):
    """Calculates Mean + 3*Sigma for both sensor channels."""
    thresholds = {}
    for sensor in ['optical', 'thermal']:
        vals = [r[sensor] for r in results]
        mu = sum(vals) / len(vals)
        sigma = (sum((v - mu)**2 for v in vals) / len(vals))**0.5
        thresholds[sensor] = mu + (3.0 * sigma) # Standard Z-score threshold
    return thresholds

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <video_file>")
        return
    
    file_path = sys.argv[1]
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

    # 1. DATA COLLECTION (Streaming for unlimited size)
    results = []
    print(f"Streaming {os.path.basename(file_path)}...")
    with open(file_path, "rb") as f:
        idx = 0
        while True:
            chunk = f.read(BLOCK_SIZE)
            if not chunk: break
            if len(chunk) < BLOCK_SIZE: chunk += b'\0' * (BLOCK_SIZE - len(chunk))
            results.append(process_block_multi_modal((chunk, idx, random.random() < 0.1)))
            idx += 1
            if idx % 5000 == 0: print(f" Processed {idx} blocks...", end='\r')

    # 2. CALIBRATE THRESHOLDS (First 20% of data)
    split = int(len(results) * 0.2)
    thresholds = calculate_multi_thresholds(results[:split])

    # 3. MULTIMODAL DETECTION (Remaining 80%)
    tp, fp, fn, tn = 0, 0, 0, 0
    for r in results[split:]:
        # Detect if EITHER Optical OR Thermal sensor exceeds threshold
        detected = (r['optical'] > thresholds['optical'] or 
                    r['thermal'] > thresholds['thermal'])
        
        if r['is_malicious'] and detected: tp += 1
        elif not r['is_malicious'] and detected: fp += 1
        elif r['is_malicious'] and not detected: fn += 1
        else: tn += 1

    # 4. FINAL PERFORMANCE REPORT
    avg_lat = sum(r['time'] for r in results) / len(results)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0

    print("\n" + "="*55)
    print(f"MULTIMODAL THRESHOLD REPORT: {os.path.basename(file_path)}")
    print(f"SIZE: {file_size_mb:.2f} MB | AVG LATENCY: {avg_lat:.6f} s")
    print("-" * 55)
    print(f"OPTICAL THRESHOLD: {thresholds['optical']:.2f}")
    print(f"THERMAL THRESHOLD: {thresholds['thermal']:.2f} °C")
    print("-" * 55)
    print(f"TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")
    print(f"PRECISION: {prec:.4f} | RECALL: {rec:.4f} | F1: {2*(prec*rec)/(prec+rec) if (prec+rec)>0 else 0:.4f}")
    print("="*55)

if __name__ == "__main__":
    main()
