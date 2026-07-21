import pandas as pd
import random
import time
import multiprocessing
import sys
import os
from Crypto.Cipher import AES

BLOCK_SIZE = 16
KEY = b'ThisIsA16ByteKey'

def aes_encrypt_block_with_anomaly(args):
    block, index, inject_anomaly = args
    start_time = time.perf_counter() 
    anomaly_type = None
    modified_block = list(block)

    if inject_anomaly:
        anomaly_type = random.choice(["delay", "fault"])
        if anomaly_type == "delay":
            time.sleep(random.uniform(0.005, 0.02))
        elif anomaly_type == "fault":
            modified_block[0] ^= 0xFF

    byte_data = bytes(modified_block)
    cipher = AES.new(KEY, AES.MODE_ECB)
    ciphertext = cipher.encrypt(byte_data)
    end_time = time.perf_counter()

    return {
        "index": index,
        "anomaly_type": anomaly_type,
        "time": end_time - start_time,
        "detected_as_malicious": False 
    }

def read_video_file(file_path):
    file_size_bytes = os.path.getsize(file_path)
    with open(file_path, "rb") as f:
        data = f.read()
    
    blocks = []
    for i in range(0, len(data), BLOCK_SIZE):
        chunk = data[i:i + BLOCK_SIZE]
        if len(chunk) < BLOCK_SIZE:
            chunk += b'\x00' * (BLOCK_SIZE - len(chunk))
        blocks.append(list(chunk))
    
    return blocks, file_size_bytes

def detect_anomalies(results, threshold=None):
    times = [r["time"] for r in results]
    if threshold is None:
        mean_time = sum(times) / len(times)
        std_dev = (sum((t - mean_time) ** 2 for t in times) / len(times)) ** 0.5
        threshold = mean_time + (2 * std_dev)

    for r in results:
        r["detected_as_malicious"] = r["time"] > threshold
    return results, threshold

def calculate_metrics(results, name, file_size_bytes=None, elapsed_wall_time=None):
    tp = sum(1 for r in results if r["anomaly_type"] and r["detected_as_malicious"])
    fp = sum(1 for r in results if not r["anomaly_type"] and r["detected_as_malicious"])
    fn = sum(1 for r in results if r["anomaly_type"] and not r["detected_as_malicious"])
    tn = sum(1 for r in results if not r["anomaly_type"] and not r["detected_as_malicious"])
    
    avg_latency = sum(r["time"] for r in results) / len(results)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # CONSOLE OUTPUT
    print("\n" + "="*55)
    print(f"      {name} PERFORMANCE SUMMARY")
    print("="*55)
    if file_size_bytes:
        file_mb = file_size_bytes / (1024*1024)
        print(f"TOTAL VIDEO SIZE:      {file_mb:.2f} MB")
        if elapsed_wall_time:
            throughput = file_mb / elapsed_wall_time
            print(f"THROUGHPUT:            {throughput:.2f} MB/s")
    
    print(f"AVERAGE LATENCY:       {avg_latency:.6f} seconds")
    print("-" * 55)
    print(f"Precision:             {precision:.4f} | Recall: {recall:.4f}")
    print(f"F1-Score:              {f1:.4f}")
    print(f"TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")
    print("="*55)

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <path_to_video.mp4_or_avi>")
        return
    
    file_path = sys.argv[1]
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext not in ['.mp4', '.avi', '.ravi']:
        print(f"Error: Unsupported file format '{ext}'. Use .mp4 or .avi")
        return

    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    # 1. READ FILE
    print(f"Reading {ext.upper()} file: {os.path.basename(file_path)}...")
    all_raw_blocks, file_size_bytes = read_video_file(file_path)
    
    total_blocks = len(all_raw_blocks)
    split_idx = int(total_blocks * 0.2)
    
    train_data = [(all_raw_blocks[i], i, random.random() < 0.5) for i in range(split_idx)]
    test_data = [(all_raw_blocks[i], i, random.random() < 0.1) for i in range(split_idx, total_blocks)]

    num_cores = multiprocessing.cpu_count()

    # 2. TRAINING
    print(f"Establishing baseline on {len(train_data)} blocks...")
    with multiprocessing.Pool(num_cores) as pool:
        train_results = pool.map(aes_encrypt_block_with_anomaly, train_data)
    
    train_results, threshold = detect_anomalies(train_results)
    calculate_metrics(train_results, "TRAINING")

    # 3. TESTING
    print(f"Running inference on {len(test_data)} blocks...")
    start_wall = time.perf_counter()
    with multiprocessing.Pool(num_cores) as pool:
        test_results = pool.map(aes_encrypt_block_with_anomaly, test_data)
    end_wall = time.perf_counter()
    
    test_results, _ = detect_anomalies(test_results, threshold=threshold)
    calculate_metrics(test_results, "TESTING", file_size_bytes, (end_wall - start_wall))

    # 4. CSV EXPORT
    pd.DataFrame(test_results).to_csv("video_anomaly_data.csv", index=False)
    print(f"\nFinal Threshold: {threshold:.6f} sec")
    print("Detailed raw data saved to 'video_anomaly_data.csv'")

if __name__ == "__main__":
    main()
