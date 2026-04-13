#!/usr/bin/env python3
"""
Timing-Only Threshold Detection Model
Detects AES anomalies using timing measurements and statistical thresholding.
No optical data. No video. Simulated timing + optional injected anomalies.
"""

import sys
import os
import time
import random
import multiprocessing
import pandas as pd
import psutil
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from _utils import (
    load_images_from_directory,
    simulate_aes_block_with_anomaly,
    extract_timing_features,
    compute_threshold
)


def process_image_as_block(args):
    """
    Treat each image index as a "data frame" generating one AES block to encrypt.
    Simulate timing with optional anomaly injection.
    """
    image_index, num_images, inject_anomaly = args
    
    # Generate synthetic plaintext block (16 bytes)
    block = bytes([random.randint(0, 255) for _ in range(16)])
    
    result = simulate_aes_block_with_anomaly(block, image_index, inject_anomaly)
    return result


def main():
    if len(sys.argv) < 2:
        print("Usage: python timing_threshold.py <image_directory> [anomaly_percent] [num_workers]")
        print("  image_directory: path to folder with optical camera images")
        print("  anomaly_percent: percentage of blocks to inject anomalies (default: 20)")
        print("  num_workers: number of CPU cores (default: 4)")
        sys.exit(1)
    
    image_dir = sys.argv[1]
    anomaly_percent = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    num_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    
    anomaly_ratio = anomaly_percent / 100.0
    
    # Load images to determine dataset size
    try:
        images = load_images_from_directory(image_dir)
        num_images = len(images)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"TIMING-ONLY THRESHOLD DETECTION")
    print(f"{'='*70}")
    print(f"Dataset: {os.path.basename(image_dir)} ({num_images} images)")
    print(f"Anomaly Rate: {anomaly_percent}%")
    print(f"Workers: {num_workers}")
    
    # Generate block processing tasks
    tasks = []
    anomaly_count = 0
    for i in range(num_images):
        inject = random.random() < anomaly_ratio
        if inject:
            anomaly_count += 1
        tasks.append((i, num_images, inject))
    
    print(f"Blocks to Process: {num_images}")
    print(f"Expected Anomalies: {anomaly_count}")
    print(f"\nProcessing...")
    
    start_time = time.time()
    
    # Process blocks with multiprocessing
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(process_image_as_block, tasks)
    
    total_time = time.time() - start_time
    
    # Extract timing features and apply threshold detection
    df = extract_timing_features(results)
    
    threshold = compute_threshold(df["timing"].tolist())
    df["threshold_detected"] = df["timing"] > threshold
    
    # Calculate metrics
    actual_anomalies = sum(1 for r in results if r["is_malicious"])
    detected_anomalies = sum(1 for d in df["threshold_detected"] if d)
    tp = sum(1 for i, row in df.iterrows() if row["is_malicious"] and row["threshold_detected"])
    fp = sum(1 for i, row in df.iterrows() if not row["is_malicious"] and row["threshold_detected"])
    fn = sum(1 for i, row in df.iterrows() if row["is_malicious"] and not row["threshold_detected"])
    
    accuracy = (tp / actual_anomalies * 100) if actual_anomalies > 0 else 0
    precision = (tp / detected_anomalies * 100) if detected_anomalies > 0 else 0
    recall = (tp / actual_anomalies * 100) if actual_anomalies > 0 else 0
    
    # Memory info
    memory = round(psutil.Process().memory_info().rss / (1024 ** 2), 2)
    
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Total Time: {total_time:.4f} sec")
    print(f"Avg Latency: {(total_time/num_images)*1e6:.2f} µs/block")
    print(f"Memory Used: {memory:.2f} MB")
    print(f"\nThreshold: {threshold:.6f} sec")
    print(f"\nAnomaly Detection:")
    print(f"  Actual Anomalies: {actual_anomalies}")
    print(f"  Detected Anomalies: {detected_anomalies}")
    print(f"  True Positives: {tp}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Precision: {precision:.2f}%")
    print(f"  Recall: {recall:.2f}%")
    
    # Save report
    output_file = f"timing_threshold_report.xlsx"
    df.to_excel(output_file, index=False)
    print(f"\n📁 Report saved: {output_file}")


if __name__ == "__main__":
    main()
