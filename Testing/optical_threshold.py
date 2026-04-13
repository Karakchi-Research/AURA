#!/usr/bin/env python3
"""
Optical-Only Threshold Detection Model
Detects AES anomalies using optical measurements (mean pixel intensity from images)
and statistical thresholding.
No timing data. Uses real image files from optical camera.
"""

import sys
import os
import random
import pandas as pd
import psutil

sys.path.insert(0, os.path.dirname(__file__))
from _utils import (
    load_images_from_directory,
    extract_optical_intensity,
    extract_optical_features,
    compute_threshold
)


def main():
    if len(sys.argv) < 2:
        print("Usage: python optical_threshold.py <image_directory> [anomaly_percent]")
        print("  image_directory: path to folder with optical camera images")
        print("  anomaly_percent: percentage of frames to inject optical anomalies (default: 20)")
        sys.exit(1)
    
    image_dir = sys.argv[1]
    anomaly_percent = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    anomaly_ratio = anomaly_percent / 100.0
    
    # Load images
    try:
        images = load_images_from_directory(image_dir)
        num_images = len(images)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"OPTICAL-ONLY THRESHOLD DETECTION")
    print(f"{'='*70}")
    print(f"Dataset: {os.path.basename(image_dir)} ({num_images} images)")
    print(f"Anomaly Injection Rate: {anomaly_percent}%")
    print(f"\nProcessing images...")
    
    # Extract optical signals from images
    results = []
    anomaly_count = 0
    
    for idx, (filename, image) in enumerate(images):
        optical_signal = extract_optical_intensity(image)
        
        # Randomly inject optical anomalies (bright flashes)
        inject_anomaly = random.random() < anomaly_ratio
        if inject_anomaly:
            optical_signal += random.uniform(40, 80)  # Bright flash from glitch
            anomaly_count += 1
        
        results.append({
            "index": idx,
            "filename": filename,
            "optical": optical_signal,
            "is_malicious": 1 if inject_anomaly else 0
        })
    
    # Extract features and apply threshold detection
    df = extract_optical_features(results)
    
    threshold = compute_threshold(df["optical"].tolist())
    df["threshold_detected"] = df["optical"] > threshold
    
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
    print(f"Images Processed: {num_images}")
    print(f"Memory Used: {memory:.2f} MB")
    print(f"\nThreshold: {threshold:.2f} (intensity units)")
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
    output_file = f"optical_threshold_report.xlsx"
    df.to_excel(output_file, index=False)
    print(f"\n📁 Report saved: {output_file}")


if __name__ == "__main__":
    main()
