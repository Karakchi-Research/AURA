#!/usr/bin/env python3
"""
Timing + Optical ML Detection Model
Detects AES anomalies by fusing timing and optical measurements
with Random Forest classifier.
Combines AES simulation (timing) with real images (optical).
"""

import sys
import os
import time
import random
import multiprocessing
import pandas as pd
import psutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from _utils import (
    load_images_from_directory,
    simulate_aes_block_with_anomaly,
    extract_optical_intensity,
    extract_fused_features
)


def process_image_with_timing(args):
    """
    Process image frame with simultaneous AES encryption (timing measurement).
    Inject anomalies that affect both timing and optical channels.
    """
    image_index, optical_signal, inject_anomaly = args
    
    # Simulate AES block encryption
    block = bytes([random.randint(0, 255) for _ in range(16)])
    result = simulate_aes_block_with_anomaly(block, image_index, inject_anomaly)
    
    # If anomaly injected, it affects optical signal too
    final_optical = optical_signal
    if inject_anomaly:
        final_optical += random.uniform(40, 80)  # Bright flash from fault
    
    return {
        "index": result["index"],
        "timing": result["timing"],
        "optical": final_optical,
        "is_malicious": result["is_malicious"]
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python timing_optical_ml.py <image_directory> [anomaly_percent] [num_workers]")
        print("  image_directory: path to folder with optical camera images")
        print("  anomaly_percent: percentage of frames to inject anomalies (default: 20)")
        print("  num_workers: number of CPU cores (default: 4)")
        sys.exit(1)
    
    image_dir = sys.argv[1]
    anomaly_percent = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
    num_workers = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    anomaly_ratio = anomaly_percent / 100.0
    
    # Load images
    try:
        images = load_images_from_directory(image_dir)
        num_images = len(images)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"TIMING + OPTICAL ML DETECTION (FUSED)")
    print(f"{'='*70}")
    print(f"Dataset: {os.path.basename(image_dir)} ({num_images} images)")
    print(f"Anomaly Rate: {anomaly_percent}%")
    print(f"Workers: {num_workers}")
    print(f"\nExtracting optical signals from images...")
    
    # Extract optical signals from all images
    optical_signals = []
    for _, image in images:
        optical_signals.append(extract_optical_intensity(image))
    
    # Generate tasks with anomaly injection
    tasks = []
    anomaly_count = 0
    for i in range(num_images):
        inject = random.random() < anomaly_ratio
        if inject:
            anomaly_count += 1
        tasks.append((i, optical_signals[i], inject))
    
    print(f"Frames to Process: {num_images}")
    print(f"Expected Anomalies: {anomaly_count}")
    print(f"\nProcessing...")
    
    start_time = time.time()
    
    # Process with multiprocessing
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.map(process_image_with_timing, tasks)
    
    total_time = time.time() - start_time
    
    # Extract fused features
    df = extract_fused_features(results)
    
    # Train/test split
    X = df.drop(columns=["index", "is_malicious"])
    y = df["is_malicious"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict on full dataset
    y_pred = model.predict(X)
    df["ml_detected"] = y_pred
    
    # Calculate metrics
    actual_anomalies = sum(1 for r in results if r["is_malicious"])
    detected_anomalies = sum(1 for pred in y_pred if pred)
    tp = sum(1 for i, row in df.iterrows() if row["is_malicious"] and row["ml_detected"])
    fp = sum(1 for i, row in df.iterrows() if not row["is_malicious"] and row["ml_detected"])
    fn = sum(1 for i, row in df.iterrows() if row["is_malicious"] and not row["ml_detected"])
    
    accuracy = (tp / actual_anomalies * 100) if actual_anomalies > 0 else 0
    precision = (tp / detected_anomalies * 100) if detected_anomalies > 0 else 0
    recall = (tp / actual_anomalies * 100) if actual_anomalies > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Memory info
    memory = round(psutil.Process().memory_info().rss / (1024 ** 2), 2)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)
    
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Total Time: {total_time:.4f} sec")
    print(f"Avg Latency: {(total_time/num_images)*1e6:.2f} µs/frame")
    print(f"Memory Used: {memory:.2f} MB")
    print(f"\nAnomaly Detection:")
    print(f"  Actual Anomalies: {actual_anomalies}")
    print(f"  Detected Anomalies: {detected_anomalies}")
    print(f"  True Positives: {tp}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Precision: {precision:.2f}%")
    print(f"  Recall: {recall:.2f}%")
    print(f"  F1-Score: {f1:.2f}%")
    print(f"\nTop 5 Important Features:")
    for idx, row in feature_importance.head(5).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    # Save report
    output_file = f"timing_optical_ml_report.xlsx"
    df.to_excel(output_file, index=False)
    print(f"\n📁 Report saved: {output_file}")


if __name__ == "__main__":
    main()
