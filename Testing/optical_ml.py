#!/usr/bin/env python3
"""
Optical-Only ML Detection Model
Detects AES anomalies using optical measurements (mean pixel intensity from images)
and Random Forest classifier.
No timing data. Uses real image files from optical camera.
"""

import sys
import os
import random
import pandas as pd
import psutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from _utils import (
    load_images_from_directory,
    extract_optical_intensity,
    extract_optical_features
)


def main():
    if len(sys.argv) < 2:
        print("Usage: python optical_ml.py <image_directory> [anomaly_percent]")
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
    print(f"OPTICAL-ONLY ML DETECTION")
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
    
    # Extract features
    df = extract_optical_features(results)
    
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
    print(f"Images Processed: {num_images}")
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
    output_file = f"optical_ml_report.xlsx"
    df.to_excel(output_file, index=False)
    print(f"\n📁 Report saved: {output_file}")


if __name__ == "__main__":
    main()
