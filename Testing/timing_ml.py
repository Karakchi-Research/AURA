#!/usr/bin/env python3
"""
Temporal-Dynamics ML Detection Model (LPBF Geometric Monitoring)
Detects geometric anomalies in LPBF thin-wall fabrication using ML on
temporal evolution features. Tracks how thickness, drift, and roughness
change from layer to layer (layer-to-layer dynamics) and uses Random Forest
to classify anomalous temporal patterns.
Based on SMASIS2026 paper methodology with temporal analysis.
"""

import sys
import os
import pandas as pd
import psutil
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope

sys.path.insert(0, os.path.dirname(__file__))
from _utils import (
    load_images_from_directory,
    detect_wall_boundaries,
    detect_wall_instability,
    extract_geometric_features
)


def main():
    if len(sys.argv) < 2:
        print("Usage: python timing_ml.py <image_directory> [temporal_threshold]")
        print("  image_directory: path to folder with LPBF optical camera images")
        print("  temporal_threshold: max acceptable layer-to-layer change % (default: 3)")
        sys.exit(1)
    
    image_dir = sys.argv[1]
    temporal_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    
    # Load images
    try:
        images = load_images_from_directory(image_dir)
        num_images = len(images)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"TEMPORAL-DYNAMICS ML DETECTION (LPBF Geometric Monitoring)")
    print(f"{'='*70}")
    print(f"Dataset: {os.path.basename(image_dir)} ({num_images} images)")
    print(f"Temporal Threshold: ±{temporal_threshold}%")
    print(f"\nProcessing images...")
    
    # Extract geometric features from images
    results = []
    
    for idx, (filename, image) in enumerate(images):
        # Detect wall boundaries and instability metrics
        boundaries = detect_wall_boundaries(image)
        instability = detect_wall_instability(image)
        
        results.append({
            "index": idx,
            "filename": filename,
            "thickness_px": boundaries["thickness_px"],
            "center_x": boundaries["center_x"],
            "left_edge": boundaries["left_edge"],
            "right_edge": boundaries["right_edge"],
            "roughness": instability["roughness"],
            "texture_variance": instability["texture_variance"],
            "contour_irregularity": instability["contour_irregularity"],
            "is_anomaly": 0
        })
    
    # Extract engineered features (including temporal dynamics)
    df = extract_geometric_features(results)
    
    if len(df) < 3:
        print("Error: Not enough valid geometric detections. Check image quality/format.")
        sys.exit(1)
    
    # Set ground truth: anomalies are frames with significant layer-to-layer changes
    baseline_thickness = df["thickness_px"].iloc[0]
    df["is_anomaly"] = (
        (df["thickness_deviation"].abs() > abs(baseline_thickness * temporal_threshold / 100)) |
        (df["roughness"] > df["roughness"].quantile(0.85))
    ).astype(int)
    
    # Feature selection for ML
    feature_cols = ["thickness_deviation", "center_drift", "thickness_rolling_std", 
                    "roughness", "texture_variance", "contour_irregularity"]
    available_features = [col for col in feature_cols if col in df.columns and df[col].std() > 0]
    
    if not available_features:
        print("Warning: No feature variance. Using single temporal feature.")
        available_features = ["thickness_deviation"]
    
    X = df[available_features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    y = df["is_anomaly"]
    
    # Train model
    if y.sum() == 0 or y.sum() == len(y):
        print(f"Warning: No mixed anomaly classes (all {'normal' if y.sum() == 0 else 'anomalous'}).")
        print("Using unsupervised anomaly detection...")
        model = EllipticEnvelope(contamination=0.2, random_state=42)
        model.fit(X_scaled)
        y_pred = (model.predict(X_scaled) == -1).astype(int)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )
        
        model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_scaled)
    
    df["ml_detected"] = y_pred
    
    # Calculate metrics
    actual_anomalies = df["is_anomaly"].sum()
    detected_anomalies = df["ml_detected"].sum()
    tp = sum((df["is_anomaly"] == 1) & (df["ml_detected"] == 1))
    fp = sum((df["is_anomaly"] == 0) & (df["ml_detected"] == 1))
    fn = sum((df["is_anomaly"] == 1) & (df["ml_detected"] == 0))
    tn = sum((df["is_anomaly"] == 0) & (df["ml_detected"] == 0))
    
    accuracy = ((tp + tn) / len(df) * 100) if len(df) > 0 else 0
    precision = (tp / detected_anomalies * 100) if detected_anomalies > 0 else 0
    recall = (tp / actual_anomalies * 100) if actual_anomalies > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Memory info
    memory = round(psutil.Process().memory_info().rss / (1024 ** 2), 2)
    
    # Feature importance (if available)
    feature_info = ""
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            "feature": available_features,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        feature_info = "\nTop 5 Important Temporal Features:\n"
        for idx, row in feature_importance.head(5).iterrows():
            feature_info += f"  {row['feature']}: {row['importance']:.4f}\n"
    
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Images Processed: {num_images}")
    print(f"Valid Geometries Extracted: {len(df)}")
    print(f"Memory Used: {memory:.2f} MB")
    print(f"\nTemporal Anomaly Detection:")
    print(f"  Actual Anomalies: {actual_anomalies}")
    print(f"  Detected Anomalies: {detected_anomalies}")
    print(f"  True Positives: {tp}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Precision: {precision:.2f}%")
    print(f"  Recall: {recall:.2f}%")
    print(f"  F1-Score: {f1:.2f}%")
    print(feature_info)
    
    # Temporal statistics
    print(f"\nTemporal Evolution Statistics:")
    if "thickness_deviation" in df.columns:
        print(f"  Max Thickness Change: {df['thickness_deviation'].abs().max():.2f} px")
        print(f"  Mean Thickness Change: {df['thickness_deviation'].abs().mean():.2f} px")
    if "center_drift" in df.columns:
        print(f"  Max Layer-to-Layer Drift: {df['center_drift'].abs().max():.2f} px")
    
    # Save report
    output_file = f"timing_ml_report.xlsx"
    df.to_excel(output_file, index=False)
    print(f"\n📁 Report saved: {output_file}")


if __name__ == "__main__":
    main()
