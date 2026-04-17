#!/usr/bin/env python3
"""
Hybrid Spatial-Temporal ML Detection Model (LPBF Geometric Monitoring)
Detects geometric anomalies in LPBF thin-wall fabrication by fusing:
  - Spatial features: Wall thickness, center position, surface roughness
  - Temporal features: Layer-to-layer changes in thickness and drift
Uses Random Forest classifier on fused multi-modal features.
Based on SMASIS2026 paper methodology with hybrid spatial-temporal ML.
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
        print("Usage: python timing_optical_ml.py <image_directory> [hybrid_threshold]")
        print("  image_directory: path to folder with LPBF optical camera images")
        print("  hybrid_threshold: combined spatial-temporal threshold % (default: 4)")
        sys.exit(1)
    
    image_dir = sys.argv[1]
    hybrid_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
    
    # Load images
    try:
        images = load_images_from_directory(image_dir)
        num_images = len(images)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"HYBRID SPATIAL-TEMPORAL ML DETECTION (LPBF Monitoring)")
    print(f"{'='*70}")
    print(f"Dataset: {os.path.basename(image_dir)} ({num_images} images)")
    print(f"Hybrid Threshold: ±{hybrid_threshold}%")
    print(f"\nProcessing images...")
    
    # Extract geometric features from images
    results = []
    
    for idx, (filename, image) in enumerate(images):
        # Detect wall boundaries (spatial) and instability metrics
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
    
    # Extract engineered features (spatial + temporal)
    df = extract_geometric_features(results)
    
    if len(df) < 3:
        print("Error: Not enough valid geometric detections. Check image quality/format.")
        sys.exit(1)
    
    # Define anomalies based on spatial + temporal criteria
    baseline_thickness = df["thickness_px"].iloc[0]
    df["is_anomaly"] = (
        (df["thickness_pct_change"].abs() > hybrid_threshold) |
        (df["thickness_deviation"].abs() > abs(baseline_thickness * hybrid_threshold / 100)) |
        (df["roughness"] > df["roughness"].quantile(0.85))
    ).astype(int)
    
    # Feature selection: combine spatial and temporal
    feature_cols = ["thickness_pct_change", "thickness_deviation", "center_drift", 
                    "center_drift_accumulation", "thickness_rolling_std", 
                    "roughness", "texture_variance", "contour_irregularity"]
    available_features = [col for col in feature_cols if col in df.columns and df[col].std() > 0]
    
    if not available_features:
        print("Warning: No feature variance. Using defaults.")
        available_features = ["thickness_pct_change", "roughness"]
    
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
        feature_info = "\nTop Features (Spatial + Temporal):\n"
        for idx, row in feature_importance.head(5).iterrows():
            feature_info += f"  {row['feature']}: {row['importance']:.4f}\n"
    
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Images Processed: {num_images}")
    print(f"Valid Geometries Extracted: {len(df)}")
    print(f"Memory Used: {memory:.2f} MB")
    print(f"\nHybrid Anomaly Detection:")
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
    
    # Statistics
    print(f"\nSpatial-Temporal Statistics:")
    if "thickness_px" in df.columns:
        print(f"  Mean Wall Thickness: {df['thickness_px'].mean():.2f} px")
        print(f"  Thickness Variability: {df['thickness_pct_change'].std():.2f}%")
    if "center_drift_accumulation" in df.columns:
        print(f"  Max Cumulative Drift: {df['center_drift_accumulation'].abs().max():.2f} px")
    if "roughness" in df.columns:
        print(f"  Mean Surface Roughness: {df['roughness'].mean():.2f}")
    
    # Save report
    output_file = f"timing_optical_ml_report.xlsx"
    df.to_excel(output_file, index=False)
    print(f"\n📁 Report saved: {output_file}")


if __name__ == "__main__":
    main()
