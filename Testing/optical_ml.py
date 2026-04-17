#!/usr/bin/env python3
"""
Optical-Only ML Detection Model (LPBF Geometric Monitoring)
Detects geometric anomalies in LPBF thin-wall fabrication using optical images.
Extracts wall geometry features (thickness, drift, roughness) and uses Random Forest
to detect instabilities and process anomalies.
Based on SMASIS2026 paper methodology.
"""

import sys
import os
import pandas as pd
import psutil
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _utils import (
    load_images_from_directory,
    detect_wall_boundaries,
    detect_wall_instability,
    extract_geometric_features
)


def main():
    if len(sys.argv) < 2:
        print("Usage: python optical_ml.py <image_directory> [thickness_threshold]")
        print("  image_directory: path to folder with LPBF optical camera images")
        print("  thickness_threshold: acceptable thickness variance in pixels (default: 5)")
        sys.exit(1)
    
    image_dir = sys.argv[1]
    thickness_threshold = float(sys.argv[2]) if len(sys.argv) > 2 else 5.0
    
    # Load images
    try:
        images = load_images_from_directory(image_dir)
        num_images = len(images)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"OPTICAL-ONLY ML DETECTION (LPBF Geometric Monitoring)")
    print(f"{'='*70}")
    print(f"Dataset: {os.path.basename(image_dir)} ({num_images} images)")
    print(f"Thickness Threshold: ±{thickness_threshold}%")
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
            "is_anomaly": 0  # Will be determined by ML model
        })
    
    # Extract engineered features
    df = extract_geometric_features(results)
    
    if len(df) < 3:
        print("Error: Not enough valid geometric detections. Check image quality/format.")
        sys.exit(1)
    
    # Feature scaling for ML
    feature_cols = ["thickness_pct_change", "center_drift_accumulation", 
                    "thickness_rolling_std", "roughness", "texture_variance", 
                    "contour_irregularity"]
    
    # Only use features that exist and have variance
    available_features = [col for col in feature_cols if col in df.columns and df[col].std() > 0]
    
    if not available_features:
        print("Warning: No feature variance detected. Defaulting to single-feature threshold detection.")
        available_features = ["thickness_pct_change"]
    
    X = df[available_features].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Generate anomaly labels based on geometric criteria
    # (In real scenario, these would be manually labeled by domain experts)
    df["is_anomaly"] = (
        (df["thickness_pct_change"].abs() > thickness_threshold) | 
        (df["roughness"] > df["roughness"].quantile(0.85))
    ).astype(int)
    
    y = df["is_anomaly"]
    
    # Only train if we have both classes
    if y.sum() == 0 or y.sum() == len(y):
        print(f"Warning: No mixed anomaly classes in data (all {'normal' if y.sum() == 0 else 'anomalous'}).")
        print("Training model on thickness deviation alone...")
        # Use unsupervised approach
        from sklearn.covariance import EllipticEnvelope
        model = EllipticEnvelope(contamination=0.2, random_state=42)
        model.fit(X_scaled)
        y_pred = (model.predict(X_scaled) == -1).astype(int)  # -1 for anomalies
    else:
        # Train Random Forest with train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )
        
        model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict on full dataset
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
    
    # Feature importance (if Random Forest was used)
    feature_info = ""
    if hasattr(model, 'feature_importances_'):
        feature_importance = pd.DataFrame({
            "feature": available_features,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)
        feature_info = "\nTop 5 Important Features:\n"
        for idx, row in feature_importance.head(5).iterrows():
            feature_info += f"  {row['feature']}: {row['importance']:.4f}\n"
    
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Images Processed: {num_images}")
    print(f"Valid Geometries Extracted: {len(df)}")
    print(f"Memory Used: {memory:.2f} MB")
    print(f"\nGeometric Anomaly Detection:")
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
    
    # Geometric statistics
    print(f"\nGeometric Statistics:")
    if "thickness_px" in df.columns and df["thickness_px"].notna().any():
        print(f"  Mean Wall Thickness: {df['thickness_px'].mean():.2f} px")
        print(f"  Thickness Std Dev: {df['thickness_px'].std():.2f} px")
    if "center_drift_accumulation" in df.columns and df["center_drift_accumulation"].notna().any():
        print(f"  Max Center Drift: {df['center_drift_accumulation'].abs().max():.2f} px")
    if "roughness" in df.columns and df["roughness"].notna().any():
        print(f"  Mean Surface Roughness: {df['roughness'].mean():.2f}")
    
    # Save report
    output_file = f"optical_ml_report.xlsx"
    df.to_excel(output_file, index=False)
    print(f"\n📁 Report saved: {output_file}")


if __name__ == "__main__":
    main()
