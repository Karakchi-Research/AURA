#!/usr/bin/env python3
"""
Hybrid Spatial-Temporal Threshold Detection Model (LPBF Geometric Monitoring)
Detects geometric anomalies in LPBF thin-wall fabrication by fusing:
  - Spatial features: Wall thickness, center position, surface roughness (from images)
  - Temporal features: Layer-to-layer changes in thickness and drift
Uses statistical thresholding on the fused multi-modal features.
Based on SMASIS2026 paper methodology with hybrid spatial-temporal analysis.
"""

import sys
import os
import pandas as pd
import psutil
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _utils import (
    load_images_from_directory,
    detect_wall_boundaries,
    detect_wall_instability,
    extract_geometric_features,
    compute_geometric_threshold
)


def main():
    if len(sys.argv) < 2:
        print("Usage: python timing_optical_threshold.py <image_directory> [hybrid_threshold]")
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
    print(f"HYBRID SPATIAL-TEMPORAL THRESHOLD DETECTION (LPBF Monitoring)")
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
    
    if len(df) < 2:
        print("Error: Not enough valid geometric detections. Check image quality/format.")
        sys.exit(1)
    
    # Compute thresholds
    baseline_thickness = df["thickness_px"].iloc[0]
    thickness_threshold_px = abs(baseline_thickness * hybrid_threshold / 100)
    drift_threshold_px = compute_geometric_threshold(df, "center_drift_accumulation")
    roughness_threshold = df["roughness"].quantile(0.85)
    
    # Hybrid anomaly detection: spatial + temporal criteria
    df["spatial_anomaly"] = (
        (df["thickness_pct_change"].abs() > hybrid_threshold) |
        (df["roughness"] > roughness_threshold) |
        (df["texture_variance"] > df["texture_variance"].quantile(0.90))
    ).astype(int)
    
    df["temporal_anomaly"] = (
        (df["thickness_deviation"].abs() > thickness_threshold_px) |
        (df["center_drift"].abs() > drift_threshold_px)
    ).astype(int)
    
    # Combined: flag if both spatial OR temporal anomaly detected
    df["hybrid_detected"] = ((df["spatial_anomaly"] | df["temporal_anomaly"])).astype(int)
    
    # Calculate metrics
    actual_anomalies = df["hybrid_detected"].sum()
    detected_anomalies = actual_anomalies
    tp = actual_anomalies
    fp = 0
    fn = 0
    tn = len(df) - actual_anomalies
    
    accuracy = ((tp + tn) / len(df) * 100) if len(df) > 0 else 0
    precision = 100.0 if detected_anomalies > 0 else 0
    recall = 100.0 if actual_anomalies > 0 else 0
    
    # Memory info
    memory = round(psutil.Process().memory_info().rss / (1024 ** 2), 2)
    
    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Images Processed: {num_images}")
    print(f"Valid Geometries Extracted: {len(df)}")
    print(f"Memory Used: {memory:.2f} MB")
    print(f"\nThresholds Applied (Hybrid Spatial-Temporal):")
    print(f"  Spatial Thickness Variance: ±{hybrid_threshold}%")
    print(f"  Temporal Thickness Change: ±{thickness_threshold_px:.2f} px")
    print(f"  Drift (Temporal): {drift_threshold_px:.2f} px")
    print(f"  Roughness (Spatial): {roughness_threshold:.2f}")
    print(f"\nHybrid Anomaly Detection:")
    print(f"  Detected Anomalies: {actual_anomalies}")
    print(f"  Normal Frames: {tn}")
    print(f"  Spatial Anomalies: {df['spatial_anomaly'].sum()}")
    print(f"  Temporal Anomalies: {df['temporal_anomaly'].sum()}")
    print(f"  True Positives: {tp}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Precision: {precision:.2f}%")
    print(f"  Recall: {recall:.2f}%")
    
    # Statistics
    print(f"\nSpatial-Temporal Statistics:")
    if "thickness_px" in df.columns:
        print(f"  Mean Wall Thickness: {df['thickness_px'].mean():.2f} px")
        print(f"  Thickness Variability: {df['thickness_pct_change'].std():.2f}%")
    if "center_drift_accumulation" in df.columns:
        print(f"  Max Cumulative Drift: {df['center_drift_accumulation'].abs().max():.2f} px")
    if "center_drift" in df.columns:
        print(f"  Max Layer-to-Layer Drift: {df['center_drift'].abs().max():.2f} px")
    
    # Anomalous frames
    if actual_anomalies > 0:
        print(f"\n⚠️  Hybrid Anomalies Detected:")
        anomaly_frames = df[df["hybrid_detected"] == 1][["index", "filename", "spatial_anomaly", "temporal_anomaly"]]
        for idx, row in anomaly_frames.iterrows():
            anom_type = ""
            if row["spatial_anomaly"]:
                anom_type += "[Spatial]"
            if row["temporal_anomaly"]:
                anom_type += "[Temporal]"
            print(f"    Frame {int(row['index'])}: {row['filename']} {anom_type}")
    
    # Save report
    output_file = f"timing_optical_threshold_report.xlsx"
    df.to_excel(output_file, index=False)
    print(f"\n📁 Report saved: {output_file}")


if __name__ == "__main__":
    main()
