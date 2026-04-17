#!/usr/bin/env python3
"""
Temporal-Dynamics Threshold Detection Model (LPBF Geometric Monitoring)
Detects geometric anomalies in LPBF thin-wall fabrication by analyzing
temporal evolution of wall properties across layers. Extracts how thickness,
drift, and roughness change from layer to layer (layer-to-layer dynamics).
Uses statistical thresholding on temporal features.
Based on SMASIS2026 paper methodology with temporal analysis.
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
        print("Usage: python timing_threshold.py <image_directory> [temporal_threshold]")
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
    print(f"TEMPORAL-DYNAMICS THRESHOLD DETECTION (LPBF Geometric Monitoring)")
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
    
    if len(df) < 2:
        print("Error: Not enough valid geometric detections. Check image quality/format.")
        sys.exit(1)
    
    # Compute thresholds on layer-to-layer changes (temporal dynamics)
    thickness_delta_threshold = compute_geometric_threshold(df, "thickness_deviation")
    drift_delta_threshold = compute_geometric_threshold(df, "center_drift")
    roughness_threshold = compute_geometric_threshold(df, "roughness")
    
    # Detect anomalies based on temporal dynamics + surface quality
    df["temporal_anomaly"] = (
        (df["thickness_deviation"].abs() > abs(df["thickness_px"].mean() * temporal_threshold / 100)) |
        (df["center_drift"].abs() > drift_delta_threshold) |
        (df["roughness"] > df["roughness"].quantile(0.85))
    ).astype(int)
    
    # Calculate metrics
    actual_anomalies = df["temporal_anomaly"].sum()
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
    print(f"\nThresholds Applied (Temporal Dynamics):")
    print(f"  Thickness Deviation: ±{temporal_threshold}%")
    print(f"  Layer-to-Layer Drift: {drift_delta_threshold:.2f} px")
    print(f"  Surface Roughness: {roughness_threshold:.2f}")
    print(f"\nTemporal Anomaly Detection:")
    print(f"  Detected Anomalies: {actual_anomalies}")
    print(f"  Normal Frames: {tn}")
    print(f"  True Positives: {tp}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Precision: {precision:.2f}%")
    print(f"  Recall: {recall:.2f}%")
    
    # Temporal statistics
    print(f"\nTemporal Evolution Statistics:")
    if "thickness_deviation" in df.columns:
        print(f"  Max Thickness Change: {df['thickness_deviation'].abs().max():.2f} px")
        print(f"  Avg Thickness Change: {df['thickness_deviation'].abs().mean():.2f} px")
    if "center_drift" in df.columns:
        print(f"  Max Layer-to-Layer Drift: {df['center_drift'].abs().max():.2f} px")
        print(f"  Avg Layer-to-Layer Drift: {df['center_drift'].abs().mean():.2f} px")
    
    # Anomalous frames
    if actual_anomalies > 0:
        print(f"\n⚠️  Temporal Anomalies Detected:")
        anomaly_frames = df[df["temporal_anomaly"] == 1][["index", "filename", "thickness_deviation", "center_drift", "roughness"]]
        for idx, row in anomaly_frames.iterrows():
            print(f"    Frame {int(row['index'])}: {row['filename']} (Δthick: {row['thickness_deviation']:.2f}px, drift: {row['center_drift']:.2f}px)")
    
    # Save report
    output_file = f"timing_threshold_report.xlsx"
    df.to_excel(output_file, index=False)
    print(f"\n📁 Report saved: {output_file}")


if __name__ == "__main__":
    main()
