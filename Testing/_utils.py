"""
Shared utilities for image-based AES anomaly detection experiments.
Handles image loading, optical signal extraction, and feature engineering.
"""

import os
import numpy as np
import pandas as pd
import cv2

def load_images_from_directory(image_dir):
    """
    Load all image files from a directory in sorted order.
    
    Args:
        image_dir: Path to directory containing image files (.jpg, .png, etc.)
    
    Returns:
        List of (filename, image_array) tuples in grayscale
    """
    if not os.path.isdir(image_dir):
        raise ValueError(f"Image directory not found: {image_dir}")
    
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = sorted([
        f for f in os.listdir(image_dir) 
        if f.lower().endswith(supported_formats)
    ])
    
    if not image_files:
        raise ValueError(f"No image files found in {image_dir}")
    
    images = []
    for filename in image_files:
        filepath = os.path.join(image_dir, filename)
        img = cv2.imread(filepath)
        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append((filename, gray))
    
    return images


# ========== LPBF GEOMETRIC FEATURE EXTRACTION ==========
def detect_wall_boundaries(image, roi_height=None):
    """
    Detect left and right wall boundaries using gradient-based edge detection.
    Implements methodology from SMASIS2026: edge detection on intensity profile.
    
    Args:
        image: Grayscale image array
        roi_height: Optional ROI height (if None, uses full image height)
    
    Returns:
        Dict with 'left_edge', 'right_edge', 'thickness_px', 'center_x'
    """
    if roi_height is None:
        roi_height = image.shape[0]
    
    # Apply Gaussian blur to reduce noise (as in SMASIS2026)
    blurred = cv2.GaussianBlur(image[:roi_height, :], (5, 5), 0)
    
    # Average intensities along height to get 1D horizontal profile
    intensity_profile = np.mean(blurred, axis=0)
    
    # Compute gradient (derivative) to find edges
    gradient = np.gradient(intensity_profile)
    
    # Find strongest positive gradient (left edge) and negative gradient (right edge)
    left_edge = None
    right_edge = None
    
    if len(gradient) > 1:
        # Search for left edge (positive gradient)
        pos_indices = np.where(gradient > np.percentile(gradient, 75))[0]
        if len(pos_indices) > 0:
            left_edge = int(pos_indices[0])
        
        # Search for right edge (negative gradient)
        neg_indices = np.where(gradient < np.percentile(gradient, 25))[0]
        if len(neg_indices) > 0:
            right_edge = int(neg_indices[-1])
    
    # Calculate derived metrics
    thickness_px = None
    center_x = None
    
    if left_edge is not None and right_edge is not None and right_edge > left_edge:
        thickness_px = right_edge - left_edge
        center_x = (left_edge + right_edge) / 2.0
    
    return {
        "left_edge": left_edge,
        "right_edge": right_edge,
        "thickness_px": thickness_px,
        "center_x": center_x,
        "intensity_profile": intensity_profile,
        "gradient": gradient
    }


def detect_wall_instability(image, roi_height=None):
    """
    Detect geometric instability indicators from image texture/roughness.
    
    Args:
        image: Grayscale image array
        roi_height: Optional ROI height
    
    Returns:
        Dict with instability metrics
    """
    if roi_height is None:
        roi_height = image.shape[0]
    
    roi = image[:roi_height, :]
    
    # Laplacian for edge roughness detection
    laplacian = cv2.Laplacian(roi, cv2.CV_64F)
    
    # Contour irregularity: high variance in gradient magnitude
    sobel_x = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    return {
        "roughness": float(np.std(laplacian)),
        "texture_variance": float(np.var(magnitude)),
        "contour_irregularity": float(np.std(np.mean(magnitude, axis=0)))
    }


def extract_geometric_features(results):
    """
    Extract geometric features from wall boundary detection results.
    
    Args:
        results: List of result dicts with geometric measurements
    
    Returns:
        DataFrame with engineered geometric features
    """
    data = []
    thicknesses = [r["thickness_px"] for r in results if r["thickness_px"] is not None]
    centers = [r["center_x"] for r in results if r["center_x"] is not None]
    
    for i, r in enumerate(results):
        # Skip if critical measurements missing
        if r["thickness_px"] is None or r["center_x"] is None:
            continue
        
        # Thickness features (relative to baseline)
        baseline_thickness = thicknesses[0] if thicknesses else 1.0
        thickness_deviation = r["thickness_px"] - baseline_thickness
        thickness_pct_change = (thickness_deviation / baseline_thickness) * 100 if baseline_thickness > 0 else 0.0
        
        # Center drift features
        prev_center = centers[i - 1] if i > 0 and len(centers) > i - 1 else centers[0]
        next_center = centers[i + 1] if i < len(centers) - 1 else centers[i]
        center_drift = r["center_x"] - prev_center
        center_drift_accumulation = r["center_x"] - centers[0]
        
        # Rolling statistics
        window_start = max(0, i - 2)
        thickness_window = thicknesses[window_start:i + 1] if window_start < len(thicknesses) else [r["thickness_px"]]
        thickness_rolling_mean = sum(thickness_window) / len(thickness_window)
        thickness_rolling_std = float(pd.Series(thickness_window).std()) if len(thickness_window) > 1 else 0.0
        
        row = {
            "index": r["index"],
            "filename": r.get("filename", ""),
            "thickness_px": r["thickness_px"],
            "thickness_deviation": thickness_deviation,
            "thickness_pct_change": thickness_pct_change,
            "center_x": r["center_x"],
            "center_drift": center_drift,
            "center_drift_accumulation": center_drift_accumulation,
            "thickness_rolling_mean": thickness_rolling_mean,
            "thickness_rolling_std": thickness_rolling_std,
            "roughness": r.get("roughness", 0.0),
            "texture_variance": r.get("texture_variance", 0.0),
            "contour_irregularity": r.get("contour_irregularity", 0.0),
            "is_anomaly": r.get("is_anomaly", 0)
        }
        data.append(row)
    
    return pd.DataFrame(data)


def compute_geometric_threshold(df, feature="thickness_pct_change"):
    """
    Compute threshold for geometric anomaly detection.
    Uses mean + 2*sigma approach for instability detection.
    
    Args:
        df: DataFrame with geometric features
        feature: Feature column to threshold on
    
    Returns:
        Float: threshold value
    """
    if feature not in df.columns or len(df) == 0:
        return 0.0
    
    values = df[feature].dropna().values
    if len(values) == 0:
        return 0.0
    
    mean = float(np.mean(values))
    std = float(np.std(values))
    
    return mean + 2 * std  # 2-sigma for higher sensitivity


def detect_geometric_anomalies(df, thickness_threshold=5.0, drift_threshold=0.5, roughness_threshold=None):
    """
    Detect geometric anomalies using multi-feature criteria.
    
    Args:
        df: DataFrame with geometric features
        thickness_threshold: Max acceptable thickness variance (%)
        drift_threshold: Max acceptable cumulative drift (pixels)
        roughness_threshold: Max acceptable roughness (if None, compute from data)
    
    Returns:
        DataFrame with anomaly flags added
    """
    df = df.copy()
    
    if roughness_threshold is None:
        roughness_values = df["roughness"].dropna().values
        if len(roughness_values) > 0:
            roughness_threshold = np.mean(roughness_values) + 2 * np.std(roughness_values)
        else:
            roughness_threshold = 1000.0
    
    # Multi-criteria anomaly detection
    thickness_anomaly = (df["thickness_pct_change"].abs() > thickness_threshold)
    drift_anomaly = (df["center_drift_accumulation"].abs() > drift_threshold)
    roughness_anomaly = (df["roughness"] > roughness_threshold)
    texture_anomaly = (df["texture_variance"] > df["texture_variance"].quantile(0.9))
    
    # Combined: flag if multiple criteria exceeded
    df["anomaly_detected"] = (thickness_anomaly | drift_anomaly | roughness_anomaly | texture_anomaly).astype(int)
    
    return df
