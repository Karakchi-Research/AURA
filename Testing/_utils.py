"""
Shared utilities for image-based AES anomaly detection experiments.
Handles image loading, optical signal extraction, and feature engineering.
"""

import os
import random
import time
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from Crypto.Cipher import AES

BLOCK_SIZE = 16
KEY = b'ThisIsA16ByteKey'


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


def extract_optical_intensity(image):
    """
    Extract mean pixel intensity from an image as optical signal.
    
    Args:
        image: Grayscale image array
    
    Returns:
        Float: mean pixel intensity (0-255 range)
    """
    return float(np.mean(image))


def simulate_aes_block_with_anomaly(block, index, inject_anomaly):
    """
    Simulate AES encryption with optional timing anomaly (delay or fault).
    
    Args:
        block: 16-byte plaintext block
        index: Block index for tracking
        inject_anomaly: Boolean, whether to inject anomaly
    
    Returns:
        Dict with timing and encryption results
    """
    anomaly_type = None
    modified_block = list(block)
    
    start_time = time.perf_counter()
    
    if inject_anomaly:
        anomaly_type = random.choice(["delay", "fault"])
        if anomaly_type == "delay":
            time.sleep(random.uniform(0.005, 0.02))
        elif anomaly_type == "fault":
            modified_block[0] ^= 0xFF
    
    # Ensure block is exactly 16 bytes
    if len(modified_block) < BLOCK_SIZE:
        modified_block += [0] * (BLOCK_SIZE - len(modified_block))
    else:
        modified_block = modified_block[:BLOCK_SIZE]
    
    cipher = AES.new(KEY, AES.MODE_ECB)
    ciphertext = cipher.encrypt(bytes(modified_block))
    
    end_time = time.perf_counter()
    
    return {
        "index": index,
        "timing": end_time - start_time,
        "anomaly_type": anomaly_type,
        "is_malicious": 1 if inject_anomaly else 0,
        "ciphertext": list(ciphertext)
    }


def generate_synthetic_optical_signal(image_index, num_images, inject_anomaly=False):
    """
    Generate synthetic optical signal (mean intensity + noise + optional anomaly).
    
    Args:
        image_index: Index of the image in sequence
        num_images: Total number of images
        inject_anomaly: Boolean, whether to inject optical anomaly
    
    Returns:
        Float: optical signal value
    """
    base_optical = 128.0  # Mid-range intensity
    
    # Data-dependent component (simulated)
    data_component = (image_index % 10) * 2.0
    
    # Noise
    noise = random.uniform(-2, 2)
    
    optical_signal = base_optical + data_component + noise
    
    if inject_anomaly:
        # Simulate optical flash/glitch from fault injection
        optical_signal += random.uniform(40, 80)
    
    return float(np.clip(optical_signal, 0, 255))


def extract_timing_features(results):
    """
    Extract temporal features from timing measurements for ML.
    
    Args:
        results: List of result dicts with 'timing' key
    
    Returns:
        DataFrame with engineered features
    """
    data = []
    times = [r["timing"] for r in results]
    
    for i, r in enumerate(results):
        prev_time = times[i - 1] if i > 0 else times[i]
        next_time = times[i + 1] if i < len(times) - 1 else times[i]
        
        delta_prev = r["timing"] - prev_time
        delta_next = next_time - r["timing"]
        
        window_start = max(0, i - 2)
        window = times[window_start:i + 1]
        rolling_mean = sum(window) / len(window)
        rolling_std = float(pd.Series(window).std()) if len(window) > 1 else 0.0
        
        row = {
            "index": r["index"],
            "timing": r["timing"],
            "delta_prev": delta_prev,
            "delta_next": delta_next,
            "rolling_mean": rolling_mean,
            "rolling_std": rolling_std,
            "is_malicious": r["is_malicious"]
        }
        data.append(row)
    
    return pd.DataFrame(data)


def extract_optical_features(results):
    """
    Extract spatial features from optical measurements for ML.
    
    Args:
        results: List of result dicts with 'optical' key
    
    Returns:
        DataFrame with engineered features
    """
    data = []
    opticals = [r["optical"] for r in results]
    
    for i, r in enumerate(results):
        prev_optical = opticals[i - 1] if i > 0 else opticals[i]
        next_optical = opticals[i + 1] if i < len(opticals) - 1 else opticals[i]
        
        delta_prev = r["optical"] - prev_optical
        delta_next = next_optical - r["optical"]
        
        window_start = max(0, i - 2)
        window = opticals[window_start:i + 1]
        rolling_mean = sum(window) / len(window)
        rolling_std = float(pd.Series(window).std()) if len(window) > 1 else 0.0
        
        row = {
            "index": r["index"],
            "optical": r["optical"],
            "delta_prev": delta_prev,
            "delta_next": delta_next,
            "rolling_mean": rolling_mean,
            "rolling_std": rolling_std,
            "is_malicious": r["is_malicious"]
        }
        data.append(row)
    
    return pd.DataFrame(data)


def extract_fused_features(results):
    """
    Extract combined timing + optical features for ML.
    
    Args:
        results: List of result dicts with 'timing' and 'optical' keys
    
    Returns:
        DataFrame with engineered features from both modalities
    """
    data = []
    timings = [r["timing"] for r in results]
    opticals = [r["optical"] for r in results]
    
    for i, r in enumerate(results):
        # Timing features
        prev_timing = timings[i - 1] if i > 0 else timings[i]
        next_timing = timings[i + 1] if i < len(timings) - 1 else timings[i]
        delta_timing = r["timing"] - prev_timing
        
        # Optical features
        prev_optical = opticals[i - 1] if i > 0 else opticals[i]
        next_optical = opticals[i + 1] if i < len(opticals) - 1 else opticals[i]
        delta_optical = r["optical"] - prev_optical
        
        window_start = max(0, i - 2)
        timing_window = timings[window_start:i + 1]
        optical_window = opticals[window_start:i + 1]
        
        row = {
            "index": r["index"],
            "timing": r["timing"],
            "optical": r["optical"],
            "delta_timing": delta_timing,
            "delta_optical": delta_optical,
            "timing_rolling_mean": sum(timing_window) / len(timing_window),
            "optical_rolling_mean": sum(optical_window) / len(optical_window),
            "timing_rolling_std": float(pd.Series(timing_window).std()) if len(timing_window) > 1 else 0.0,
            "optical_rolling_std": float(pd.Series(optical_window).std()) if len(optical_window) > 1 else 0.0,
            "is_malicious": r["is_malicious"]
        }
        data.append(row)
    
    return pd.DataFrame(data)


def compute_threshold(values):
    """
    Compute adaptive threshold as mean + 3-sigma.
    
    Args:
        values: List of numeric values
    
    Returns:
        Float: threshold value
    """
    if not values:
        return 0.0
    
    mean = sum(values) / len(values)
    sigma = (max(values) - min(values)) / len(values)
    
    return mean + 3 * sigma
