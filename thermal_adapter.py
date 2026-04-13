import sys
import os
import cv2
import numpy as np
import pandas as pd

from Threshold_Detect import detect_anomalies
from ML_Detect import ml_detect


def video_to_results(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    results = []
    index = 0
    prev_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            diff = cv2.absdiff(gray, prev_frame)
            change_intensity = float(np.mean(diff))

            results.append({
                "index": index,
                "time": change_intensity,
                "anomaly_type": None,
                "original_block": [0] * 16
            })

            index += 1

        prev_frame = gray

    cap.release()
    return results


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage:")
        print("python3 thermal_adapter.py <video_file>")
        sys.exit(1)

    video_path = sys.argv[1]

    if not os.path.exists(video_path):
        print(f"Error: File not found -> {video_path}")
        sys.exit(1)

    mode = input("Select detection mode (threshold / ml / hybrid): ").strip().lower()

    if mode not in ["threshold", "ml", "hybrid"]:
        print("Invalid mode selected.")
        sys.exit(1)

    print(f"\nProcessing: {video_path}")
    print(f"Mode selected: {mode}")

    results = video_to_results(video_path)

    if mode == "threshold":
        results, threshold = detect_anomalies(results)
        anomalies = sum(1 for r in results if r["threshold_detected"])

        print("Threshold:", threshold)
        print("Detected anomalies:", anomalies)

    elif mode == "ml":
        results = ml_detect(results)
        anomalies = sum(1 for r in results if r["ml_detected"])

        print("ML detected anomalies:", anomalies)

    elif mode == "hybrid":
        results, threshold = detect_anomalies(results)
        results = ml_detect(results)

        for r in results:
            r["hybrid_detected"] = bool(r["threshold_detected"] or r["ml_detected"])

        anomalies = sum(1 for r in results if r["hybrid_detected"])

        print("Threshold:", threshold)
        print("Hybrid detected anomalies:", anomalies)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_file = f"{base_name}_thermal_report.xlsx"

    df = pd.DataFrame(results)
    df.to_excel(output_file, index=False)

    print(f"Saved report: {output_file}")