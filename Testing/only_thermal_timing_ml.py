import os
import time
import random
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

BLOCK_SIZE = 4096


def process_block_thermal_timing(args):
    chunk, idx, is_malicious = args

    start = time.time()

    # simulate thermal signal
    thermal_sig = sum(chunk) % 255

    # timing simulation
    time.sleep(random.uniform(0.005, 0.02))

    latency = time.time() - start

    return {
        "latency": latency,
        "thermal_sig": thermal_sig,
        "is_malicious": int(is_malicious)
    }


def main():

    if len(sys.argv) < 2:
        print("Usage: python3 only_thermal_timing_ml.py <video_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)

    results = []

    print(f"Analyzing {os.path.basename(file_path)} (THERMAL + TIMING ML)...")

    with open(file_path, "rb") as f:

        idx = 0
        MAX_BLOCKS = 20000

        while True:

            chunk = f.read(BLOCK_SIZE)

            if not chunk or idx >= MAX_BLOCKS:
                break

            if len(chunk) < BLOCK_SIZE:
                chunk += b'\0' * (BLOCK_SIZE - len(chunk))

            results.append(
                process_block_thermal_timing(
                    (chunk, idx, random.random() < 0.1)
                )
            )

            idx += 1

            if idx % 5000 == 0:
                print(f"Processed {idx} blocks...", end="\r")

    df = pd.DataFrame(results)

    # feature matrix
    features = ["latency", "thermal_sig"]

    X = df[features]
    y = df["is_malicious"]

    split_idx = int(len(df) * 0.3)

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]

    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    print(f"\nTraining Thermal-Timing Random Forest on {len(X_train)} samples...")

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    avg_latency = df["latency"].mean()

    throughput = file_size_mb / (df["latency"].sum() if df["latency"].sum() > 0 else 1)

    print("\n" + "=" * 55)
    print(f"THERMAL + TIMING ML REPORT: {os.path.basename(file_path)}")
    print("=" * 55)

    print(f"VIDEO SIZE:      {file_size_mb:.2f} MB")
    print(f"AVG LATENCY:     {avg_latency:.6f} sec/block")
    print(f"THROUGHPUT:      {throughput:.2f} MB/s")

    print("-" * 55)

    print(f"TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"PRECISION: {precision:.4f}")
    print(f"RECALL:    {recall:.4f}")
    print(f"F1-SCORE:  {f1:.4f}")

    print("=" * 55)

    importances = rf.feature_importances_

    print("\nFEATURE IMPORTANCE:")
    for feat, imp in zip(features, importances):
        print(f"{feat}: {imp:.4f}")


if __name__ == "__main__":
    main()
