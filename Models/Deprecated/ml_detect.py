import pandas as pd
import random
import time
import multiprocessing
import psutil
from Crypto.Cipher import AES
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

BLOCK_SIZE = 16
KEY = b'ThisIsA16ByteKey'


def aes_encrypt_block_with_anomaly(args):
    block, index, inject_anomaly = args
    start_time = time.time()

    anomaly_type = None
    modified_block = block.copy()

    if inject_anomaly:
        anomaly_type = random.choice(["delay", "fault"])
        if anomaly_type == "delay":
            time.sleep(random.uniform(0.005, 0.02))
        elif anomaly_type == "fault":
            modified_block[0] ^= 0xFF

    if len(modified_block) < BLOCK_SIZE:
        modified_block += [0] * (BLOCK_SIZE - len(modified_block))
    elif len(modified_block) > BLOCK_SIZE:
        modified_block = modified_block[:BLOCK_SIZE]

    byte_data = bytes(modified_block)
    cipher = AES.new(KEY, AES.MODE_ECB)
    ciphertext = cipher.encrypt(byte_data)
    end_time = time.time()

    return {
        "index": index,
        "original_block": block,
        "encrypted_block": list(ciphertext),
        "anomaly_type": anomaly_type,
        "time": end_time - start_time
    }


def generate_blocks(num_blocks, anomaly_ratio=0.2):
    blocks = []
    for i in range(num_blocks):
        block = [random.randint(0, 255) for _ in range(BLOCK_SIZE)]
        inject_anomaly = random.random() < anomaly_ratio
        blocks.append((block, i, inject_anomaly))
    return blocks


def extract_features(results):
    data = []
    times = [r["time"] for r in results]

    for i, r in enumerate(results):
        prev_time = times[i - 1] if i > 0 else times[i]
        next_time = times[i + 1] if i < len(times) - 1 else times[i]
        delta_prev = r["time"] - prev_time
        delta_next = next_time - r["time"]

        window_start = max(0, i - 2)
        window = times[window_start:i + 1]
        rolling_mean = sum(window) / len(window)
        rolling_std = pd.Series(window).std()
        if pd.isna(rolling_std):
            rolling_std = 0.0

        row = {
            "index": r["index"],
            "time": r["time"],
            "prev_time": prev_time,
            "delta_prev": delta_prev,
            "delta_next": delta_next,
            "rolling_mean": rolling_mean,
            "rolling_std": float(rolling_std),
            "actual_label": 1 if r.get("anomaly_type") else 0
        }

        if "original_block" in r:
            block = r["original_block"]
            for j in range(min(len(block), 16)):
                row[f"b{j}"] = block[j]

        data.append(row)

    return pd.DataFrame(data)


def ml_detect(results):
    if not results:
        return results

    df = extract_features(results)

    X = df.drop(columns=["index", "actual_label"])
    y = df["actual_label"].copy()

    # If no true anomaly labels exist (like thermal video input),
    # create pseudo-labels from timing behavior so ML mode still works.
    if y.nunique() < 2:
        pseudo_threshold = df["time"].mean() + df["time"].std()
        y = (df["time"] > pseudo_threshold).astype(int)

    if y.nunique() < 2:
        df["ml_detected"] = 0
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        df["ml_detected"] = model.predict(X)

    for i, r in enumerate(results):
        r["ml_detected"] = int(df["ml_detected"].iloc[i])

    return results


def main():
    num_blocks = int(input("Enter number of plaintext blocks to encrypt: "))
    malicious_percent = float(input("Enter percentage of malicious blocks to inject (0-100): "))
    anomaly_ratio = malicious_percent / 100.0
    num_cores = int(input("Enter number of CPU cores to use: "))

    print(f"\nEncrypting {num_blocks} blocks with {malicious_percent}% malicious blocks using {num_cores} cores...\n")

    start_time = time.time()
    blocks = generate_blocks(num_blocks, anomaly_ratio)

    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(aes_encrypt_block_with_anomaly, blocks)

    total_time = time.time() - start_time
    avg_latency = total_time / num_blocks
    throughput = num_blocks / total_time
    memory_used = round(psutil.Process().memory_info().rss / (1024 ** 2), 2)

    print("\n🕒 Performance Metrics")
    print(f"Total time: {total_time:.4f} sec")
    print(f"Average latency per block: {avg_latency:.6f} sec")
    print(f"Throughput: {throughput:.2f} blocks/sec")
    print(f"Memory used (RSS): {memory_used:.2f} MB")

    df = extract_features(results)
    X = df.drop(columns=["index", "actual_label"])
    y = df["actual_label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("\n📊 Machine Learning Detection Report")
    print(classification_report(y_test, y_pred, digits=2))

    df["predicted_label"] = model.predict(X)

    tp = ((df["actual_label"] == 1) & (df["predicted_label"] == 1)).sum()
    fp = ((df["actual_label"] == 0) & (df["predicted_label"] == 1)).sum()
    fn = ((df["actual_label"] == 1) & (df["predicted_label"] == 0)).sum()

    total = len(df)
    malicious_actual = df["actual_label"].sum()
    malicious_detected = df["predicted_label"].sum()
    accuracy = tp / malicious_actual * 100 if malicious_actual else 0

    threshold = df["time"].mean() + 3 * (df["time"].max() - df["time"].min()) / len(df)

    print("\n🔒 Anomaly Detection Summary")
    print(f"Total Blocks: {total}")
    print(f"Injected Malicious Blocks: {malicious_actual}")
    print(f"Detected Malicious Blocks: {malicious_detected}")
    print(f"Correct Detections (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"Detection Accuracy: {accuracy:.2f}%")
    print(f"Detection Threshold (sec): {threshold:.6f}")

    df.to_excel("ml_aes_anomaly_report.xlsx", index=False)
    print("\n📁 Report saved to 'ml_aes_anomaly_report.xlsx'")


if __name__ == "__main__":
    main()