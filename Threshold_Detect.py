import pandas as pd
import random
import time
import multiprocessing
import psutil
from Crypto.Cipher import AES

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


def detect_anomalies(results):
    if not results:
        return results, 0.0

    times = [r["time"] for r in results]
    threshold = sum(times) / len(times) + 3 * (max(times) - min(times)) / len(times)

    for r in results:
        r["threshold_detected"] = r["time"] > threshold

    return results, threshold


def main():
    num_blocks = int(input("Enter number of plaintext blocks to encrypt: "))
    malicious_percent = float(input("Enter percentage of malicious blocks to inject (0-100): "))
    anomaly_ratio = malicious_percent / 100.0
    num_cores = int(input("Enter number of CPU cores to use: "))

    print(f"\nEncrypting {num_blocks} blocks with {malicious_percent}% malicious blocks using {num_cores} cores...\n")

    blocks = generate_blocks(num_blocks, anomaly_ratio)

    with multiprocessing.Pool(processes=num_cores) as pool:
        results = pool.map(aes_encrypt_block_with_anomaly, blocks)

    results, threshold = detect_anomalies(results)

    malicious_actual = sum(1 for r in results if r["anomaly_type"] is not None)
    malicious_detected = sum(1 for r in results if r["threshold_detected"])
    correct_detections = sum(1 for r in results if r["anomaly_type"] and r["threshold_detected"])
    false_positives = sum(1 for r in results if r["anomaly_type"] is None and r["threshold_detected"])
    false_negatives = sum(1 for r in results if r["anomaly_type"] and not r["threshold_detected"])

    accuracy = correct_detections / malicious_actual * 100 if malicious_actual > 0 else 0

    print("\n🔒 Anomaly Detection Summary")
    print(f"Total Blocks: {num_blocks}")
    print(f"Injected Malicious Blocks: {malicious_actual}")
    print(f"Detected Malicious Blocks: {malicious_detected}")
    print(f"Correct Detections: {correct_detections}")
    print(f"False Positives: {false_positives}")
    print(f"False Negatives: {false_negatives}")
    print(f"Detection Accuracy: {accuracy:.2f}%")
    print(f"Detection Threshold (sec): {threshold:.6f}")

    print("\n📄 Sample Benign Block:")
    for r in results:
        if r["anomaly_type"] is None:
            print(f"Index: {r['index']}\nPlain: {r['original_block']}\nCipher: {r['encrypted_block']}\nTime: {r['time']:.6f}s")
            break

    print("\n📄 Sample Malicious Block:")
    for r in results:
        if r["anomaly_type"] is not None:
            print(f"Index: {r['index']}\nPlain: {r['original_block']}\nCipher: {r['encrypted_block']}\nTime: {r['time']:.6f}s\nType: {r['anomaly_type']}")
            break

    df = pd.DataFrame(results)
    df.to_excel("aes_anomaly_report.xlsx", index=False)
    print("\n📝 Report saved to 'aes_anomaly_report.xlsx'")


if __name__ == "__main__":
    main()