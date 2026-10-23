# summary_helper.py
# Helper module for generating and saving summary reports for different models.

# Import necessary libraries
import os
import csv

# Global constants
SUMMARY_FILE = "summary.csv"
DEFAULT_COLUMNS = [
    "model",
    "file",
    "file_size_mb",
    "avg_latency_sec",
    "end_to_end_latency_sec",
    "throughput_mb_s",
    "memory_usage_mb",
    "tp",
    "fp",
    "fn",
    "tn",
    "precision",
    "recall",
    "f1",
]


# _apprend_to_master()
# Appends a row to the master summary file.
#
# @param out_dir - Directory where the master summary file is located.
# @param row - Dictionary containing the row data to append.
def _append_to_master(out_dir, row):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    master_path = os.path.join(out_dir, SUMMARY_FILE)
    write_header = not os.path.exists(master_path)
    with open(master_path, "a") as f:
        writer = csv.DictWriter(f, fieldnames=DEFAULT_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# save_summary()
# Saves a model-specific summary file and appends a row to the master summary.
#
# @param out_dir - Directory where the summary files are saved.
# @param base - Base name of the file being processed.
# @param model_name - Name of the model (e.g., "Thermal Timing Threshold").
# @param summary - Dictionary containing the summary data.
def save_summary(out_dir, base, model_name, summary):
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # Write model-specific summary file
    model_summary_path = os.path.join(
        out_dir, "{}_{}_summary.csv".format(model_name, base)
    )
    with open(model_summary_path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    # Prepare master row
    master_row = {
        "model": model_name,
        "file": summary.get("file", base),
        "file_size_mb": summary.get("file_size_mb", ""),
        "avg_latency_sec": summary.get("avg_latency_sec", ""),
        "end_to_end_latency_sec": summary.get("end_to_end_latency_sec", ""),
        "throughput_mb_s": summary.get("throughput_mb_s", ""),
        "memory_usage_mb": summary.get("memory_usage_mb", ""),
        "tp": int(summary.get("tp", 0)),
        "fp": int(summary.get("fp", 0)),
        "fn": int(summary.get("fn", 0)),
        "tn": int(summary.get("tn", 0)),
        "precision": summary.get("precision", ""),
        "recall": summary.get("recall", ""),
        "f1": summary.get("f1", ""),
    }

    _append_to_master(out_dir, master_row)


# print_report()
# Prints a consistent console report for all models.
#
# @param model_name - Name of the model (e.g., "Thermal Timing Threshold").
# @param base - Base name of the file being processed.
# @param summary - Dictionary containing the summary data.
def print_report(model_name, base, summary):
    print("\n" + "=" * 55)
    print("{} REPORT: {}".format(model_name, base))
    print("VIDEO SIZE:    {:.2f} MB".format(float(summary.get("file_size_mb", 0))))
    print(
        "AVG LATENCY:   {:.6f} sec/block".format(
            float(summary.get("avg_latency_sec", 0))
        )
    )
    print(
        "END-TO-END:    {:.4f} sec".format(
            float(summary.get("end_to_end_latency_sec", 0))
        )
    )
    print("THROUGHPUT:    {:.2f} MB/s".format(float(summary.get("throughput_mb_s", 0))))
    print("MEMORY USAGE:  {:.2f} MB".format(float(summary.get("memory_usage_mb", 0))))
    print("-" * 55)
    print(
        "TP: {} | FP: {} | FN: {} | TN: {}".format(
            int(summary.get("tp", 0)),
            int(summary.get("fp", 0)),
            int(summary.get("fn", 0)),
            int(summary.get("tn", 0)),
        )
    )
    print("PRECISION: {:.4f}".format(float(summary.get("precision", 0))))
    print("RECALL:    {:.4f}".format(float(summary.get("recall", 0))))
    print("F1-SCORE:  {:.4f}".format(float(summary.get("f1", 0))))
    print("=" * 55)
