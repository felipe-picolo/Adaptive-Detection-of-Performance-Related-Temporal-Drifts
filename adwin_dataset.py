# Imports
import pm4py
import pandas as pd
import matplotlib.pyplot as plt
from river.drift import ADWIN
import os
from pm4py.objects.log.util import interval_lifecycle

# Input logs
LOG_FILES = [
    "dataset_manufacturing/ST_20.xes",
    "dataset_manufacturing/DR_20.xes",
    "dataset_manufacturing/DR_MS_20.xes",
    "dataset_manufacturing/DR_MS_ST_20.xes"
]

# Configurations
OUTPUT_DIR = "resultados_drift"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Real drift calculator based on filename and sample number
def calculate_real_drifts(xes_file):
    """
    Rules for real drifts based on filename patterns:
    - DR_01: drift 10, +11 per file (DR_02, DR_03, ...)
    - DR_MS_01: 0, 100, 200, 300, 400; except 0, +1 per file
    - DR_MS_ST_01: 20, 60, 100, 140, 180; increments +1, +3, +5, +7, +9 per file
    """
    base = os.path.basename(xes_file).replace(".xes", "")
    parts = base.split("_")

    # Sample number expected as last token (e.g., DR_01 -> 01)
    try:
        sample_n = int(parts[-1])
    except ValueError:
        sample_n = 1

    if base.startswith("DR_MS_ST"):
        base_drifts = [20, 60, 100, 140, 180]
        increments = [1, 3, 5, 7, 9]
        return [
            d + (sample_n - 1) * inc
            for d, inc in zip(base_drifts, increments)
        ]

    if base.startswith("DR_MS"):
        base_drifts = [0, 100, 200, 300, 400]
        return [0] + [
            d + (sample_n - 1)
            for d in base_drifts[1:]
        ]

    if base.startswith("DR"):
        return [10 + 11 * (sample_n - 1)]

    return []

# Quantitative detection metrics
def detection_metrics(real_drifts, detected_drifts, trace_times):
    # Convert to sorted lists of integers
    real = sorted(int(x) for x in real_drifts)
    detected = sorted(int(x) for x in detected_drifts)

    matched_detected = set()
    matched_pairs = []

    for r in real:
        # Match the first detected drift at or after the real drift
        candidates = [
            d for d in detected
            if d not in matched_detected and d >= r
        ]
        if candidates:
            d_best = min(candidates)
            matched_detected.add(d_best)
            matched_pairs.append((r, d_best))

    true_positives = len(matched_pairs)
    false_positives = len(detected) - true_positives
    false_negatives = len(real) - true_positives

    precision = true_positives / len(detected) if detected else 0.0

    # Delay is counted as detected - real for matched pairs
    if matched_pairs:
        delays_traces = [d - r for r, d in matched_pairs]
        mean_delay_traces = sum(delays_traces) / len(delays_traces)

        delays_seconds = []
        for r, d in matched_pairs:
            if 0 <= r < len(trace_times) and 0 <= d < len(trace_times):
                delta = trace_times[d] - trace_times[r]
                delays_seconds.append(delta.total_seconds())
        mean_delay_seconds = (
            sum(delays_seconds) / len(delays_seconds)
            if delays_seconds else 0.0
        )
    else:
        mean_delay_traces = 0.0
        mean_delay_seconds = 0.0

    return {
        "precision": precision,
        "mean_delay_traces": mean_delay_traces,
        "mean_delay_seconds": mean_delay_seconds,
        "false_positives": false_positives,
        "true_positives": true_positives,
        "false_negatives": false_negatives
    }

def plot_graph(xes_file, activity_name, real_drifts):
    print(f"\nProcessing: {xes_file}")
    base_name = os.path.basename(xes_file)

    # 1. Read and preprocess log
    log = pm4py.read_xes(xes_file)
    log = pm4py.convert_to_event_log(log)
    log = interval_lifecycle.to_interval(log)
    df = pm4py.convert_to_dataframe(log)

    # 2. Create trace index
    first_events = df.groupby("case:concept:name").first()
    first_events.sort_values(by="time:timestamp", inplace=True)
    first_events["trace_index"] = range(first_events.shape[0])
    trace_times = first_events["time:timestamp"].tolist()

    df = pd.merge(
        df,
        first_events[["trace_index"]],
        how="inner",
        on=["case:concept:name"]
    )

    df.sort_values(by=["trace_index", "time:timestamp"], inplace=True)

    # 3. Filter by activity
    df = df[df["concept:name"] == activity_name]

    if "@@duration" not in df.columns:
        raise ValueError("Column @@duration not found in the log.")

    series = (
        df.groupby("trace_index", sort=False)["@@duration"]
        .last()
        .reset_index(drop=True)
    )

    # 4. Drift detection with ADWIN
    adwin = ADWIN()
    drifts = []

    for i, v in enumerate(series):
        adwin.update(v)
        if adwin.drift_detected:
            drifts.append(i)

    # 5. Plot
    plt.figure(figsize=(5, 5))
    plt.plot(series, linewidth=1.2, label="Sojourn Time")

    for drift in drifts:
        plt.axvline(
            x=drift,
            color='red',
            linestyle='--',
            label='Drift detected' if drift == drifts[0] else ""
        )
    
    for drift in real_drifts:
        plt.axvline(
            x=drift,
            color='green',
            linestyle='--',
            label='Real drift' if drift == real_drifts[0] else ""
        )
    
    plt.title(f"{base_name} - {activity_name}")
    plt.xlabel("Trace")
    plt.ylabel("Sojourn Time (seconds)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f"{OUTPUT_DIR}/{base_name}_{activity_name}.png", dpi=200)
    plt.close()

    return drifts, trace_times

if __name__ == "__main__":
    activity_name = "Machine_Operating"
    # Main execution
    for log in LOG_FILES:
        real_drifts = calculate_real_drifts(log)
        detected_drifts, trace_times = plot_graph(log, activity_name, real_drifts)
        metrics = detection_metrics(real_drifts, detected_drifts, trace_times)

        print(f"\nSummary for {os.path.basename(log)}")
        print(f"Real drifts: {real_drifts}")
        print(f"Detected drifts: {detected_drifts}")
        print(f"Precision: {metrics['precision']:.3f}")
        print(f"Mean delay (traces): {metrics['mean_delay_traces']:.3f}")
        print(f"Mean delay (seconds): {metrics['mean_delay_seconds']:.3f}")
        print(f"False positives: {metrics['false_positives']}")