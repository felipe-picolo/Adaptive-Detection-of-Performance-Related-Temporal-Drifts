# Imports
import pm4py
import matplotlib.pyplot as plt
import time
from river.drift import ADWIN
import os
import random

# Input logs
LOG_FILES = [
    #"dataset_manufacturing/ST_09.xes",
    "dataset_manufacturing/DR_18.xes",
    "dataset_manufacturing/DR_MS_13.xes",
    "dataset_manufacturing/DR_MS_ST_28.xes"
]

# Configurations
OUTPUT_DIR = "resultados_drift"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Data treatment and streaming
def stream(xes_file, activity_name, delay_seconds=0.05, real_delay=False, random_variation=0):
    log = pm4py.read_xes(xes_file)
    log = pm4py.convert_to_event_log(log)
    df = pm4py.convert_to_dataframe(log)

    # Sort by timestamp
    df.sort_values(by="time:timestamp", inplace=True)

    # Filter by activity
    df = df[df["concept:name"] == activity_name]

    # Generate sojourn times
    starts = {}
    for _, row in df.iterrows():
        transition = str(row["lifecycle:transition"]).lower().strip()
        case_id = row["case:concept:name"]
        key = case_id

        if transition == "start":
            starts.setdefault(key, []).append(row["time:timestamp"])
        elif transition == "complete":
            if key in starts and starts[key]:
                start_ts = starts[key].pop(0)
                duration = (row["time:timestamp"] - start_ts).total_seconds()
                yield duration

        if real_delay:
            if duration > 0:
                time.sleep(duration)
        else:
            if delay_seconds > 0:
                time.sleep(delay_seconds)

# Live plotting with drift detection
def run_live_plot(stream_iter, title):
    plt.ion()
    fig, ax = plt.subplots(figsize=(16, 9))
    line, = ax.plot([], [], linewidth=1.2, label="Sojourn Time")
    ax.set_title(title)
    ax.set_xlabel("Evento")
    ax.set_ylabel("Sojourn Time (seconds)")
    ax.grid(True)
    ax.legend()

    xs, ys = [], []
    adwin = ADWIN()
    drifts = []

    for i, v in enumerate(stream_iter):
        xs.append(i)
        ys.append(v)
        adwin.update(v)
        if adwin.drift_detected:
            drifts.append(i)
            ax.axvline(
                x=i,
                color='red',
                linestyle='--',
                label='Drift Detected' 
            )
            # Update legend to show only one label for drift
            if len(drifts) == 1: 
                ax.legend()

        line.set_data(xs, ys)
        ax.relim()
        ax.autoscale_view()
        plt.pause(0.001)

    plt.ioff()
    plt.show()
    return drifts


activity_name = "Machine_Operating"
live_plot = True # Turn on live plot
real_delay = True # Use the actual duration between events as delay in streaming (only if live_plot is True)
add_random_soujurn_time = True # Adds random delay to simulate variability in sojourn times
stream_delay = 0.02  # Delay between events in seconds, oly used if real_delay is False and live_plot is True

# Main execution
for log in LOG_FILES:
    base_name = os.path.basename(log)
    if add_random_soujurn_time:
        random_variation = 0.01  # Max random variation in seconds
        stream_delay += random.uniform(0, random_variation)
    if live_plot:
        drifts = run_live_plot(
            stream(log, activity_name, delay_seconds=stream_delay),
            title=f"{base_name} - {activity_name}"
        )
        print("Drifts detected (stream):", drifts)
    else:
        adwin = ADWIN()
        for i, v in enumerate(stream(log, activity_name, delay_seconds=stream_delay)):
            adwin.update(v)
            if adwin.drift_detected:
                print("Drift detected at:", i)
