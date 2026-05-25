# Imports
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from river.drift import ADWIN
import os

# Input log
LOG_FILE = "real_dataset/Prod1Torno.csv"

# Configurations
OUTPUT_DIR = "resultados_drift_real"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def replace_machine_times_above_limit(series, limit=60, window=2):
    cleaned = series.reset_index(drop=True).astype(float).copy()
    original = cleaned.copy()

    for i, value in original.items():
        if value <= limit:
            continue

        neighbor_values = []
        for j in range(max(0, i - window), i):
            if original[j] <= limit:
                neighbor_values.append(original[j])

        for j in range(i + 1, min(len(original), i + window + 1)):
            if original[j] <= limit:
                neighbor_values.append(original[j])

        if neighbor_values:
            cleaned.at[i] = sum(neighbor_values) / len(neighbor_values)

    return cleaned

def calculate_moving_average(series):
    rolling_window = min(15, max(3, len(series) // 20))
    moving_average = series.rolling(
        window=rolling_window,
        min_periods=1,
        center=True
    ).mean()

    return moving_average, rolling_window

def detect_drifts_with_adwin(series):
    adwin = ADWIN()
    drifts = []

    for i, value in enumerate(series):
        adwin.update(value)
        if adwin.drift_detected:
            drifts.append(i)

    return drifts

def format_axis(ax):
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=12))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=8))
    ax.tick_params(axis="both", labelsize=10)

    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.18)
    ax.minorticks_on()

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#555555")
    ax.spines["bottom"].set_color("#555555")
    ax.margins(x=0.01)

def plot_series_with_drifts(ax, series, drifts, title, series_label, color):
    ax.plot(
        series.index,
        series,
        color=color,
        linewidth=1.55,
        alpha=0.9,
        label=series_label
    )

    for i, drift in enumerate(drifts):
        ax.axvline(
            x=drift,
            color="#d62728",
            linestyle="--",
            linewidth=1.5,
            alpha=0.82,
            label="Drift detectado" if i == 0 else None
        )
        ax.scatter(
            drift,
            series.iloc[drift],
            s=42,
            color="#d62728",
            edgecolor="white",
            linewidth=0.8,
            zorder=5
        )

    ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=10)
    ax.set_ylabel("Tempo de sojourn (s)", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=9)
    format_axis(ax)

def plot_graph(LOG_FILE, activity_name):
    print(f"\nProcessing: {LOG_FILE}")
    base_name = os.path.basename(LOG_FILE)

    # 1. Read and preprocess log
    df = pd.read_csv(LOG_FILE, sep=";", parse_dates=["Inicio", "Fim"], dayfirst=True)

    # 2. Trace index
    df.sort_values(by=["Case", "Inicio"], inplace=True)

    # 3. Filter by activity
    df = df[df["Atividade"] == activity_name]

    series = (
        df[df["Atividade"] == activity_name]
        .sort_values("Inicio")["Tempo(s)"]
    )
    series = replace_machine_times_above_limit(series)
    
    # 4. Drift detection with ADWIN
    moving_average, rolling_window = calculate_moving_average(series)
    raw_drifts = detect_drifts_with_adwin(series)
    moving_average_drifts = detect_drifts_with_adwin(moving_average)

    # 5. Plot
    plt.style.use("seaborn-v0_8-whitegrid")

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15, 9),
        sharex=True,
        constrained_layout=True
    )

    fig.suptitle(
        f"{base_name} - {activity_name}",
        fontsize=16,
        fontweight="bold",
        y=1.02
    )

    plot_series_with_drifts(
        axes[0],
        series,
        raw_drifts,
        "ADWIN aplicado ao tempo original tratado",
        "Tempo por trace",
        "#4c78a8"
    )
    plot_series_with_drifts(
        axes[1],
        moving_average,
        moving_average_drifts,
        f"ADWIN aplicado a media movel ({rolling_window} traces)",
        f"Media movel ({rolling_window} traces)",
        "#f58518"
    )

    if len(series) > 0:
        y_min = max(0, min(series.min(), moving_average.min()) * 0.95)
        y_max = max(series.max(), moving_average.max()) * 1.08
        if y_max > y_min:
            for ax in axes:
                ax.set_ylim(y_min, y_max)

    axes[1].set_xlabel("Trace", fontsize=12, fontweight="bold")

    output_path = f"{OUTPUT_DIR}/{base_name}_{activity_name}_comparacao_adwin.png"
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return raw_drifts, moving_average_drifts

if __name__ == "__main__":
    activity_name = "Maquina trabalhando"
    # Main execution
    detected_drifts, moving_average_drifts = plot_graph(LOG_FILE, activity_name)
    
    print(f"\nSummary for {os.path.basename(LOG_FILE)}")
    print(f"Detected drifts without moving average: {detected_drifts}")
    print(f"Detected drifts with moving average: {moving_average_drifts}")
