# Imports
import pandas as pd
import matplotlib.pyplot as plt
from river.drift import ADWIN
import os

# Input log
LOG_FILE = "real_dataset/Prod1Torno.csv"

# Configurations
OUTPUT_DIR = "resultados_drift_real"
os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    
    # 4. Drift detection with ADWIN
    adwin = ADWIN()
    drifts = []

    for i, v in enumerate(series):
        adwin.update(v)
        if adwin.drift_detected:
            drifts.append(i)

    # 5. Plot
    plt.figure(figsize=(8, 8))
    plt.plot(series, linewidth=1.2, label="Sojourn Time")

    for drift in drifts:
        plt.axvline(
            x=drift,
            color='red',
            linestyle='--',
            label='Drift detected' if drift == drifts[0] else ""
        )
    
    plt.title(f"{base_name} - {activity_name}")
    plt.xlabel("Trace")
    plt.ylabel("Sojourn Time (seconds)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f"{OUTPUT_DIR}/{base_name}_{activity_name}.png", dpi=200)
    plt.close()

    return drifts

if __name__ == "__main__":
    activity_name = "Maquina trabalhando"
    # Main execution
    detected_drifts = plot_graph(LOG_FILE, activity_name)
    
    print(f"\nSummary for {os.path.basename(LOG_FILE)}")
    print(f"Detected drifts: {detected_drifts}")