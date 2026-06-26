# Imports
from glob import glob
from river.drift import ADWIN
from pm4py.objects.log.util import interval_lifecycle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

import random
import pm4py
import pandas as pd
import matplotlib.pyplot as plt
import os

# Input logs
LOG_FILES = [
    "dataset_manufacturing/DR_MS_04.xes",
    "dataset_manufacturing/DR_MS_ST_25.xes"
]

WINDOW_SIZE = 20
# Configurations
OUTPUT_DIR = "prediction"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def log_to_dataframe(log_file):
    print(f"\nProcessing: {log_file}")

    # 1. Read and preprocess log
    log = pm4py.read_xes(log_file)
    log = pm4py.convert_to_event_log(log)
    log = interval_lifecycle.to_interval(log)
    df = pm4py.convert_to_dataframe(log)

    # 2. Create trace index
    first_events = df.groupby("case:concept:name").first()
    first_events.sort_values(by="time:timestamp", inplace=True)
    first_events["Trace"] = range(first_events.shape[0])

    df = pd.merge(
        df,
        first_events[["Trace"]],
        how="inner",
        on=["case:concept:name"]
    )

    df.sort_values(by=["Trace", "time:timestamp"], inplace=True)
    return df

def calculate_real_drifts(xes_file):
    """
    Rules for real drifts based on filename patterns:
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

    return []

def graph_test(log_file,output_dir):
    # Convert to DataFrame
    df = log_to_dataframe(log_file)
    activities = df["Activity"].unique()

    for activity in activities:
        activity_df = df[df["Activity"] == activity]
        plt.figure(figsize=(10, 6))
        plt.plot(
            activity_df["Trace"],
            activity_df["Duration"],
            marker=".",
            markersize=4,
            linewidth=1
        )
        plt.title(f"Activity: {activity} in {os.path.basename(log_file)}")
        plt.xlabel("Trace")
        plt.ylabel("Duration")
        plt.legend()
        plt.grid()
        output_path = os.path.join(output_dir, f"{os.path.basename(log_file).replace('.xes', '')}_{activity}.png")
        plt.savefig(output_path)
        plt.close()

def matrix_test(log_file, output_dir):
    # Convert to DataFrame
    df = log_to_dataframe(log_file)
    
    continuous_activities = ["Machine_Operating","Raw_Material_Loading","Short_Downtime"]
    binary_activities = ["Equipment_Failure","Maintenance"]

    cases = sorted(df["Trace"].unique())

    matrix = pd.DataFrame(index=cases)

    for activity in continuous_activities:
        activity_df = df[df["Activity"] == activity]
        matrix[activity] = activity_df.groupby("Trace")["Duration"].sum().reindex(cases)
    
    for activity in binary_activities:
        activity_df = df[df["Activity"] == activity]
        matrix[activity] = activity_df.groupby("Trace").size().reindex(cases, fill_value=0).gt(0).astype(int)
    
    matrix = matrix.reset_index(names="Trace")

    # print(matrix.head())
    # print(matrix.shape)

    fig, ax = plt.subplots(figsize=(18, 8))
    x = matrix["Trace"]

    ax.plot(x, matrix["Machine_Operating"], label="Machine_Operating", linewidth=1.5)
    ax.plot(x, matrix["Raw_Material_Loading"], label="Raw_Material_Loading", linewidth=1.5)
    ax.plot(x, matrix["Short_Downtime"], label="Short_Downtime", linewidth=1.5)

    markers = ["o", "s"]

    positions = {"Equipment_Failure": -5, "Maintenance": -10, "Drift": -15}

    for i, marker in zip(binary_activities, markers):
        oc = matrix[matrix[i] == 1]

        ax.scatter(oc["Trace"], [positions[i]] * len(oc), label=i, marker=marker, s=80)
    
    ax.set_title(f"Activity Matrix for {os.path.basename(log_file)}")
    ax.set_xlabel("Trace")
    ax.set_ylabel("Duration / Binary Events")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{os.path.basename(log_file).replace('.xes', '')}_matrix.png"))
    
    return matrix

def adwin(log_file, output_dir, matrix):
    # Convert to DataFrame
    df = log_to_dataframe(log_file)
    real_drifts = calculate_real_drifts(log_file)
    activity_name = "Machine_Operating"

    # Filter by activity
    df = df[df["Activity"] == activity_name]

    series = (
        df.groupby("Trace", sort=False)["Duration"]
        .last()
        .reset_index(drop=True)
    )

    adwin = ADWIN()
    drifts = []

    for i, v in enumerate(series):
        adwin.update(v)

        if adwin.drift_detected:
            drifts.append(i)
    
    # Add drift information to the matrix
    matrix["DriftD"] = 0
    matrix.loc[matrix.index.isin(drifts), "DriftD"] = 1

    matrix["DriftR"] = 0
    matrix.loc[matrix.index.isin(real_drifts), "DriftR"] = 1

    #Plott matrix with drift information
    binary_activities = ["Equipment_Failure","Maintenance"]

    fig, ax = plt.subplots(figsize=(18, 8))
    x = matrix["Trace"]
    ax.plot(x, matrix["Machine_Operating"], label="Machine_Operating", linewidth=1.5)
    ax.plot(x, matrix["Raw_Material_Loading"], label="Raw_Material_Loading", linewidth=1.5)
    ax.plot(x, matrix["Short_Downtime"], label="Short_Downtime", linewidth=1.5)
    
    # Detected drifts
    for i, drift in enumerate(drifts):
        ax.axvline(
            x=drift,
            color="#d62728",
            linestyle="--",
            linewidth=1.8,
            alpha=0.9,
            label="Detected drift" if i == 0 else None
        )

    # Real drifts
    for i, drift in enumerate(real_drifts):
        ax.axvline(
            x=drift,
            color="#2ca02c",
            linestyle="--",
            linewidth=1.8,
            alpha=0.9,
            label="Real drift" if i == 0 else None
        )

    markers = ["o", "s"]

    positions = {"Equipment_Failure": -5, "Maintenance": -10}

    for i, marker in zip(binary_activities, markers):
        oc = matrix[matrix[i] == 1]

        ax.scatter(oc["Trace"], [positions[i]] * len(oc), label=i, marker=marker, s=80)
    
    ax.set_title(f"Activity Matrix for {os.path.basename(log_file)}")
    ax.set_xlabel("Trace")
    ax.set_ylabel("Duration / Binary Events")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{os.path.basename(log_file).replace('.xes', '')}_matrix.png"))

    output_log_path = OUTPUT_DIR+"/matrix_log"
    os.makedirs(output_log_path, exist_ok=True)
    matrix.to_csv(os.path.join(output_log_path, f"{os.path.basename(log_file).replace('.xes', '')}_matrix.csv"), index=False, sep=";")
    
def load_dataset(file_list):
    dfs = []
    for file in file_list:
        df = pd.read_csv(file, sep=";")
        for col in [
            "Machine_Operating",
            "Raw_Material_Loading",
            "Short_Downtime"
        ]:
            df[f"{col}_MA5"] = (
                df[col]
                .rolling(5, min_periods=1)
                .mean()
            )
            df[f"{col}_MA10"] = (
                df[col]
                .rolling(10, min_periods=1)
                .mean()
            )
            df[f"{col}_STD5"] = (
                df[col]
                .rolling(5, min_periods=1)
                .std()
                .fillna(0)
            )
            df[f"{col}_STD10"] = (
                df[col]
                .rolling(10, min_periods=1)
                .std()
                .fillna(0)
            )
            df[f"{col}_MAX5"] = (
                df[col]
                .rolling(5, min_periods=1)
                .max()
            )
            df[f"{col}_MIN5"] = (
                df[col]
                .rolling(5, min_periods=1)
                .min()
            )
            df[f"{col}_RANGE5"] = (
                df[f"{col}_MAX5"]
                -
                df[f"{col}_MIN5"]
            )
            df[f"{col}_DIFF1"] = (
                df[col]
                .diff()
                .fillna(0)
            )
            df[f"{col}_DIFF5"] = (
                df[col]
                .diff(5)
                .fillna(0)
            )

        df["Target"] = 0
        failure = df[df["Equipment_Failure"] == 1]
        if len(failure):
            failure_trace = failure.iloc[0]["Trace"]
            df.loc[
                (
                    (failure_trace - df["Trace"] > 0)
                    &
                    (failure_trace - df["Trace"] <= WINDOW_SIZE)
                ),
                "Target"
            ] = 1
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)

def random_forest():
    dr_ms = sorted(glob("prediction/matrix_log/DR_MS_[0-9][0-9]_matrix.csv"))
    dr_ms_st = sorted(glob("prediction/matrix_log/DR_MS_ST_[0-9][0-9]_matrix.csv"))

    dr_train, dr_temp = train_test_split(
        dr_ms,
        test_size=0.30,
    )
    dr_val, dr_test = train_test_split(
        dr_temp,
        test_size=1/3,
    )
    st_train, st_temp = train_test_split(
        dr_ms_st,
        test_size=0.30,
    )
    st_val, st_test = train_test_split(
        st_temp,
        test_size=1/3,
        random_state=42,
        shuffle=True
    )

    train_files = dr_train + st_train
    val_files = dr_val + st_val
    test_files = dr_test + st_test

    train = load_dataset(train_files)
    val = load_dataset(val_files)
    test = load_dataset(test_files)

    features = [
        "Machine_Operating",
        "Machine_Operating_MA5",
        "Machine_Operating_MA10",
        "Machine_Operating_STD5",
        "Machine_Operating_STD10",
        "Machine_Operating_MAX5",
        "Machine_Operating_MIN5",
        "Machine_Operating_RANGE5",
        "Machine_Operating_DIFF1",
        "Machine_Operating_DIFF5",

        "Raw_Material_Loading",
        "Raw_Material_Loading_MA5",
        "Raw_Material_Loading_MA10",
        "Raw_Material_Loading_STD5",
        "Raw_Material_Loading_STD10",
        "Raw_Material_Loading_MAX5",
        "Raw_Material_Loading_MIN5",
        "Raw_Material_Loading_RANGE5",
        "Raw_Material_Loading_DIFF1",
        "Raw_Material_Loading_DIFF5",

        "Short_Downtime",
        "Short_Downtime_MA5",
        "Short_Downtime_MA10",
        "Short_Downtime_STD5",
        "Short_Downtime_STD10",
        "Short_Downtime_MAX5",
        "Short_Downtime_MIN5",
        "Short_Downtime_RANGE5",
        "Short_Downtime_DIFF1",
        "Short_Downtime_DIFF5",

        "DriftD",
        "DriftR"
    ]

    X_train = train[features]
    y_train = train["Target"]

    X_val = val[features]
    y_val = val["Target"]

    X_test = test[features]
    y_test = test["Target"]

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("VALIDAÇÃO")
    pred = model.predict(X_val)
    print(classification_report(y_val, pred))
    print(confusion_matrix(y_val, pred))
    print()
    print("TESTE")

    pred = model.predict(X_test)

    print(classification_report(y_test, pred))
    print(confusion_matrix(y_test, pred))
    print()

    importance = pd.DataFrame({
        "Feature": features,
        "Importance": model.feature_importances_
    })
    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("IMPORTÂNCIA DAS VARIÁVEIS")
    print(importance)



if __name__ == "__main__":
    task = "0"
    while task in ["0","1", "2", "3", "4", "5", "6"]:
        task = input("select task: 1 - graph test, 2 - matrix test, 3 - ADWIN, 4 - Create log files, 5 - Prevision test, 6 - RUL, 9 - Exit: ")
        if task == "1":
            output = OUTPUT_DIR+"/graph_test"
            os.makedirs(output, exist_ok=True)
            for log_file in LOG_FILES:
                # Graph the event log
                graph_test(log_file, output)
        
        if task == "2":
            output = OUTPUT_DIR+"/matrix_test"
            os.makedirs(output, exist_ok=True)
            for log_file in LOG_FILES: 
                # Create matrix representation
                matrix_test(log_file, output)
        
        if task == "3":
            output = OUTPUT_DIR+"/adwin_test"
            os.makedirs(output, exist_ok=True)
            for log_file in LOG_FILES:
                # Create matrix representation
                matrix = matrix_test(log_file, output)
                # Apply ADWIN drift detection
                adwin(log_file, output, matrix)
        
        if task == "4":
            #Select all .xes files in the dataset_manufacturing folder and create a log file with the matrix representation for each of them
            all_files = sorted(
                glob("dataset_manufacturing/DR_MS_*.xes") +
                glob("dataset_manufacturing/DR_MS_ST_*.xes")
            )

            output = OUTPUT_DIR+"/adwin_test"
            os.makedirs(output, exist_ok=True)

            for log_file in all_files:
                # Create matrix representation
                matrix = matrix_test(log_file, output)
                # Apply ADWIN drift detection
                adwin(log_file, output, matrix)
        
        if task == "5":
            random_forest()
