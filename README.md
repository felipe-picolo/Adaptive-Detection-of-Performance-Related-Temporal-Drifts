# ADWIN Drift Detection for Process Mining Datasets

This repository contains three executable Python scripts for drift detection using ADWIN:
- offline analysis of synthetic XES logs,
- streaming-style analysis of XES logs,
- analysis of a real CSV production dataset.

## Project Files
- `adwin_dataset.py`: offline processing of XES logs, graph generation, and quantitative metrics.
- `adwin_streaming.py`: event-by-event streaming simulation with optional live plotting.
- `adwin_real_dataset.py`: drift detection on a real CSV dataset.
- `dataset_manufacturing/`: synthetic XES logs.
- `real_dataset/`: real CSV dataset (`Prod1Torno.csv`).
- `resultados_drift/`: output PNG graphs for XES-based scripts.
- `resultados_drift_real/`: output PNG graphs for CSV-based script.

## Requirements
- Python 3.8+
- Dependencies in `requirements.txt`:
  - `pm4py`
  - `matplotlib`
  - `river`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Scripts and Behavior

### 1) `adwin_dataset.py` (offline XES + metrics)
Purpose:
- Reads each log from `LOG_FILES`.
- Converts lifecycle events to interval format (`@@duration`) using `pm4py`.
- Filters one activity (default: `Machine_Operating`).
- Builds a sojourn-time series by trace index.
- Detects drifts with ADWIN.
- Saves a static PNG with detected drifts (red) and expected real drifts (green).
- Prints quality metrics per file.

Current default input logs:
- `dataset_manufacturing/ST_20.xes`
- `dataset_manufacturing/DR_20.xes`
- `dataset_manufacturing/DR_MS_20.xes`
- `dataset_manufacturing/DR_MS_ST_20.xes`

Real drift rules implemented in code:
- `DR_XX`: one drift at `10 + 11 * (sample - 1)`.
- `DR_MS_XX`: drifts at `[0, 100, 200, 300, 400]` with +1 shift per sample (except the zero point).
- `DR_MS_ST_XX`: base drifts `[20, 60, 100, 140, 180]` with per-drift increments `[1, 3, 5, 7, 9]` per sample.

Printed metrics:
- `precision`
- `mean_delay_traces`
- `mean_delay_seconds`
- `false_positives`
- `true_positives`
- `false_negatives`

Run:

```bash
python adwin_dataset.py
```

### 2) `adwin_streaming.py` (stream simulation on XES)
Purpose:
- Reads each XES log from `LOG_FILES`.
- Filters one activity (default: `Machine_Operating`).
- Reconstructs sojourn times from `start` and `complete` transitions.
- Streams durations one by one (optional delay).
- Runs ADWIN online and marks detected drifts.

Main config variables:
- `LOG_FILES`
- `activity_name`
- `live_plot` (`True` to show interactive chart, `False` for console-only detection)
- `stream_delay` (seconds between streamed events)

Notes:
- Requires `lifecycle:transition` values compatible with `start` and `complete`.
- Creates `resultados_drift/` folder, but this script does not save a file by default.

Run:

```bash
python adwin_streaming.py
```

### 3) `adwin_real_dataset.py` (real CSV dataset)
Purpose:
- Reads `real_dataset/Prod1Torno.csv` using `;` separator.
- Parses `Inicio` and `Fim` as dates.
- Filters selected activity (default: `Maquina trabalhando`).
- Uses `Tempo(s)` as sojourn-time series.
- Detects drifts with ADWIN and saves a static PNG.

Expected CSV columns used by the script:
- `Case`
- `Atividade`
- `Inicio`
- `Fim`
- `Tempo(s)`

Output:
- PNG saved in `resultados_drift_real/`.
- Drift indices printed in console.

Run:

```bash
python adwin_real_dataset.py
```

## Typical Execution Order
1. `python adwin_dataset.py` to evaluate synthetic logs with metrics.
2. `python adwin_streaming.py` to visualize online detection behavior.
3. `python adwin_real_dataset.py` to evaluate the real dataset.