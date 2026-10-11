"""
Raw AUC summary statistics (mean, SD, range) per model, across seeds/configs
=============================================================================
For the Paper 3 abstract: DAE and TrAE are currently reported only as a
raw-AUC range (e.g., "0.13-0.19"), while SVDD is reported as mean+-SD
(0.474+-0.102). To report all three consistently, this computes mean, SD,
min, and max raw AUC for every model across all 20 seed/configuration
combinations.

WHAT YOU NEED TO EDIT
----------------------
I don't have your Paper 3 result files or their schema, so this script
makes a guess at the shape of your data and needs to be pointed at your
actual results. Two common shapes are supported out of the box -- edit
INPUT_MODE and the paths/column names below to match what you actually
have:

MODE "csv": one CSV with a row per (model, seed, config) and a raw AUC
    column, e.g.:
        model,seed,config,raw_auc
        DAE,42,single,0.15
        DAE,42,multi,0.17
        ...
    Set CSV_PATH, and the column names if they differ from the defaults.

MODE "json": one JSON file per model, each a list of per-seed/config
    records with a raw AUC field, e.g.:
        [{"seed": 42, "config": "single", "raw_auc": 0.15}, ...]
    Set JSON_PATHS to map each model name to its file.

If your results live somewhere else entirely (a pickle, a set of per-seed
folders, a notebook's in-memory arrays, etc.), tell me the actual layout
and I'll rewrite the loader -- the stats computation below (mean/SD/min/max
via numpy) will stay the same either way.

Run from wherever your result files live:
    python raw_auc_summary_stats.py
"""

import json
import os

import numpy as np

# ---------------------------------------------------------------------------
# EDIT THIS SECTION to match your actual results
# ---------------------------------------------------------------------------

INPUT_MODE = "csv"  # "csv" or "json"

# --- mode "csv" ---
CSV_PATH = "raw_auc_results.csv"
CSV_MODEL_COL = "model"
CSV_VALUE_COL = "raw_auc"

# --- mode "json" ---
JSON_PATHS = {
    "DAE": "dae_raw_auc.json",
    "TrAE": "trae_raw_auc.json",
    "SVDD": "svdd_raw_auc.json",
}
JSON_VALUE_KEY = "raw_auc"

# ---------------------------------------------------------------------------


def load_csv():
    import csv as csv_module

    values_by_model = {}
    with open(CSV_PATH, newline="") as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            model = row[CSV_MODEL_COL]
            values_by_model.setdefault(model, []).append(float(row[CSV_VALUE_COL]))
    return values_by_model


def load_json():
    values_by_model = {}
    for model, path in JSON_PATHS.items():
        if not os.path.exists(path):
            print(f"  WARNING: {path} not found for {model} -- skipped")
            continue
        with open(path) as f:
            recs = json.load(f)
        values_by_model[model] = [float(r[JSON_VALUE_KEY]) for r in recs]
    return values_by_model


def summarize(values_by_model):
    print(f"{'=' * 70}\nRaw AUC summary statistics, per model\n{'=' * 70}")
    results = {}
    for model, values in values_by_model.items():
        arr = np.array(values)
        n = len(arr)
        mean = float(arr.mean())
        sd = float(arr.std(ddof=1)) if n > 1 else float("nan")
        vmin, vmax = float(arr.min()), float(arr.max())
        print(f"  {model:6s}  n={n:3d}  mean={mean:.3f}  SD={sd:.3f}  "
              f"range=[{vmin:.2f}, {vmax:.2f}]")
        results[model] = {"n": n, "mean": mean, "sd": sd, "min": vmin, "max": vmax}

    print(f"\n{'=' * 70}\nAbstract-ready phrasing\n{'=' * 70}")
    for model, r in results.items():
        print(f"  {model}: mean raw AUC {r['mean']:.2f}±{r['sd']:.2f}, "
              f"range {r['min']:.2f}–{r['max']:.2f}")

    with open("raw_auc_summary_stats.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved raw_auc_summary_stats.json")
    return results


def main():
    if INPUT_MODE == "csv":
        values_by_model = load_csv()
    elif INPUT_MODE == "json":
        values_by_model = load_json()
    else:
        raise ValueError(f"Unknown INPUT_MODE: {INPUT_MODE!r}")

    if not values_by_model:
        print("No data loaded -- check INPUT_MODE and paths above.")
        return
    summarize(values_by_model)


if __name__ == "__main__":
    main()
