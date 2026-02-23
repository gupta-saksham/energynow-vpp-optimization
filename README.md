# Energy Now – VPP battery optimization

This project helps you **see how profitable batteries can be** when used in a Virtual Power Plant (VPP): partly for your own consumption (behind-the-meter, BTM) and partly for the grid frequency market (FCR, front-of-meter). It is used for the **Energy Now competition** and for general profitability assessment.

---

## What you need to run it

- **Python:** 3.11, 3.12, or 3.13 (Gurobi does not support 3.14).
- **Gurobi:** You need a working Gurobi license (e.g. [academic](https://www.gurobi.com/academia/academic-program-and-licenses/)). Install and activate it (e.g. with the command Gurobi gives you after login).
- **Packages:** Install from the project folder:
  ```bash
  python -m venv .venv
  source .venv/bin/activate   # or: .venv\Scripts\activate on Windows
  pip install -r requirements.txt
  ```

---

## Quick start

1. Put your data files in the **data/** folder (see [Input data](#input-data) below).
2. Run the main optimization:
   ```bash
   python model_multi_battery.py
   ```
3. Open the generated HTML dashboard in your browser (e.g. `outputs/multi_battery_dashboard.html`).

---

## What each script does

**Entry-point scripts (project root)** – run these for optimization, scenario analysis, or dashboard generation:

| Script | What it does |
|--------|----------------|
| **model_multi_battery.py** | **The optimization model.** Many sites, each with a battery split between BTM (own load) and FTM (FCR). Run this for all VPP optimizations. |
| **scenario_analysis.py** | Runs many combinations of “how much BTM vs FTM” and load size, then you can compare results in dashboards. |
| **generate_dashboard.py** | Builds HTML dashboards from **saved** results so you don’t have to re-run the optimizer. Use `--list` to see saved runs, `--file path/to/file.pkl` for a specific run, `--scenarios` for scenario comparison. |
| **view_results.py** | Lists and builds dashboards from `results/dashboard_data_*.pkl.gz` (btm20/50/100). Options: `--list`, `--scenario btm50`, `--all`, `--master`. |

**Support code (lib/)** – used by the scripts above; you only run two of them for data prep:

- **lib/paths.py** – Project directories (`data/`, `results/`, `outputs/`).
- **lib/results_io.py** – Save/load optimization and scenario results (`.pkl`).
- **lib/dashboard_multi_battery.py** – Plotly dashboard builders (portfolio, site, comparison report).
- **lib/data_utils.py** – Data loading helpers (e.g. for the notebook).
- **lib/extract_frequ.py** – Fetches grid frequency data and writes FCR CSVs under `data/`. Run when you need real FCR activation: `python -m lib.extract_frequ`.
- **lib/fix_missing_frequencies.py** – Fills gaps in the FCR activation CSV. Run after extract: `python -m lib.fix_missing_frequencies`.

---

## Input data

Place these files in the **data/** folder:

| File | Description |
|------|-------------|
| **data/Day ahead.csv** | Day-ahead electricity prices (e.g. EUR/MWh, 15-min or hourly; code aligns as needed). |
| **data/Load profile.csv** | Load profiles per site/column (e.g. `LG 18`, `Site_01`), 15-minute. |
| **data/FCR prices 2024.csv** | FCR capacity prices (e.g. EUR/MW per 4 h block). |
| **data/industrial_tarrifs.csv** | Industrial tariff used in the multi-battery model. |
| **data/FCR_Energy_2024_15min.csv** or **data/FCR_Energy_2024_15min_full.csv** | 15-minute FCR activation (up/down). To generate: `python -m lib.extract_frequ` then `python -m lib.fix_missing_frequencies` if you use real activation. |

If you don’t have FCR activation data, the code can use a synthetic profile instead.

---

## Outputs

All generated files go into **results/** (serialized runs) or **outputs/** (dashboards and reports). Nothing is written to the project root.

- **results/** – Saved optimization runs (e.g. `.pkl`, `.pkl.gz`, `summary_*.json`). Used by `generate_dashboard.py` and `view_results.py`.
- **outputs/** – HTML dashboards and CSV reports, e.g. `multi_battery_dashboard.html`, `dashboard_Site_01.html`, `scenario_comparison_dashboard.html`, `master_dashboard.html`, `multi_battery_results.csv`, `site_comparison_report.csv`, `scenario_results.csv`, `comparison_btm20.csv`, and `scenario_outputs/*.html`. Open the HTML files in your browser.

---

## Results: what comes from which run

It’s easy to mix up files. Here’s how they relate to each run.

### 1. Single optimization run (`model_multi_battery.py`)

You set **START_DATE** and **END_DATE** at the bottom of `model_multi_battery.py` (e.g. 2024-01-01 to 2024-01-07), then run it once.

| Output | Meaning |
|--------|--------|
| **results/optimization_YYYY-MM-DD_to_YYYY-MM-DD_*.pkl** | Saved run for that date range. Filename includes the range and a timestamp. |
| **outputs/multi_battery_results.csv** | Time series and metrics for that run (overwritten each time). |
| **outputs/multi_battery_dashboard.html** | Portfolio dashboard for that run. |
| **outputs/dashboard_Site_01.html** (etc.) | Per-site dashboard for that run. |
| **outputs/site_comparison_report.csv** | Site comparison for that run. |

**Regenerate dashboards from a saved run:**  
`python generate_dashboard.py` (uses latest `.pkl` in `results/`) or `python generate_dashboard.py --file results/optimization_....pkl`.

---

### 2. Scenario analysis run (`scenario_analysis.py`)

You set **START_DATE**, **END_DATE**, **BTM_RATIOS** (e.g. `[0.2, 0.5, 0.8]`), and **SCALER_INPUTS** (e.g. `[1.0]`) at the bottom of `scenario_analysis.py`. It runs the same model once per combination (e.g. btm0.2+scale1.0, btm0.5+scale1.0, btm0.8+scale1.0).

| Output | Meaning |
|--------|--------|
| **outputs/scenario_results.csv** | One row per scenario (btm_ratio, scaler, net benefit, etc.) for this run. Overwritten each time you run scenario_analysis. |
| **outputs/scenario_master.html** | Entry page with links to all scenario dashboards from this run. |
| **outputs/scenario_comparison_dashboard.html** | Comparison charts (e.g. heatmaps) across scenarios. |
| **outputs/scenario_insights.html** | Detailed insights across scenarios. |
| **outputs/scenario_outputs/dashboard_btm0.2_scale1.0.html** (etc.) | One HTML per scenario. Name = `btm{ratio}_scale{scaler}` (e.g. 0.2 = 20% BTM). |

So: **outputs/scenario_*.html** and **outputs/scenario_outputs/dashboard_btm*.html** always correspond to the **last** `scenario_analysis.py` run (dates and BTM/scaler settings in that script).

**Regenerate scenario dashboards from saved scenario results:**  
If you saved with `save_scenario_results` in `scenario_analysis.py`, use  
`python generate_dashboard.py --scenarios` (and optionally `--file results/scenarios_*....pkl`).

---

### 3. Results in `results/` used by `view_results.py`

**view_results.py** looks for:

- **results/dashboard_data_btm20_2024-01-01_to_2024-12-31.pkl.gz** (and btm50, btm100, etc.)
- **results/summary_btm20_2024-01-01_to_2024-12-31.json** (and btm50, btm100)

Those are **full-year 2024** runs with **20%, 50%, 100% BTM**. Naming uses **btm20 / btm50 / btm100** (integer percent).  
**No script in this repo writes** `dashboard_data_*.pkl.gz` or `summary_*.json`; they were produced by another process (e.g. an older or custom export). **view_results.py** only reads them to list scenarios and build `outputs/dashboard_btm20.html`, `outputs/comparison_btm20.csv`, and `outputs/master_dashboard.html`.

So:

- **Single-run outputs** → from `model_multi_battery.py` (and optionally `generate_dashboard.py` from `results/optimization_*.pkl`).
- **Scenario outputs** (outputs/scenario_*.html, outputs/scenario_outputs/) → from `scenario_analysis.py` (and optionally `generate_dashboard.py --scenarios` from `results/scenarios_*.pkl`).
- **btm20/btm50/btm100 dashboards and master_dashboard** → from `view_results.py` into **outputs/** when you have the matching `dashboard_data_*.pkl.gz` (and summary_*.json) in `results/`.

**outputs/optimization_dashboard_final.html** – One-off dashboard from the Dec 2025 “Energy Now presentation” run (full year 2024, BTM 0.2 / 0.5 / 0.8). Same data as in the scenario run; just a separate export.

---

## Typical workflows

- **One VPP run, then look at results:**  
  `python model_multi_battery.py` → open the HTML it prints at the end (in **outputs/**).

- **Compare different BTM/FTM splits and load sizes:**  
  `python scenario_analysis.py` → then `python generate_dashboard.py --scenarios` and open the scenario dashboards.

- **Re-build dashboards without re-optimizing:**  
  `python generate_dashboard.py` (uses latest saved run) or `python generate_dashboard.py --file results/optimization_....pkl`.

- **Use pre-made scenario result files:**  
  `python view_results.py --list` → `python view_results.py --scenario btm50` or `--master`.

- **Regenerate all dashboards from existing results:**  
  - **BTM20/50/100 + master:** requires `results/dashboard_data_btm*.pkl.gz` → `python view_results.py --all` then `python view_results.py --master`.  
  - **Single-run (portfolio + site):** requires `results/optimization_*.pkl` → `python generate_dashboard.py`.  
  - **Scenario comparison/insights:** requires `results/scenarios_*.pkl` (saved by `scenario_analysis.py`) → `python generate_dashboard.py --scenarios`. If no `scenarios_*.pkl` exists, scenario HTMLs are not regenerated.

---

## Main ideas (in simple words)

- **BTM (behind-the-meter):** Battery is used on your side of the meter: reduce peak demand, shift load to cheap hours. Saves you money on energy and peak tariffs.
- **FTM (front-of-meter):** Battery capacity is offered to the grid for **FCR** (frequency containment reserve). You get paid for capacity and possibly for activation.
- **Optimization:** The model decides when to charge and discharge and how much to bid in FCR so that **total benefit** (savings + FCR revenue minus degradation and other costs) is as high as possible over the chosen period.

---
