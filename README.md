# ✈️ Fuel Flow Prediction — NASA DASHlink Tail_687

Predicting aircraft fuel consumption from second-by-second flight recorder data using machine learning.

---

## Project Overview

Fuel is one of the largest operating costs for airlines, accounting for 20–30% of total expenses. This project builds regression models to predict **total engine fuel flow (lb/hr)** from flight state parameters recorded by the Flight Data Recorder (FDR) aboard a commercial regional jet.

The dataset comes from **NASA DASHlink** — a real, de-identified commercial aviation archive — making this a genuine applied machine learning project on operational flight data.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | [NASA DASHlink — Tail_687](https://c3.ndc.nasa.gov/dashlink/projects/85/) |
| Aircraft | Regional jet (de-identified) |
| Flights loaded | 100 (out of 4,000+ available) |
| Raw rows | 413,232 |
| Airborne rows (ALT > 1000 ft) | 312,435 |
| Sampling rate | 4 Hz → downsampled to 1 Hz |
| File format | MATLAB `.mat` files |

### Variable Dictionary

| Variable | Description | Units |
|----------|-------------|-------|
| `ALT` | Pressure Altitude | ft |
| `CAS` | Calibrated Airspeed | knots |
| `MACH` | Mach Number | ratio |
| `SAT` | Static Air Temperature | °C |
| `N1_1` | Engine 1 Fan Speed | % |
| `N1_2` | Engine 2 Fan Speed | % |
| `FF_1` | Engine 1 Fuel Flow | lb/hr |
| `FF_2` | Engine 2 Fuel Flow | lb/hr |
| `IVV` | Inertial Vertical Velocity | ft/min |

**Target variable:** `FF_total = FF_1 + FF_2`

---

## Project Structure

```
├── av_fixed.ipynb          # Main notebook (with all fixes applied)
├── README.md               # This file
└── Tail_687/               # NASA DASHlink .mat files (download separately)
```

---

## Notebook Pipeline

| Step | Description |
|------|-------------|
| 1 | Data loading from MATLAB `.mat` files |
| 2 | Data quality assessment (missing values, duplicates) |
| 3 | Exploratory Data Analysis (EDA) |
| 4 | Airborne filter — ALT > 1000 ft |
| 5 | Feature engineering |
| 6 | Model training — 4 algorithms |
| 7 | Model evaluation & comparison |
| 8 | Feature importance analysis |

---

## Feature Engineering

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `FF_total` | FF_1 + FF_2 | Regression target — total fuel burn |
| `N1_avg` | (N1_1 + N1_2) / 2 | Average engine power setting |
| `delta_T` | SAT − ISA_temp | Deviation from standard atmosphere |
| `mach_sq` | MACH² | Captures nonlinear aerodynamic drag |
| `n1_sq` | N1_avg² | Thrust scales with N1² |
| `abs_ivv` | \|IVV\| | Magnitude of vertical movement |

---

## Models

Four regression algorithms were trained and compared:

| Model | Notes |
|-------|-------|
| Linear Regression | Baseline; interpretable coefficients |
| Ridge Regression | L2 regularisation; handles collinearity |
| Random Forest | Ensemble of decision trees; captures nonlinear interactions |
| Gradient Boosting | Sequential tree boosting; strong on tabular data |

---

## Key Fixes Applied (v2 — `av_fixed.ipynb`)

The original notebook had three methodological issues that inflated R². These have been corrected:

### Fix 1 — False duplicates from slow sensors
**Problem:** FDR sensors like `FF_1`, `FF_2`, and `SAT` record at 0.25 Hz natively. The `.mat` file stores them at 4 Hz by repeating the last value, causing `pandas.duplicated()` to report 19,979 false duplicates.  
**Fix:** Added `t_sec = np.arange(n)` timestamp in `load_single_flight()` so consecutive rows with identical sensor values are never mistaken for duplicates.

### Fix 2 — N1 data leakage
**Problem:** `N1_avg` and `n1_sq` were included as features. N1 (fan speed) directly commands fuel flow via the engine FADEC system — the relationship `FF ≈ f(N1²)` is a near-deterministic engineering equation. Including N1 gave the model trivial access to the answer.  
**Fix:** Removed `N1_avg` and `n1_sq` from `FEATURES`. The model now predicts fuel flow from **external flight state only** (altitude, speed, temperature, vertical velocity) — the operationally meaningful version.

### Fix 3 — Temporal leakage in train/test split
**Problem:** `train_test_split` with random row-level shuffling caused adjacent seconds from the same flight to appear in both train and test sets. The model effectively memorised flights it had already seen.  
**Fix:** Replaced with `GroupShuffleSplit(groups=flight_id)` — entire flights go to either train or test, never both. This gives an honest estimate of performance on a completely unseen flight.

### Fix 4 — Redundant feature
**Problem:** `FL = ALT / 100` is a linear rescaling of `ALT` — carries zero additional information.  
**Fix:** Removed `FL` from `FEATURES`.

---

## How to Run

### 1. Download the data

Register (free) at NASA DASHlink and download Tail_687 `.mat` files:
```
https://c3.ndc.nasa.gov/dashlink/projects/85/
```
Place all `.mat` files inside a folder named `Tail_687/` in the same directory as the notebook.

### 2. Install dependencies

```bash
pip install numpy pandas scipy matplotlib seaborn scikit-learn jupyter
```

### 3. Run the notebook

```bash
jupyter notebook av_fixed.ipynb
```

For development, set `MAX_FILES = 100` in the Configuration cell.  
For full dataset, set `MAX_FILES = None`.

---

## Results Summary

> Results below are from the **original notebook** (with N1 leakage).  
> After applying fixes, expect R² in the range of **0.80 – 0.93** — a genuinely strong result.

| Model | MAE (lb/hr) | RMSE (lb/hr) | R² |
|-------|------------|--------------|-----|
| Linear Regression | 78.4 | 113.1 | 0.9899 |
| Ridge Regression | 78.6 | 113.1 | 0.9899 |
| Random Forest | 16.1 | 30.5 | 0.9993 ⚠️ |
| Gradient Boosting | 47.3 | 72.1 | 0.9959 ⚠️ |

⚠️ Inflated due to N1 leakage + row-level temporal leakage. Fixed in `av_fixed.ipynb`.

---

## Feature Importance (Original — with N1)

Top features identified by Random Forest:

| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | n1_sq | 0.391 |
| 2 | N1_avg | 0.304 |
| 3 | IVV | 0.209 |
| 4 | ALT | 0.044 |
| 5 | FL | 0.043 |

This confirms N1 was dominating the model — together accounting for **69.5% of importance**.

---

## Business Interpretation

**Key findings:**
1. Engine power setting (N1) is the strongest driver of fuel consumption
2. Vertical velocity (IVV) significantly affects fuel burn — climb phases are expensive
3. Altitude and speed contribute to fuel efficiency through aerodynamic effects
4. Atmospheric temperature deviation (delta_T) affects engine thermodynamic efficiency

**Recommendations for airlines:**
- Maintain optimal N1 settings during cruise to minimise unnecessary fuel burn
- Optimise climb profiles — steep climbs consume significantly more fuel
- Account for atmospheric conditions (hot/cold days) in fuel planning
- Deploy predictive fuel-flow models for real-time performance monitoring

---

## Limitations

1. **Gross Weight unavailable** — aircraft weight is a primary driver of fuel burn but is not present in Tail_687
2. **No wind data** — headwind/tailwind can change fuel burn by 5–15%
3. **Single aircraft type** — model is specific to the Tail_687 aircraft variant
4. **No aircraft configuration** — flap position, spoiler state, and engine bleed settings are excluded

---

## References

- NASA DASHlink: https://c3.ndc.nasa.gov/dashlink/projects/85/
- Herskovits, M. et al. (2013). *Aviation Safety Information Analysis and Sharing (ASIAS)*
- ICAO Doc 9157 — Aerodrome Design Manual

---

## Author

Built as part of an applied machine learning regression project using real NASA aviation data.
