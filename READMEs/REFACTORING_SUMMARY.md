# Refactoring Summary: Removal of Measurement Model Nuisance Parameters

## Overview

This refactoring removes `tau_probe` and `pH_offset` from the inference set and focuses the pipeline on **core mechanistic parameter inference** (activity_scale, k_d) in **true pH space**.

---

## Changes Made

### 1. Data Generation (`generate_early_inference_data.py`)

**Removed:**
- `tau_probe` and `pH_offset` from `infer_params`
- `tau_probe` and `pH_offset` from `param_ranges`
- Probe lag and offset from measurement model (only noise remains)

**Updated:**
- `infer_params`: Now only `['activity_scale', 'k_d']`
- `measurement_model`: `use_probe_lag=False`, `use_offset=False` (only noise)
- Worker function: No longer extracts or uses tau_probe/pH_offset

**Result:**
- Training data in true pH space (with sensor noise only)
- Model learns to infer only mechanistic parameters

---

### 2. Model Architecture (`early_inference_model.py`)

**No changes needed** - Architecture is flexible (n_output_params is configurable)

**Result:**
- Model automatically adapts to 2 output parameters instead of 4

---

### 3. Training (`train_early_inference.py`)

**No changes needed** - Automatically uses correct number of parameters from data

**Result:**
- Training works with 2-parameter output

---

### 4. Forecasting (`forecast_early_inference.py`)

**Removed:**
- `apply_measurement_model()` function
- Measurement model application in `forecast_ph()`

**Updated:**
- Default `infer_params`: `['activity_scale', 'k_d']`
- Forecast returns true pH directly (no lag/offset)

**Result:**
- Forecasts are in true pH space (consistent with training)

---

### 5. Mechanistic Fitting (`fit_mechanistic.py`)

**Removed:**
- `apply_measurement_model()` function
- `tau_probe` and `pH_offset` from default bounds

**Updated:**
- Default bounds: Only `activity_scale` and `k_d`
- Residual function: Fits directly to true pH space (no measurement model)

**Result:**
- Fitting only estimates mechanistic parameters
- Direct comparison to true pH (with noise from data)

---

### 6. Evaluation (`evaluate_early_inference.py`)

**Complete rewrite** with new requirements:

**Added:**
- Per-sample CSV output (`sample_XXXX_trajectories.csv`)
  - Columns: `time_s`, `pH_ground_truth`, `pH_prefix`, `pH_ml_forecast`, `pH_fit_forecast`
- Aggregate CSV (`aggregate_metrics.csv`)
  - All parameter estimates, ground truth, and metrics per sample
- Test sample IDs JSON (`test_sample_ids.json`)
  - Deterministic seed, sample IDs, configuration

**Updated:**
- Parameter recovery comparison (ML vs Fit vs Ground Truth)
- Full-horizon forecast comparison (0-2000s)
- Systematic file naming
- Deterministic seed for reproducibility

**Result:**
- Complete evaluation pipeline with CSV outputs
- Reproducible test sets
- Easy plotting from saved CSVs

---

## Key Principles

### 1. True pH Space
- All pH curves are in **true pH space** (from mechanistic simulator)
- Only **sensor noise** is added (no lag, no offset)
- Forecasts and comparisons use true pH

### 2. Mechanistic Parameters Only
- Inference set: `['activity_scale', 'k_d']`
- No measurement model nuisance parameters
- Focus on physically meaningful parameters

### 3. Consistent Comparison
- ML and Fit both predict true pH
- Both compared to ground truth true pH
- Fair apples-to-apples comparison

### 4. Reproducibility
- Deterministic seeds for test set generation
- Saved sample IDs for exact reproducibility
- CSV outputs for easy analysis/plotting

---

## File Structure

```
evaluation_early_inference/
├── sample_0000_trajectories.csv    # Per-sample trajectory data
├── sample_0001_trajectories.csv
├── ...
├── aggregate_metrics.csv            # Summary metrics per sample
├── test_sample_ids.json             # Test set configuration
└── metrics.json                     # Aggregate statistics
```

---

## Usage

### Generate Data
```bash
python generate_early_inference_data.py
```
- Generates data with only noise (no lag/offset)
- Only 2 parameters to infer

### Train Model
```bash
python train_early_inference.py
```
- Trains on 2-parameter output
- Model learns: prefix pH → [activity_scale, k_d]

### Evaluate
```bash
python evaluate_early_inference.py
```
- Generates test set with deterministic seed
- Writes per-sample CSVs
- Writes aggregate CSV
- Saves test sample IDs

### Plot from CSVs
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load aggregate metrics
df = pd.read_csv('evaluation_early_inference/aggregate_metrics.csv')

# Load per-sample trajectory
traj = pd.read_csv('evaluation_early_inference/sample_0000_trajectories.csv')
plt.plot(traj['time_s'], traj['pH_ground_truth'], label='Ground Truth')
plt.plot(traj['time_s'], traj['pH_ml_forecast'], label='ML Forecast')
plt.plot(traj['time_s'], traj['pH_fit_forecast'], label='Fit Forecast')
```

---

## Expected Results

### Parameter Estimation
- **activity_scale**: MAE ~0.1-0.5 (depending on prefix length)
- **k_d**: MAE ~0.0005-0.002 (small values, relative error more meaningful)

### Trajectory Forecasting
- **RMSE**: ~0.05-0.15 pH units (full horizon 0-2000s)
- **R²**: 0.7-0.95 (should be positive, high values expected)
- **MAE**: ~0.04-0.12 pH units

### Comparison: ML vs Fit
- ML should be faster (amortized inference)
- ML may be more accurate for short prefixes (<30s)
- Fit may be more accurate for longer prefixes (>60s)

---

## Migration Notes

### Old Code (Before Refactoring)
```python
infer_params = ['activity_scale', 'k_d', 'tau_probe', 'pH_offset']
# Measurement model applied everywhere
pH_forecast = apply_measurement_model(pH_true, t, tau, offset)
```

### New Code (After Refactoring)
```python
infer_params = ['activity_scale', 'k_d']
# True pH space everywhere
pH_forecast = sim.simulate_forward(params, t)  # Direct true pH
```

### Breaking Changes
- Models trained before refactoring won't work (different output size)
- Must retrain with new data generation
- Evaluation outputs different format (CSVs instead of in-memory)

---

## Verification

Check that:
1. ✅ Data generation only has 2 parameters in `infer_params`
2. ✅ Model outputs 2 parameters (check `n_output_params`)
3. ✅ Forecasts are in true pH space (no measurement model)
4. ✅ Evaluation writes CSVs correctly
5. ✅ Test sample IDs are saved

---

## Next Steps

1. **Regenerate data** with new configuration
2. **Retrain model** on 2-parameter output
3. **Run evaluation** to generate CSVs
4. **Create plotting scripts** that read from CSVs
5. **Compare results** to previous 4-parameter version

---

**Date:** January 2026  
**Status:** ✅ Complete
