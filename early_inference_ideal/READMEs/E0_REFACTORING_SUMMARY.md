# Refactoring Summary: E0_g_per_L as Primary Activity Parameter

## Overview

This refactoring changes the inference framework to infer **effective active enzyme loading** (`E0_g_per_L`) directly, instead of inferring `activity_scale` or `powder_activity_frac`. This makes the parameter space more physically interpretable and eliminates the redundancy between activity scaling and powder activity fraction.

---

## Key Changes

### 1. Parameter Space

**Before:**
- Inferred: `['activity_scale', 'k_d']`
- Known inputs: `['substrate_mM', 'grams_urease_powder', 'temperature_C', 'initial_pH', 'powder_activity_frac', 'volume_L']`
- Relationship: `E_eff0 = activity_scale * grams_urease_powder * powder_activity_frac / volume_L`

**After:**
- Inferred: `['E0_g_per_L', 'k_d']`
- Known inputs: `['substrate_mM', 'grams_urease_powder', 'temperature_C', 'initial_pH', 'volume_L']`
- Direct usage: `E_eff0 = E0_g_per_L` (no scaling)
- Derived (for reporting): `powder_activity_frac_derived = clip(E0_g_per_L * volume_L / grams_urease_powder, 0, 1)`

---

## Implementation Details

### 1. Data Generation (`generate_early_inference_data.py`)

**Changes:**
- ✅ `infer_params`: `['E0_g_per_L', 'k_d']`
- ✅ `param_ranges`: Added `E0_g_per_L: [0.01, 2.0]` (removed `activity_scale`)
- ✅ Removed `powder_activity_frac` from `param_ranges` and `known_inputs`
- ✅ Worker function uses `E0_g_per_L` directly via `E_eff0` parameter
- ✅ Computes `powder_activity_frac_derived` for storage (reporting only)
- ✅ `known_input_names`: Now 5 inputs (powder_activity_frac removed)

**Simulator Usage:**
```python
params = {
    'E_eff0': E0_g_per_L,  # Direct use, no scaling
    'k_d': k_d,
    't_shift': 0.0,
    'tau_probe': 0.0,
}
```

---

### 2. Model Architecture (`early_inference_model.py`)

**Changes:**
- ✅ Updated default `n_known_inputs`: 6 → 5
- ✅ Updated default `n_output_params`: 4 → 2
- ✅ Architecture unchanged (automatically adapts to data)

---

### 3. Forecasting (`forecast_early_inference.py`)

**Changes:**
- ✅ Default `infer_params`: `['E0_g_per_L', 'k_d']`
- ✅ `known_input_names`: Removed `powder_activity_frac`
- ✅ Simulator uses `E_eff0` directly (no `activity_scale`)

**Forecast Flow:**
```python
ml_params = {'E0_g_per_L': 0.8, 'k_d': 0.001}
sim_params = {
    'E_eff0': ml_params['E0_g_per_L'],  # Direct use
    'k_d': ml_params['k_d'],
}
pH_forecast = sim.simulate_forward(sim_params, t_forecast)
```

---

### 4. Mechanistic Fitting (`fit_mechanistic.py`)

**Changes:**
- ✅ Default bounds: `{'E0_g_per_L': (0.01, 2.0), 'k_d': (0.0, 5e-3)}`
- ✅ Default initial guess: `{'E0_g_per_L': 0.5, 'k_d': 0.0}`
- ✅ Removed `powder_activity_frac` from known inputs
- ✅ Residual function uses `E_eff0` directly

---

### 5. Evaluation (`evaluate_early_inference.py`)

**Changes:**
- ✅ Fit bounds: Uses `E0_g_per_L` instead of `activity_scale`
- ✅ Computes derived `powder_activity_frac` for ML and Fit (reporting only)
- ✅ CSV outputs include:
  - `ml_E0_g_per_L`, `fit_E0_g_per_L`, `true_E0_g_per_L`
  - `ml_k_d`, `fit_k_d`, `true_k_d`
  - `ml_powder_activity_frac_derived`, `fit_powder_activity_frac_derived`, `true_powder_activity_frac_derived`
  - Parameter errors (MAE) for `E0_g_per_L` and `k_d`

**Derived Powder Activity Fraction:**
```python
# Computed after inference (for interpretability)
powder_activity_frac_derived = clip(
    E0_g_per_L_hat * volume_L / grams_urease_powder,
    0.0, 1.0
)
```

---

## Physical Interpretation

### E0_g_per_L (Effective Active Enzyme Loading)

**Definition:** Mass of active enzyme per liter of solution at t=0

**Units:** g/L

**Range:** 0.01 - 2.0 g/L (configurable)

**Physical Meaning:**
- Direct measure of enzyme concentration
- Determines initial reaction rate
- More interpretable than activity_scale (which was a multiplier)

### Relationship to Powder Activity Fraction

**Before (inferred):**
```python
powder_activity_frac = 0.2  # 20% of powder is active enzyme
activity_scale = 1.5        # Additional 50% activity multiplier
E_eff0 = activity_scale * grams * frac / volume
```

**After (E0_g_per_L inferred, frac derived):**
```python
E0_g_per_L = 0.8  # Direct: 0.8 g active enzyme per liter
powder_activity_frac_derived = E0_g_per_L * volume / grams
                            = 0.8 * 0.2 / 0.1 = 1.6 → clipped to 1.0
```

**Advantages:**
- ✅ Single parameter captures total activity
- ✅ No redundancy between activity_scale and powder_activity_frac
- ✅ Physically meaningful (concentration)
- ✅ Derived frac provides interpretability without being a target

---

## Data Structure Changes

### Known Inputs (5 parameters)

```python
known_inputs = {
    'substrate_mM': 20.0,
    'grams_urease_powder': 0.1,
    'temperature_C': 40.0,
    'initial_pH': 7.0,
    'volume_L': 0.2,
    # powder_activity_frac REMOVED
}
```

### Target Parameters (2 parameters)

```python
target_params = {
    'E0_g_per_L': 0.8,  # [g/L] - inferred directly
    'k_d': 0.001,       # [1/s] - deactivation rate
}
```

### Derived Quantities (for reporting)

```python
powder_activity_frac_derived = clip(
    E0_g_per_L * volume_L / grams_urease_powder,
    0.0, 1.0
)
# Example: 0.8 * 0.2 / 0.1 = 1.6 → 1.0 (clipped)
```

---

## Evaluation Outputs

### Per-Sample CSV (`sample_XXXX_trajectories.csv`)

```csv
time_s,pH_ground_truth,pH_prefix,pH_ml_forecast,pH_fit_forecast,is_prefix
0.0,7.36,7.36,7.36,7.36,1
5.0,7.45,7.44,7.46,7.45,1
...
```

### Aggregate CSV (`aggregate_metrics.csv`)

```csv
sample_id,ml_E0_g_per_L,ml_k_d,fit_E0_g_per_L,fit_k_d,true_E0_g_per_L,true_k_d,
         ml_E0_g_per_L_mae,ml_k_d_mae,fit_E0_g_per_L_mae,fit_k_d_mae,
         ml_powder_activity_frac_derived,fit_powder_activity_frac_derived,true_powder_activity_frac_derived,
         ml_rmse,ml_mae,ml_r2,fit_rmse,fit_mae,fit_r2,...
0,0.82,0.0012,0.79,0.0013,0.80,0.0011,0.02,0.0001,0.01,0.0002,1.64,1.58,1.60,0.05,0.04,0.95,...
```

**Key Columns:**
- `*_E0_g_per_L`: Effective enzyme loading estimates/ground truth
- `*_k_d`: Deactivation rate estimates/ground truth
- `*_powder_activity_frac_derived`: Derived fraction (for interpretability)
- `*_*_mae`: Parameter estimation errors
- Trajectory metrics: RMSE, MAE, R²

---

## Migration Notes

### Breaking Changes

1. **Model Compatibility:**
   - Models trained before refactoring won't work (different output size: 4 → 2 params)
   - Must retrain with new data generation

2. **Known Inputs:**
   - `powder_activity_frac` removed from known inputs
   - Model input dimension: 6 → 5 features

3. **Parameter Names:**
   - `activity_scale` → `E0_g_per_L`
   - All code references updated

### Required Actions

1. **Regenerate Data:**
   ```bash
   python generate_early_inference_data.py
   ```
   - Uses new parameter structure
   - Only 2 parameters to infer
   - 5 known inputs

2. **Retrain Model:**
   ```bash
   python train_early_inference.py
   ```
   - Model automatically adapts to 2-parameter output
   - Input dimension: 5 (powder_activity_frac removed)

3. **Re-evaluate:**
   ```bash
   python evaluate_early_inference.py
   ```
   - Outputs include `E0_g_per_L` and derived `powder_activity_frac`

---

## Benefits

### 1. Physical Interpretability

**Before:**
- `activity_scale = 1.5` → What does this mean? 50% more than what?
- Requires knowledge of base loading and powder activity to interpret

**After:**
- `E0_g_per_L = 0.8` → Direct: 0.8 grams of active enzyme per liter
- Immediately interpretable without additional context

### 2. Reduced Parameter Redundancy

**Before:**
- Two parameters (`activity_scale`, `powder_activity_frac`) both affect same quantity
- Can be confusing: is low activity due to bad powder or scaling?

**After:**
- Single parameter (`E0_g_per_L`) captures total activity
- Derived `powder_activity_frac` provides interpretability without redundancy

### 3. Simpler Inference

**Before:**
- Must infer both `activity_scale` and use known `powder_activity_frac`
- Relationship: `E_eff0 = scale * grams * frac / volume`

**After:**
- Direct inference of `E0_g_per_L`
- Relationship: `E_eff0 = E0_g_per_L` (trivial)

### 4. Better for Experimental Design

**Use Case:** Testing different enzyme batches
- **Before:** Need to separate "powder quality" (frac) from "activity scaling" (scale)
- **After:** Just measure `E0_g_per_L` directly (total active enzyme concentration)

---

## Example Usage

### Inferring Parameters

```python
from forecast_early_inference import forecast_ph

# Observed prefix
pH_prefix = np.array([7.0, 7.1, 7.2, ...])  # First 30 seconds
t_prefix = np.linspace(0, 30, len(pH_prefix))

# Known conditions (powder_activity_frac NOT needed)
known_inputs = {
    'substrate_mM': 20.0,
    'grams_urease_powder': 0.1,
    'temperature_C': 40.0,
    'initial_pH': 7.0,
    'volume_L': 0.2,
}

# Forecast
pH_forecast, estimated_params = forecast_ph(
    pH_prefix, t_prefix, known_inputs,
    model_path="models_early_inference/best_model_prefix_30s.pt",
    t_forecast=np.linspace(30, 2030, 1000)
)

# Direct parameter estimates
print(f"Effective enzyme loading: {estimated_params['E0_g_per_L']:.4f} g/L")
print(f"Deactivation rate: {estimated_params['k_d']:.6f} 1/s")

# Derived interpretability (optional)
powder_activity_frac = np.clip(
    estimated_params['E0_g_per_L'] * known_inputs['volume_L'] / known_inputs['grams_urease_powder'],
    0.0, 1.0
)
print(f"Derived powder activity fraction: {powder_activity_frac:.2f}")
```

---

## Validation Checklist

After refactoring, verify:

- [ ] Data generation creates `E0_g_per_L` in `infer_params`
- [ ] `known_inputs` has 5 elements (powder_activity_frac removed)
- [ ] Model outputs 2 parameters (`E0_g_per_L`, `k_d`)
- [ ] Forecasts use `E_eff0` parameter directly
- [ ] Evaluation CSVs include `E0_g_per_L` columns
- [ ] Derived `powder_activity_frac` computed and stored
- [ ] No references to `activity_scale` in inference code (only in docs/simulator defaults)

---

## Files Modified

### Core Implementation
1. ✅ `generate_early_inference_data.py` - Parameter structure, simulator usage
2. ✅ `forecast_early_inference.py` - Default params, simulator usage
3. ✅ `fit_mechanistic.py` - Bounds, initial guess, residual function
4. ✅ `evaluate_early_inference.py` - Parameter handling, CSV outputs, derived frac
5. ✅ `early_inference_model.py` - Default dimensions (documentation)

### Documentation
6. ✅ `E0_REFACTORING_SUMMARY.md` - This document

---

## Summary

**Status:** ✅ Complete

**Key Achievement:** Direct inference of physically meaningful `E0_g_per_L` parameter instead of redundant `activity_scale`/`powder_activity_frac` combination.

**Impact:**
- More interpretable parameters (concentration vs. multipliers)
- Reduced redundancy (1 parameter instead of 2 for activity)
- Simpler inference (direct vs. compound relationship)
- Better for experimental design (direct measurement)

**Date:** January 2026
