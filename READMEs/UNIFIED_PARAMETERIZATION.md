# Unified Parameterization: E0_g_per_L + k_d Only

## Overview

The entire codebase has been unified on a **single, consistent parameterization**:
- **Inferred parameters**: `['E0_g_per_L', 'k_d']` (2 parameters)
- **Known inputs**: `['substrate_mM', 'grams_urease_powder', 'temperature_C', 'initial_pH', 'volume_L']` (5 inputs)
- **Removed**: `activity_scale`, `powder_activity_frac` (as known input), `tau_probe`, `pH_offset`

---

## Key Changes

### 1. **Data Generation** (`generate_early_inference_data.py`)

**Parameterization:**
```python
infer_params = ['E0_g_per_L', 'k_d']
known_input_names = ['substrate_mM', 'grams_urease_powder', 'temperature_C', 'initial_pH', 'volume_L']
```

**E0_g_per_L Range:**
```python
"E0_g_per_L": [5e-4, 2.5]  # Wide range covering slow to fast regimes [g/L]
```
- Lower bound `5e-4` ensures slow cases are included
- Upper bound `2.5` covers fast regimes

**Simulator Usage:**
```python
params = {
    'E_eff0': E0_g_per_L,  # Direct enzyme loading [g/L]
    'k_d': k_d,
    't_shift': 0.0,
    'tau_probe': 0.0,  # Not used (true pH space)
}
```

**Measurement Model:**
- True pH space only (no probe lag, no offset)
- Optional small noise: `noise_std = 0.01`

---

### 2. **Model Architecture** (`early_inference_model.py`)

**Defaults:**
```python
n_known_inputs = 5   # substrate_mM, grams_urease_powder, temperature_C, initial_pH, volume_L
n_output_params = 2  # E0_g_per_L, k_d
```

**Architecture:**
- TCN for pH sequence processing
- MLP for known inputs processing
- Output head: 2 parameters (mean + log-variance for uncertainty)

---

### 3. **Training** (`train_early_inference.py`)

**Automatic Adaptation:**
- Reads `infer_params` and `known_input_names` from metadata
- Automatically sets model dimensions:
  - Input: 5 known inputs
  - Output: 2 parameters

**No Changes Required:**
- Training script automatically adapts to data structure

---

### 4. **Forecasting** (`forecast_early_inference.py`)

**Input:**
```python
known_inputs = {
    'substrate_mM': 20.0,
    'grams_urease_powder': 0.1,
    'temperature_C': 40.0,
    'initial_pH': 7.0,
    'volume_L': 0.2,
    # NO powder_activity_frac
}
```

**Prediction:**
```python
estimated_params = {
    'E0_g_per_L': 0.8,  # [g/L]
    'k_d': 0.001,       # [1/s]
}
```

**Simulator Usage:**
```python
sim_params = {
    'E_eff0': estimated_params['E0_g_per_L'],  # Direct enzyme loading
    'k_d': estimated_params['k_d'],
    't_shift': 0.0,
    'tau_probe': 0.0,  # Not used (true pH space)
}
pH_forecast = sim.simulate_forward(sim_params, t_forecast, apply_probe_lag=False)
```

---

### 5. **Mechanistic Fitting** (`fit_mechanistic.py`)

**Bounds:**
```python
param_bounds = {
    'E0_g_per_L': (5e-4, 2.5),  # [g/L]
    'k_d': (0.0, 5e-3),          # [1/s]
}
```

**Initial Guess:**
```python
initial_guess = {
    'E0_g_per_L': 0.5,  # Mid-range
    'k_d': 0.001,       # Small deactivation
}
```

**Residual Function:**
```python
def residual(params_vec):
    sim_params = {
        'E_eff0': params_dict['E0_g_per_L'],  # Direct enzyme loading
        'k_d': params_dict['k_d'],
        't_shift': 0.0,
        'tau_probe': 0.0,  # Not used (true pH space)
    }
    pH_sim = sim.simulate_forward(sim_params, t_measured, apply_probe_lag=False)
    return pH_sim - pH_measured  # True pH space
```

---

### 6. **Evaluation** (`evaluate_early_inference.py`)

**Parameter Metrics:**
- Compares `E0_g_per_L` and `k_d` between ML, Fit, and Ground Truth
- Computes MAE, RMSE, Relative Error

**Derived Quantity (Reporting Only):**
```python
# Computed AFTER inference for interpretability
powder_activity_frac_derived = clip(
    E0_g_per_L * volume_L / grams_urease_powder,
    0.0, 1.0
)
```
- Stored in CSV for reporting
- **NOT used as input or target**

**CSV Outputs:**
- Per-sample: `sample_XXXX_trajectories.csv`
- Aggregate: `aggregate_metrics.csv`
- Includes derived `powder_activity_frac` for interpretability

---

## Consistency Checklist

✅ **Data Generation:**
- `infer_params = ['E0_g_per_L', 'k_d']`
- `known_input_names` = 5 inputs (no `powder_activity_frac`)
- E0_g_per_L range: `[5e-4, 2.5]` (includes slow cases)
- Uses `E_eff0` directly in simulator

✅ **Model:**
- `n_known_inputs = 5`
- `n_output_params = 2`

✅ **Training:**
- Automatically adapts to data structure

✅ **Forecasting:**
- Accepts 5 known inputs
- Outputs 2 parameters
- Uses `E_eff0` directly

✅ **Fitting:**
- Fits only `E0_g_per_L` and `k_d`
- Uses `E_eff0` directly
- True pH space (no lag/offset)

✅ **Evaluation:**
- Compares `E0_g_per_L` and `k_d`
- Computes derived `powder_activity_frac` for reporting
- Outputs include all parameter estimates

---

## Physical Interpretation

### E0_g_per_L (Effective Active Enzyme Loading)

**Definition:** Mass of active enzyme per liter of solution at t=0

**Units:** g/L

**Range:** 5e-4 to 2.5 g/L

**Physical Meaning:**
- Direct measure of enzyme concentration
- Determines initial reaction rate
- Physically interpretable (concentration)

**Advantages over activity_scale:**
- Direct physical quantity (not a multiplier)
- No redundancy with powder_activity_frac
- More interpretable for experimentalists

### k_d (Deactivation Rate)

**Definition:** First-order enzyme deactivation rate constant

**Units:** 1/s

**Range:** 0 to 5e-3 1/s

**Physical Meaning:**
- Rate at which enzyme loses activity over time
- Affects long-term pH trajectory
- Independent of initial loading

---

## Derived Quantity: powder_activity_frac

**Computation (Post-Inference):**
```python
powder_activity_frac_derived = clip(
    E0_g_per_L * volume_L / grams_urease_powder,
    0.0, 1.0
)
```

**Purpose:**
- Interpretability for experimentalists
- Relates inferred E0 to powder quality
- **NOT used in training or inference**

**Example:**
```python
E0_g_per_L = 0.8  # [g/L]
volume_L = 0.2    # [L]
grams_urease_powder = 0.1  # [g]

powder_activity_frac_derived = clip(0.8 * 0.2 / 0.1, 0, 1)
                              = clip(1.6, 0, 1)
                              = 1.0  # Clipped
```

**Interpretation:**
- If derived fraction > 1.0 (clipped): Inferred E0 suggests more active enzyme than physically possible from powder alone
- Likely indicates:
  - Underestimated powder mass
  - Higher-than-assumed powder activity
  - Model uncertainty

---

## Migration from Old Parameterization

### Old (Removed):
```python
infer_params = ['activity_scale', 'k_d', 'tau_probe', 'pH_offset']
known_inputs = ['substrate_mM', 'grams_urease_powder', 'temperature_C', 
                'initial_pH', 'powder_activity_frac', 'volume_L']
```

### New (Unified):
```python
infer_params = ['E0_g_per_L', 'k_d']
known_inputs = ['substrate_mM', 'grams_urease_powder', 'temperature_C', 
                'initial_pH', 'volume_L']
```

### Breaking Changes:
1. **Model incompatibility**: Old models (4 outputs) won't work with new data (2 outputs)
2. **Data incompatibility**: Old data (6 known inputs) won't work with new model (5 inputs)
3. **Must regenerate data and retrain model**

---

## Workflow

### 1. Generate Data
```bash
cd ML/early_inference
python generate_early_inference_data.py
```
- Generates data with E0_g_per_L parameterization
- 5 known inputs, 2 target parameters
- Includes slow cases (E0 ≥ 5e-4)

### 2. Train Model
```bash
python train_early_inference.py
```
- Automatically adapts to 5 inputs, 2 outputs
- No configuration changes needed

### 3. Evaluate
```bash
python evaluate_early_inference.py
```
- Compares ML vs Fit
- Outputs parameter metrics (E0_g_per_L, k_d)
- Includes derived powder_activity_frac for reporting

### 4. Forecast (Programmatic)
```python
from forecast_early_inference import forecast_ph

known_inputs = {
    'substrate_mM': 20.0,
    'grams_urease_powder': 0.1,
    'temperature_C': 40.0,
    'initial_pH': 7.0,
    'volume_L': 0.2,
}

pH_forecast, params = forecast_ph(
    pH_prefix, t_prefix, known_inputs,
    model_path="models_early_inference/best_model_prefix_30s.pt",
    t_forecast=np.linspace(30, 2030, 1000)
)

print(f"E0_g_per_L: {params['E0_g_per_L']:.4f} g/L")
print(f"k_d: {params['k_d']:.6f} 1/s")
```

---

## Benefits

### 1. Simplicity
- Fewer parameters to infer (2 vs 4)
- Fewer known inputs (5 vs 6)
- No redundant parameters

### 2. Physical Interpretability
- E0_g_per_L is a direct concentration (not a multiplier)
- Clear physical meaning for experimentalists

### 3. Consistency
- Single parameterization across all scripts
- No confusion about which parameters to use

### 4. Robustness
- Includes slow cases (E0 ≥ 5e-4)
- Wide range covers diverse regimes

### 5. True pH Space
- No probe lag/offset artifacts
- Simpler measurement model
- Focus on mechanistic parameters

---

## Verification

To verify the unified parameterization:

```bash
# Check data generation config
grep -A 5 "infer_params" generate_early_inference_data.py

# Check metadata after generation
cat Generated_Data_EarlyInference_20000/metadata.json | grep -A 10 "infer_params"

# Check model defaults
grep "n_known_inputs\|n_output_params" early_inference_model.py

# Check forecasting defaults
grep "infer_params\|known_input_names" forecast_early_inference.py

# Check fitting defaults
grep "param_bounds" fit_mechanistic.py
```

Expected outputs:
- `infer_params`: `['E0_g_per_L', 'k_d']`
- `known_input_names`: 5 inputs (no `powder_activity_frac`)
- `n_known_inputs`: 5
- `n_output_params`: 2

---

## Summary

**Status:** ✅ **Fully Unified**

**Parameterization:**
- Inferred: `E0_g_per_L` [g/L], `k_d` [1/s]
- Known: 5 inputs (no `powder_activity_frac`)
- Removed: `activity_scale`, `tau_probe`, `pH_offset`

**All Scripts Updated:**
1. ✅ `generate_early_inference_data.py`
2. ✅ `early_inference_model.py`
3. ✅ `train_early_inference.py`
4. ✅ `forecast_early_inference.py`
5. ✅ `fit_mechanistic.py`
6. ✅ `evaluate_early_inference.py`

**Date:** January 2026
