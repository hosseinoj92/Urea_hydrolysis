# Quick Reference: Unified Parameterization

## At a Glance

**Inferred Parameters (2):**
```python
['E0_g_per_L', 'k_d']
```

**Known Inputs (5):**
```python
['substrate_mM', 'grams_urease_powder', 'temperature_C', 'initial_pH', 'volume_L']
```

**Removed:**
- ❌ `activity_scale`
- ❌ `powder_activity_frac` (as known input)
- ❌ `tau_probe`
- ❌ `pH_offset`

---

## Parameter Ranges

```python
"E0_g_per_L": [5e-4, 2.5]  # [g/L] - includes slow cases
"k_d": [0.0, 5e-3]          # [1/s]
```

---

## Simulator Usage

```python
params = {
    'E_eff0': E0_g_per_L,  # Direct enzyme loading [g/L]
    'k_d': k_d,             # Deactivation rate [1/s]
    't_shift': 0.0,
    'tau_probe': 0.0,       # Not used (true pH space)
}
pH = sim.simulate_forward(params, t_grid, apply_probe_lag=False)
```

---

## Derived Quantity (Reporting Only)

```python
# Computed AFTER inference for interpretability
powder_activity_frac_derived = np.clip(
    E0_g_per_L * volume_L / grams_urease_powder,
    0.0, 1.0
)
```

---

## Workflow

```bash
# 1. Generate data
python generate_early_inference_data.py

# 2. Train model
python train_early_inference.py

# 3. Evaluate
python evaluate_early_inference.py
```

---

## Verification

```bash
# Check metadata
cat Generated_Data_EarlyInference_20000/metadata.json | grep -A 2 "infer_params"

# Expected: ["E0_g_per_L", "k_d"]
```

---

## Example (Forecasting)

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
    model_path="models_early_inference/best_model_prefix_30s.pt"
)

print(f"E0: {params['E0_g_per_L']:.4f} g/L")
print(f"k_d: {params['k_d']:.6f} 1/s")
```

---

## Status

✅ **All scripts unified** (January 2026)
