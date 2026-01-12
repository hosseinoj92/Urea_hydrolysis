# Dimensionality Consistency Summary

## ✅ Status: FIXED

All files now have consistent dimensionality and `volume_L` is always fixed.

---

## Consistent Dimensionality Across All Files

### **Inferred Parameters (2):**
```python
infer_params = ['E0_g_per_L', 'k_d']
```

### **Known Inputs (5):**
```python
known_input_names = [
    'substrate_mM',       # [mM] - sampled
    'grams_urease_powder', # [g] - sampled
    'temperature_C',       # [°C] - sampled
    'initial_pH',          # [-] - sampled
    'volume_L',            # [L] - FIXED (from fixed_params)
]
```

**Important:** The order must be **exactly the same** across all files.

---

## Changes Made

### 1. **`generate_early_inference_data.py`** ✅

**Fixed:**
- `volume_L` removed from `param_ranges` (commented out with clear note)
- `volume_L` always comes from `fixed_params` = 0.2 L
- Updated `sample_parameters()` to explicitly handle fixed params
- Added comments clarifying order and fixed nature of `volume_L`
- `known_input_names` metadata includes comments about each input

**Key Code:**
```python
# Fixed parameters (always constant, never sampled)
"fixed_params": {
    "volume_L": 0.2,  # Fixed volume [L] - consistent across all samples
}

# In sample_parameters():
# Add fixed parameters (constant across all samples)
for param_name, fixed_value in CONFIG.get("fixed_params", {}).items():
    params[param_name] = np.full(n_samples, fixed_value)
```

### 2. **`early_inference_model.py`** ✅

**Fixed:**
- Updated default `n_known_inputs` from 6 → **5**
- Updated default `n_output_params` from 4 → **2**
- Added comments explaining unified parameterization

**Key Code:**
```python
def __init__(
    self,
    seq_length: int,
    n_known_inputs: int = 5,  # Unified: substrate_mM, grams_urease_powder, temperature_C, initial_pH, volume_L
    n_output_params: int = 2,  # Unified: E0_g_per_L, k_d
    ...
)
```

### 3. **`forecast_early_inference.py`** ✅

**Already Consistent:**
- Uses `known_input_names` from metadata (reads from model checkpoint)
- Default fallback has correct 5 inputs in correct order
- Order: `['substrate_mM', 'grams_urease_powder', 'temperature_C', 'initial_pH', 'volume_L']`

### 4. **`evaluate_early_inference.py`** ✅

**Already Consistent:**
- Reads `known_input_names` from metadata
- Uses same order when extracting known inputs

### 5. **`train_early_inference.py`** ✅

**Already Consistent:**
- Reads data from NPZ file (order determined by data generation)
- Automatically adapts to `known_inputs` shape from data
- No hardcoded dimensions

---

## Verification

### Check `volume_L` is Fixed

```python
# In generate_early_inference_data.py:
# ✅ volume_L is NOT in param_ranges (commented out)
# ✅ volume_L IS in fixed_params = 0.2

# In sample_parameters():
# ✅ Fixed params are added separately from sampled params
# ✅ All samples get the same volume_L = 0.2
```

### Check Order Consistency

**All files use this order:**
1. `substrate_mM`
2. `grams_urease_powder`
3. `temperature_C`
4. `initial_pH`
5. `volume_L` (fixed)

**Files Verified:**
- ✅ `generate_early_inference_data.py` (lines 471-475)
- ✅ `forecast_early_inference.py` (lines 143-146)
- ✅ `evaluate_early_inference.py` (reads from metadata)

---

## Data Structure

### Generated Data Shape

```python
# Training data NPZ structure:
{
    "prefix_30.0": {
        "pH_prefix": (n_samples, prefix_n_points),  # pH sequence
        "known_inputs": (n_samples, 5),  # 5 inputs in fixed order
        "target_params": (n_samples, 2),  # 2 parameters [E0_g_per_L, k_d]
    }
}

# known_inputs array order:
# Column 0: substrate_mM
# Column 1: grams_urease_powder
# Column 2: temperature_C
# Column 3: initial_pH
# Column 4: volume_L (always 0.2 for all samples)
```

### Metadata Structure

```python
{
    "known_input_names": [
        "substrate_mM",
        "grams_urease_powder",
        "temperature_C",
        "initial_pH",
        "volume_L",
    ],
    "infer_params": ["E0_g_per_L", "k_d"],
    "fixed_params": {"volume_L": 0.2},
    ...
}
```

---

## Benefits

### 1. Consistency
- ✅ All files use same dimensionality
- ✅ All files use same input order
- ✅ No confusion about which inputs go where

### 2. Fixed `volume_L`
- ✅ Always 0.2 L (constant across all samples)
- ✅ Not sampled (reduces parameter space)
- ✅ Clear documentation of fixed nature

### 3. Maintainability
- ✅ Single source of truth (generate_early_inference_data.py)
- ✅ Metadata stored for reproducibility
- ✅ Clear comments explain order and fixed params

---

## Usage Example

### Generating Data

```python
# volume_L is automatically fixed to 0.2 for all samples
params = sample_parameters(n_samples=1000, seed=42)

# All samples have:
# params['volume_L'] = np.full(1000, 0.2)  # All same value
# params['substrate_mM'] = random values
# params['grams_urease_powder'] = random values
# etc.
```

### Using in Forecasting

```python
# Must provide exactly 5 inputs in correct order
known_inputs = {
    'substrate_mM': 20.0,
    'grams_urease_powder': 0.1,
    'temperature_C': 40.0,
    'initial_pH': 7.0,
    'volume_L': 0.2,  # Must match fixed_params value
}

# Model expects 5 inputs (n_known_inputs=5)
# Model outputs 2 parameters (n_output_params=2)
```

---

## Summary

**✅ All Changes Complete:**

1. **`volume_L` is always fixed** at 0.2 L (from `fixed_params`)
2. **Dimensionality is consistent** across all files:
   - `n_known_inputs = 5`
   - `n_output_params = 2`
3. **Input order is consistent** across all files:
   - Order: `substrate_mM, grams_urease_powder, temperature_C, initial_pH, volume_L`
4. **All files updated** with correct defaults and comments

**Ready to use!** 🎉
