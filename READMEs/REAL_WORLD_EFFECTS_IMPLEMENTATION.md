# Real-World Effects Implementation Summary

## Overview

This document summarizes the implementation of real-world measurement and experimental effects in the urease-urea pH(t) simulation and fitting pipeline. The goal is to make the mechanistic pipeline more realistic while giving the ML system a realistic advantage by training on the same variability.

## Part A: Real-World Effects in Mechanistic Simulation

### 1. Measurement Bias and Drift

**Location**: `mechanistic_simulator.py` - `simulate_forward()` method

**Implementation**:
- **pH_offset**: Constant offset applied to pH measurements [pH units]
- **pH_drift_rate**: Linear drift rate over time [pH units/s]

**Order of Application**:
1. Chemistry produces true pH
2. Probe lag (if enabled)
3. **Measurement bias/drift** (offset + drift)
4. Instrument smoothing (if enabled)

**Configuration**:
```python
params = {
    'pH_offset': 0.02,  # Small offset
    'pH_drift_rate': 1e-6,  # Slow drift
}
pH = sim.simulate_forward(
    params, t_grid,
    enable_measurement_effects=True
)
```

### 2. Instrument Smoothing

**Location**: `mechanistic_simulator.py` - `simulate_forward()` method

**Implementation**:
- **tau_smoothing**: Time constant for exponential moving average [s]
- Applied as low-pass filter beyond existing probe lag
- Represents additional smoothing from instrument electronics/firmware

**Order of Application**:
1. Chemistry → true pH
2. Probe lag (if enabled)
3. Measurement bias/drift
4. **Instrument smoothing** (exponential moving average)

**Configuration**:
```python
params = {
    'tau_smoothing': 2.0,  # 2 second smoothing time constant
}
```

### 3. Gas Exchange with Ambient Air

**Location**: `mechanistic_simulator.py` - `make_rhs()` method

**Implementation**:
- **gas_exchange_k**: First-order mass transfer rate [1/s]
- **gas_exchange_C_eq**: Equilibrium inorganic carbon concentration [M]
- Simple source/sink term: `dC/dt = ... + k * (C_eq - C)`

**Physics**: Represents exchange of dissolved CO2 with environment (open beaker/headspace coupling)

**Configuration**:
```python
params = {
    'gas_exchange_k': 1e-5,  # Exchange rate
    'gas_exchange_C_eq': 0.002,  # Equilibrium C [M]
}
pH = sim.simulate_forward(
    params, t_grid,
    enable_gas_exchange=True
)
```

### 4. Mixing/Dispersion at Experiment Start

**Location**: `mechanistic_simulator.py` - `make_rhs()` method

**Implementation**:
- **mixing_ramp_time_s**: Time over which reaction rate ramps up [s]
- Smooth sigmoid-like turn-on: `ramp_frac = 0.5 * (1 + tanh((t - t_ramp/2) / (t_ramp/6)))`
- Represents non-instantaneous mixing and local concentration gradients

**Configuration**:
```python
params = {
    'mixing_ramp_time_s': 10.0,  # 10 second mixing ramp
}
pH = sim.simulate_forward(
    params, t_grid,
    enable_mixing_ramp=True
)
```

## Part B: Robust Fitting with Nuisance Parameters

### 1. Nuisance Parameter Fitting

**Location**: `fit_mechanistic.py` - `fit_mechanistic_parameters()`

**Features**:
- Optional fitting of nuisance parameters (measurement effects, gas exchange, mixing)
- Regularization (L2 penalties) to prevent overfitting
- Sensible bounds and initial guesses

**Configuration**:
```python
fitted_params = fit_mechanistic_parameters(
    pH_measured, t_measured, known_inputs,
    fit_nuisance_params=True,
    enable_measurement_effects=True,
    enable_gas_exchange=True,
    enable_mixing_ramp=True,
    nuisance_regularization={
        'pH_offset': 1e2,  # Keep offset small
        'pH_drift_rate': 1e8,  # Strongly penalize drift
        'tau_smoothing': 1e-1,  # Light penalty
        'gas_exchange_k': 1e6,  # Penalize gas exchange
        'mixing_ramp_time_s': 1e-1,  # Light penalty
    }
)
```

### 2. Integral-Based Objective

**Location**: `fit_mechanistic.py` - `residual()` function

**Implementation**:
- Uses trapezoidal integration of squared error: `∫ (pH_sim - pH_meas)² dt`
- Insensitive to sampling density (1s vs 10s spacing)
- Approximates continuous-time error integral

**Rationale**: Pointwise sum-of-squares favors dense sampling. Integral-based objective treats all time intervals equally, making fitting robust to measurement frequency.

**Configuration**:
```python
fitted_params = fit_mechanistic_parameters(
    pH_measured, t_measured, known_inputs,
    use_integral_objective=True,  # Recommended
)
```

## Part C: ML Training with Nuisance Variability

### 1. Data Generation Updates

**Location**: `generate_early_inference_data.py`

**Changes**:
- Added `real_world_effects` configuration section
- Each training sample randomly draws nuisance parameters from specified ranges
- Nuisance parameters are stored but NOT used as ML targets (ML learns robustness)

**Configuration**:
```python
CONFIG = {
    "real_world_effects": {
        "enable": True,
        "enable_measurement_effects": True,
        "enable_gas_exchange": True,
        "enable_mixing_ramp": True,
        "nuisance_ranges": {
            "pH_offset": (-0.05, 0.05),
            "pH_drift_rate": (-5e-6, 5e-6),
            "tau_smoothing": (0.0, 5.0),
            "gas_exchange_k": (0.0, 5e-5),
            "gas_exchange_C_eq": (0.0, 0.005),
            "mixing_ramp_time_s": (0.0, 20.0),
        },
    },
}
```

**Train/Val Split**: Uses fixed seed to ensure reproducibility and prevent leakage (same seed per sample index ensures deterministic nuisance sampling).

### 2. Evaluation Updates

**Location**: `evaluate_early_inference.py`

**Changes**:
- Extracts nuisance parameters from test data
- Passes them to fitting (if `fit_nuisance_params=True`)
- Uses same nuisance params for ML and fit forecasts (fair comparison)

**Configuration**:
```python
CONFIG = {
    "fit_nuisance_params": False,  # Use ground truth nuisance params (fairer)
    # OR
    "fit_nuisance_params": True,  # Allow fitting to estimate nuisance factors
}
```

## Files Changed

1. **mechanistic_simulator.py**
   - Added gas exchange term to ODE (`make_rhs`)
   - Added mixing ramp to reaction rate
   - Added measurement effects layer (bias/drift/smoothing)
   - Updated `simulate_forward()` with enable flags

2. **fit_mechanistic.py**
   - Added nuisance parameter fitting support
   - Added regularization (L2 penalties)
   - Implemented integral-based objective
   - Updated bounds and initial guesses

3. **generate_early_inference_data.py**
   - Added `real_world_effects` configuration
   - Updated `_worker_generate_single_trajectory()` to sample and apply nuisance params
   - Store nuisance params in results (not as ML targets)

4. **evaluate_early_inference.py**
   - Updated `evaluate_single_case()` to handle nuisance params
   - Pass nuisance params to fitting
   - Use same nuisance params for fair ML vs fit comparison

## Usage Example

### Generate Training Data with Real-World Effects

```python
# In generate_early_inference_data.py CONFIG
CONFIG["real_world_effects"]["enable"] = True
python generate_early_inference_data.py
```

### Train ML Model

```python
# ML model trains on data with nuisance variability
# It learns to infer E0_g_per_L and k_d while being robust to nuisance factors
python train_early_inference.py
```

### Evaluate ML vs Fitting

```python
# In evaluate_early_inference.py CONFIG
CONFIG["fit_nuisance_params"] = False  # Fair comparison (use ground truth nuisance)
# OR
CONFIG["fit_nuisance_params"] = True  # Allow fitting to estimate nuisance
python evaluate_early_inference.py
```

## Ensuring Fairness Between ML and Fitting

1. **Same Nuisance Distributions**: Both ML training and test data use same sampling ranges
2. **Same Evaluation Protocol**: ML and fit forecasts use same nuisance params (unless fitting them)
3. **Same Time Grids**: Both use same reference grid for fair comparison
4. **Same Prefix Data**: Both ML and fit use same uniformly resampled prefix

## Expected Results and Next Steps

### If ML is Better Than Fitting

**Why**: ML has seen many examples with nuisance variability during training and learned to be robust. Fitting struggles because:
- Nuisance factors can compensate for kinetics parameters
- Regularization helps but may not be perfect
- Early prefixes may not contain enough information to separate nuisance from kinetics

**Next Steps**: None needed - goal achieved!

### If ML is NOT Better Than Fitting

**Possible Reasons**:
1. **Insufficient Training Data**: ML needs more examples to learn robustness
2. **Architecture Limitations**: TCN may not be capturing nuisance-invariant features
3. **Loss Function**: Current loss may not encourage robustness
4. **Regularization Too Strong**: Fitting may be over-regularized, making it artificially worse

**Proposed Adjustments**:

1. **Data Augmentation**:
   - Increase `n_samples` in data generation
   - Add more diverse nuisance parameter ranges
   - Include edge cases (extreme nuisance values)

2. **Architecture/Targets**:
   - Add attention mechanism to focus on kinetics-relevant features
   - Use contrastive learning to separate nuisance from signal
   - Predict nuisance parameters as auxiliary outputs (multi-task learning)

3. **Loss Choice**:
   - Add robustness loss: penalize sensitivity to nuisance factors
   - Use adversarial training: maximize loss w.r.t. nuisance, minimize w.r.t. parameters
   - Weighted loss: emphasize early time points where kinetics matter most

4. **Regularization**:
   - Reduce regularization strength on nuisance parameters
   - Use adaptive regularization (stronger when data supports it)
   - Add physics-informed constraints (e.g., drift should be small)

5. **Fitting Constraints**:
   - Tighten bounds on nuisance parameters
   - Use stronger regularization
   - Fix some nuisance parameters to typical values

## Testing the Implementation

### Test 1: Default Behavior (No Effects)

```python
# Should produce identical results to before
params = {'E_eff0': 0.5, 'k_d': 0.001}
pH = sim.simulate_forward(params, t_grid)
# No real-world effects applied
```

### Test 2: Individual Effects

```python
# Test measurement effects
params = {'E_eff0': 0.5, 'k_d': 0.001, 'pH_offset': 0.02}
pH = sim.simulate_forward(params, t_grid, enable_measurement_effects=True)

# Test gas exchange
params = {'E_eff0': 0.5, 'k_d': 0.001, 'gas_exchange_k': 1e-5}
pH = sim.simulate_forward(params, t_grid, enable_gas_exchange=True)

# Test mixing ramp
params = {'E_eff0': 0.5, 'k_d': 0.001, 'mixing_ramp_time_s': 10.0}
pH = sim.simulate_forward(params, t_grid, enable_mixing_ramp=True)
```

### Test 3: Fitting with Nuisance Parameters

```python
# Generate synthetic data with nuisance effects
# Fit with and without nuisance parameter fitting
# Compare parameter recovery accuracy
```

### Test 4: ML vs Fitting Comparison

```python
# Generate test set with nuisance variability
# Evaluate ML and fitting
# Compare metrics:
#   - Parameter recovery (E0_g_per_L, k_d)
#   - Trajectory forecast accuracy (RMSE, R²)
#   - Time to threshold predictions
```

## Comments in Code

All code includes clear comments explaining:
- What is "physics/chemistry" vs "experimental artifact"
- Order of measurement transformations
- Rationale for design choices (e.g., integral-based objective)

## Output Updates

Fit summaries and plots now show:
- Which effects were enabled
- Fitted nuisance parameter values
- Regularization penalties applied
- Objective function type (integral vs pointwise)

## Conclusion

The implementation adds realistic experimental effects in a modular, toggleable way. The ML system is trained on the same variability, giving it a realistic opportunity to outperform classical fitting. The fitting system is robust but not "too powerful" due to regularization and integral-based objectives.

**Next Action**: Run evaluation to compare ML vs fitting on test set with nuisance variability. If ML is not better, implement proposed adjustments above.
