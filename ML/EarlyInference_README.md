# Early Inference ML Pipeline for Urease pH Kinetics

This pipeline implements an amortized early inference approach for predicting latent kinetic/deactivation parameters from the first ~10–60 seconds of measured pH(t) data, then using those parameters with the mechanistic simulator to forecast future pH trajectories.

## Overview

The pipeline consists of:

1. **Dataset Generation** (`generate_early_inference_data.py`): Generates training examples with:
   - Full pH trajectories from mechanistic simulator
   - Measurement model (probe lag, noise, offset)
   - Multiple prefix lengths (10s, 30s, 60s) extracted from each trajectory
   - Known inputs (substrate, temperature, etc.) and target parameters (activity_scale, k_d, etc.)

2. **Model Architecture** (`early_inference_model.py`): TCN-based sequence-to-parameter regression model with uncertainty quantification

3. **Training** (`train_early_inference.py`): Trains the early inference model on prefix sequences

4. **Forecasting** (`forecast_early_inference.py`): Uses ML estimator + mechanistic simulator to forecast future pH

5. **Evaluation** (`evaluate_early_inference.py`): Benchmarks ML early inference vs mechanistic-only parameter fitting

6. **Mechanistic Fitting** (`fit_mechanistic.py`): Classical least-squares parameter fitting for comparison

## Quick Start

### 1. Generate Training Data

```bash
cd ML
python generate_early_inference_data.py
```

This creates a dataset in `Generated_Data_EarlyInference_20000/` with:
- Multiple prefix lengths (10s, 30s, 60s)
- Measurement model applied (probe lag, noise, offset)
- Known inputs and target parameters

### 2. Train Model

```bash
python train_early_inference.py
```

Edit `CONFIG` in `train_early_inference.py` to:
- Set `prefix_length` (10.0, 30.0, or 60.0)
- Adjust hyperparameters (batch size, learning rate, etc.)
- Configure model architecture

Trained models are saved to `models_early_inference/`.

### 3. Evaluate

```bash
python evaluate_early_inference.py
```

This compares:
- **ML early inference**: Uses trained model to estimate parameters from prefix, then forecasts
- **Mechanistic fitting**: Uses least-squares optimization on prefix, then forecasts

Results include:
- Parameter estimation accuracy (MAE, RMSE, relative error)
- Forecast accuracy (RMSE, MAE, R²) for multiple horizons
- Time-to-threshold predictions
- Comparison plots

## Configuration

### Dataset Generation (`generate_early_inference_data.py`)

Key parameters in `CONFIG`:
- `n_samples`: Number of full trajectories to generate
- `prefix_lengths`: List of prefix lengths to extract (e.g., [10.0, 30.0, 60.0])
- `infer_params`: Parameters to infer (e.g., ['activity_scale', 'k_d', 'tau_probe', 'pH_offset'])
- `param_ranges`: Sampling ranges for all parameters
- `measurement_model`: Configuration for probe lag, noise, offset

### Training (`train_early_inference.py`)

Key parameters:
- `prefix_length`: Which prefix length to train on (must match one in dataset)
- `batch_size`, `epochs`, `lr`: Training hyperparameters
- `tcn_channels`, `mlp_hidden_dims`: Model architecture
- `use_uncertainty`: Enable uncertainty quantification (mean + log-variance)

### Evaluation (`evaluate_early_inference.py`)

Key parameters:
- `prefix_length`: Length of prefix to reveal (e.g., 30.0s)
- `forecast_horizons`: List of forecast horizons to evaluate (e.g., [300.0, 1000.0, 2000.0])
- `fit_bounds`: Parameter bounds for mechanistic fitting

## Model Architecture

The early inference model uses:

- **TCN (Temporal Convolutional Network)**: Processes pH sequence with dilated convolutions
- **MLP**: Processes known inputs (substrate, temperature, etc.)
- **Output Head**: Predicts parameter means and log-variances (if uncertainty enabled)

Input:
- pH sequence: (batch_size, seq_length)
- Known inputs: (batch_size, n_known_inputs)

Output:
- Parameter estimates: (batch_size, n_output_params)
- Uncertainty (optional): (batch_size, n_output_params) log-variance

## Forecasting Workflow

1. **Observe prefix**: Collect pH(t) for first 10–60 seconds
2. **Estimate parameters**: Use trained ML model to predict activity_scale, k_d, etc.
3. **Simulate forward**: Plug estimated parameters into `UreaseSimulator` to forecast future pH
4. **Make decisions**: Use forecast to decide when to add urease, change temperature, etc.

## Evaluation Metrics

The evaluation script reports:

### Parameter Metrics
- MAE, RMSE, Relative Error per parameter
- Comparison: ML vs Mechanistic fitting

### Trajectory Metrics
- RMSE, MAE, R² for forecasted pH
- Time-to-threshold error (e.g., predicted time to reach pH=8.0)
- Evaluated at multiple forecast horizons (300s, 1000s, 2000s)

### Outputs
- `metrics.json`: Aggregated metrics
- `sample_predictions.png`: Comparison plots showing prefix, true future, ML forecast, fit forecast

## File Structure

```
ML/
├── generate_early_inference_data.py    # Dataset generation
├── early_inference_model.py            # Model architecture
├── train_early_inference.py            # Training script
├── forecast_early_inference.py         # Forecast function
├── fit_mechanistic.py                  # Mechanistic fitting
├── evaluate_early_inference.py         # Evaluation benchmark
├── mechanistic_simulator.py            # Existing simulator (used by all)
└── EarlyInference_README.md            # This file
```

## Dependencies

- PyTorch
- NumPy
- SciPy
- Matplotlib
- tqdm

## Notes

- The pipeline is designed to work with the existing `mechanistic_simulator.py`
- Training data generation can be time-consuming (parallel processing is used)
- Model training supports GPU acceleration (auto-detected)
- Evaluation generates synthetic test cases; for real evaluation, generate a separate test set with full trajectories

## Example Usage

```python
from forecast_early_inference import forecast_ph
import numpy as np

# Observed prefix
pH_prefix = np.array([7.0, 7.1, 7.2, ...])  # First 30 seconds
t_prefix = np.linspace(0, 30, len(pH_prefix))

# Known conditions
known_inputs = {
    'substrate_mM': 20.0,
    'grams_urease_powder': 0.1,
    'temperature_C': 40.0,
    'initial_pH': 7.0,
    'powder_activity_frac': 1.0,
    'volume_L': 0.2,
}

# Forecast future pH
t_forecast = np.linspace(30, 2030, 1000)
pH_forecast, estimated_params = forecast_ph(
    pH_prefix, t_prefix, known_inputs,
    model_path="models_early_inference/best_model_prefix_30s.pt",
    t_forecast=t_forecast
)

print(f"Estimated activity_scale: {estimated_params['activity_scale']:.4f}")
print(f"Estimated k_d: {estimated_params['k_d']:.6f}")
```
