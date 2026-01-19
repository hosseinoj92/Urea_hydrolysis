"""
Evaluate traditional ML models and compare with mechanistic parameter fitting.

Similar to evaluate_early_inference.py but for traditional ML models (GPR, XGBoost, etc.).
"""

# Set matplotlib backend before importing pyplot to avoid tkinter threading issues
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

import torch
import numpy as np
from pathlib import Path
import json
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import pearsonr
from scipy.interpolate import interp1d
import pandas as pd
import warnings

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

from fit_mechanistic import fit_mechanistic_parameters
from generate_early_inference_data import extract_prefix, sample_parameters, build_time_grid, generate_trajectories
from mechanistic_simulator import UreaseSimulator

# ╔══════════════════════════════════════════════════════════════╗
# ║                       USER CONFIG                             ║
# ╚══════════════════════════════════════════════════════════════╝
CONFIG = {
    # Model and data paths
    "models_dir": r"C:\Users\vt4ho\Simulations\simulation_data\models\imperfect\version_experiment\traditional_ml_models",
    "data_dir": r"C:\Users\vt4ho\Simulations\simulation_data\generated_data\imperfect\version_experiment\Generated_Data_EarlyInference_50000",
    "output_dir": r"C:\Users\vt4ho\Simulations\simulation_data\evaluation\imperfect\version_experiment\traditional_ml_evaluation",
    
    # Evaluation parameters
    "n_test_samples": 100,              # Number of test cases
    "prefix_length": 30.0,              # Length of prefix to reveal [s]
    "t_max": 2000.0,                    # Full trajectory length [s]
    "reference_grid_dt": 5.0,           # Reference grid spacing [s] for fair comparison
    "seed": 12345,                       # Deterministic seed for test set
    
    # Parameter fitting bounds
    "fit_bounds": {
        "powder_activity_frac": (0.01, 1.0),  # Fraction of powder that is active enzyme [0-1]
        "k_d": (1e-5, 5e-3),
    },
    
    # Fitting configuration
    "fit_nuisance_params": True,  # If True, fit nuisance parameters; if False, use ground truth
}


def load_models(models_dir: Path, prefix_length: float):
    """Load all trained traditional ML models."""
    models = {}
    scaler = None
    
    # Load scaler
    scaler_file = models_dir / f"scaler_prefix_{int(prefix_length)}s.pkl"
    if scaler_file.exists():
        with open(scaler_file, "rb") as f:
            scaler = pickle.load(f)
    
    # Load metadata to see which models were trained
    metadata_file = models_dir / f"training_metadata_prefix_{int(prefix_length)}s.json"
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        trained_models = metadata.get("trained_models", [])
    else:
        # Try to find all model files
        trained_models = []
        for model_file in models_dir.glob("*_prefix_*.pkl"):
            if "scaler" not in model_file.name:
                model_name = model_file.name.split("_prefix_")[0]
                trained_models.append(model_name)
    
    # Load each model
    for model_name in trained_models:
        model_file = models_dir / f"{model_name}_prefix_{int(prefix_length)}s.pkl"
        if model_file.exists():
            try:
                with open(model_file, "rb") as f:
                    models[model_name] = pickle.load(f)
                print(f"Loaded {model_name}")
            except Exception as e:
                print(f"Warning: Could not load {model_name}: {e}")
    
    return models, scaler, metadata if metadata_file.exists() else {}


def prepare_features(pH_prefix: np.ndarray, t_prefix: np.ndarray, known_inputs: np.ndarray) -> np.ndarray:
    """Prepare feature vector (same as training)."""
    # Ensure inputs are 2D for consistent processing
    if pH_prefix.ndim == 1:
        pH_prefix = pH_prefix.reshape(1, -1)
        t_prefix = t_prefix.reshape(1, -1)
        known_inputs = known_inputs.reshape(1, -1)
    
    n_samples = pH_prefix.shape[0]
    
    # pH sequence features
    pH_mean = np.mean(pH_prefix, axis=1)
    pH_std = np.std(pH_prefix, axis=1)
    pH_min = np.min(pH_prefix, axis=1)
    pH_max = np.max(pH_prefix, axis=1)
    pH_initial = pH_prefix[:, 0]
    pH_final = pH_prefix[:, -1]
    pH_range = pH_max - pH_min
    pH_change = pH_final - pH_initial
    
    # Time sequence features
    t_mean = np.mean(t_prefix, axis=1)
    t_std = np.std(t_prefix, axis=1)
    t_max = np.max(t_prefix, axis=1)
    t_final = t_prefix[:, -1]
    
    # pH rate of change
    if pH_prefix.shape[1] > 1:
        pH_diff = np.diff(pH_prefix, axis=1)
        t_diff = np.diff(t_prefix, axis=1)
        t_diff_safe = np.where(t_diff > 1e-8, t_diff, 1e-8)
        pH_rate = pH_diff / t_diff_safe
        pH_rate_mean = np.mean(pH_rate, axis=1)
        pH_rate_max = np.max(pH_rate, axis=1)
    else:
        pH_rate_mean = np.zeros(n_samples)
        pH_rate_max = np.zeros(n_samples)
    
    # Early vs late pH
    n_early = max(1, int(0.3 * pH_prefix.shape[1]))
    n_late = max(1, int(0.3 * pH_prefix.shape[1]))
    pH_early_mean = np.mean(pH_prefix[:, :n_early], axis=1)
    pH_late_mean = np.mean(pH_prefix[:, -n_late:], axis=1)
    pH_early_late_diff = pH_late_mean - pH_early_mean
    
    # Combine all features
    features = np.column_stack([
        pH_mean, pH_std, pH_min, pH_max, pH_initial, pH_final, pH_range, pH_change,
        t_mean, t_std, t_max, t_final,
        pH_rate_mean, pH_rate_max,
        pH_early_mean, pH_late_mean, pH_early_late_diff,
        known_inputs,
    ])
    
    # Return 1D for single sample, 2D for batch
    if n_samples == 1:
        return features[0]  # Return 1D array for single sample
    else:
        return features  # Return 2D array for batch


def compute_metrics(y_true, y_pred):
    """Compute regression metrics."""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-12))
    
    if len(y_true) > 1:
        corr, _ = pearsonr(y_true, y_pred)
    else:
        corr = np.nan
    
    return {
        'MAE': float(mae),
        'RMSE': float(rmse),
        'R²': float(r2),
        'Correlation': float(corr),
    }


def time_to_threshold(pH: np.ndarray, t: np.ndarray, threshold: float) -> float:
    """Find time when pH crosses threshold."""
    if len(pH) == 0 or len(t) == 0:
        return np.nan
    
    mask = pH >= threshold
    if not np.any(mask):
        return np.nan
    
    idx = np.argmax(mask)
    if idx == 0:
        return t[0]
    
    t1, t2 = t[idx-1], t[idx]
    pH1, pH2 = pH[idx-1], pH[idx]
    
    if pH2 == pH1:
        return t1
    
    t_thresh = t1 + (threshold - pH1) * (t2 - t1) / (pH2 - pH1)
    return float(t_thresh)


def evaluate_single_case(
    sample_id: int,
    pH_meas_full: np.ndarray,
    pH_true_full: np.ndarray,
    t_full: np.ndarray,
    known_inputs: dict,
    true_params: dict,
    models: dict,
    scaler,
    metadata: dict,
    prefix_length: float,
    t_max: float,
    reference_grid_dt: float,
    output_dir: Path,
    nuisance_params: dict = None,
    fit_nuisance_params: bool = False,
) -> dict:
    """Evaluate a single test case for all models."""
    # Extract prefix
    prefix_n_points = metadata.get('prefix_n_points', 100)
    t_prefix_raw, pH_prefix_raw = extract_prefix(
        pH_meas_full, t_full, prefix_length, prefix_n_points
    )
    
    if len(t_prefix_raw) == 0:
        return None
    
    # Interpolate to exactly prefix_n_points
    if len(pH_prefix_raw) < prefix_n_points:
        t_prefix_uniform = np.linspace(0, prefix_length, prefix_n_points)
        pH_prefix = np.interp(t_prefix_uniform, t_prefix_raw, pH_prefix_raw)
    elif len(pH_prefix_raw) > prefix_n_points:
        indices = np.linspace(0, len(pH_prefix_raw) - 1, prefix_n_points, dtype=int)
        t_prefix_uniform = t_prefix_raw[indices]
        pH_prefix = pH_prefix_raw[indices]
    else:
        t_prefix_uniform = t_prefix_raw
        pH_prefix = pH_prefix_raw
    
    # Prepare known inputs array
    known_input_names = metadata.get('known_input_names', [])
    known_inputs_array = np.array([known_inputs[name] for name in known_input_names])
    
    # Prepare features
    features = prepare_features(pH_prefix, t_prefix_uniform, known_inputs_array)
    
    # Ensure features is 1D for single sample (prepare_features returns 1D for single sample)
    if features.ndim > 1:
        features = features.squeeze()
    if features.ndim == 0:
        features = features.reshape(-1)
    
    # Always reshape to 2D for models (single sample: (1, n_features))
    features_2d = features.reshape(1, -1) if features.ndim == 1 else features
    
    # Scale features if scaler available
    if scaler is not None:
        features_scaled = scaler.transform(features_2d)  # Already 2D
    else:
        features_scaled = features_2d  # Already 2D
    
    # Predict with all models
    model_predictions = {}
    for model_name, model in models.items():
        try:
            # Determine if model needs scaled features
            use_scaled = model_name in ["gpr", "mlp"]
            if use_scaled:
                # For scaled models, use features_scaled (already 2D)
                X_pred = features_scaled
            else:
                # For non-scaled models, use features_2d (already 2D)
                X_pred = features_2d
            
            # Double-check X_pred is 2D (should always be at this point)
            if X_pred.ndim == 1:
                X_pred = X_pred.reshape(1, -1)
            assert X_pred.ndim == 2, f"X_pred must be 2D, got shape {X_pred.shape}"
            
            pred = model.predict(X_pred)
            
            # Debug: print prediction shape for troubleshooting
            # print(f"DEBUG {model_name}: X_pred shape={X_pred.shape}, pred shape={pred.shape}, pred={pred}")
            
            # Handle prediction output shape
            # GPR and MLP can return different shapes for multi-output
            if pred.ndim == 1:
                if len(pred) == 2:
                    # Single prediction with 2 outputs (1D array with 2 elements)
                    model_predictions[model_name] = {
                        "powder_activity_frac": float(pred[0]),
                        "k_d": float(pred[1]),
                    }
                elif len(pred) == 1:
                    # Single output - this shouldn't happen for 2 outputs, but handle gracefully
                    print(f"Warning: {model_name} returned single output, expected 2. Shape: {pred.shape}")
                    model_predictions[model_name] = {
                        "powder_activity_frac": float(pred[0]),
                        "k_d": np.nan,
                    }
                else:
                    # Unexpected length
                    print(f"Warning: {model_name} returned unexpected 1D shape: {pred.shape}")
                    model_predictions[model_name] = {
                        "powder_activity_frac": np.nan,
                        "k_d": np.nan,
                    }
            elif pred.ndim == 2:
                # 2D prediction: (n_samples, n_outputs) or (n_outputs, n_samples)
                if pred.shape[0] == 1 and pred.shape[1] == 2:
                    # Standard case: (1, 2)
                    model_predictions[model_name] = {
                        "powder_activity_frac": float(pred[0, 0]),
                        "k_d": float(pred[0, 1]),
                    }
                elif pred.shape[0] == 2 and pred.shape[1] == 1:
                    # Transposed: (2, 1) - some models might return this
                    model_predictions[model_name] = {
                        "powder_activity_frac": float(pred[0, 0]),
                        "k_d": float(pred[1, 0]),
                    }
                else:
                    # Unexpected 2D shape
                    print(f"Warning: {model_name} returned unexpected 2D shape: {pred.shape}")
                    # Try to extract first two values
                    pred_flat = pred.flatten()
                    if len(pred_flat) >= 2:
                        model_predictions[model_name] = {
                            "powder_activity_frac": float(pred_flat[0]),
                            "k_d": float(pred_flat[1]),
                        }
                    else:
                        model_predictions[model_name] = {
                            "powder_activity_frac": np.nan,
                            "k_d": np.nan,
                        }
            else:
                # Unexpected dimensionality
                print(f"Warning: {model_name} returned unexpected shape: {pred.shape}, ndim={pred.ndim}")
                model_predictions[model_name] = {
                    "powder_activity_frac": np.nan,
                    "k_d": np.nan,
                }
        except Exception as e:
            print(f"Warning: Prediction failed for {model_name}: {e}")
            import traceback
            traceback.print_exc()
            model_predictions[model_name] = {
                "powder_activity_frac": np.nan,
                "k_d": np.nan,
            }
    
    # Mechanistic fitting
    try:
        enable_meas = (nuisance_params is not None and 
                      any(k in nuisance_params for k in ['pH_offset', 'pH_drift_rate', 'tau_smoothing']))
        enable_gas = (nuisance_params is not None and 
                     any(k in nuisance_params for k in ['gas_exchange_k', 'gas_exchange_C_eq']))
        enable_mix = (nuisance_params is not None and 
                     'mixing_ramp_time_s' in nuisance_params)
        
        fit_params = fit_mechanistic_parameters(
            pH_prefix, t_prefix_uniform, known_inputs,
            param_bounds=CONFIG["fit_bounds"],
            fit_nuisance_params=fit_nuisance_params,
            enable_measurement_effects=enable_meas,
            enable_gas_exchange=enable_gas,
            enable_mixing_ramp=enable_mix,
            use_integral_objective=True,
        )
    except Exception as e:
        print(f"Warning: Fitting failed for sample {sample_id}: {e}")
        fit_params = {k: np.nan for k in CONFIG["fit_bounds"].keys()}
    
    # Compute E0_g_per_L for all predictions
    true_powder_frac = true_params.get('powder_activity_frac', np.nan)
    true_E0_g_per_L = true_powder_frac * known_inputs['grams_urease_powder'] / known_inputs['volume_L'] if not np.isnan(true_powder_frac) else np.nan
    
    fit_powder_frac = fit_params.get('powder_activity_frac', np.nan)
    fit_E0_g_per_L = fit_powder_frac * known_inputs['grams_urease_powder'] / known_inputs['volume_L'] if not np.isnan(fit_powder_frac) else np.nan
    
    # Build simulator
    S0 = known_inputs['substrate_mM'] / 1000.0
    T_K = known_inputs['temperature_C'] + 273.15
    
    sim = UreaseSimulator(
        S0=S0, N0=0.0, C0=0.0, Pt_total_M=0.0,
        T_K=T_K, initial_pH=known_inputs['initial_pH'],
        E_loading_base_g_per_L=1.0,
        use_T_dependent_pH_activity=True,
    )
    
    # Create reference grid
    t_ref = np.arange(0.0, t_max + reference_grid_dt/2, reference_grid_dt)
    t_ref[-1] = t_max
    
    # Simulate forecasts for all models
    enable_meas = (nuisance_params is not None and 
                  any(k in nuisance_params for k in ['pH_offset', 'pH_drift_rate', 'tau_smoothing']))
    enable_gas = (nuisance_params is not None and 
                 any(k in nuisance_params for k in ['gas_exchange_k', 'gas_exchange_C_eq']))
    enable_mix = (nuisance_params is not None and 
                 'mixing_ramp_time_s' in nuisance_params)
    
    model_forecasts = {}
    for model_name, pred in model_predictions.items():
        try:
            pred_powder_frac = pred["powder_activity_frac"]
            if not np.isnan(pred_powder_frac):
                pred_E0 = pred_powder_frac * known_inputs['grams_urease_powder'] / known_inputs['volume_L']
                sim_params = {
                    'E_eff0': pred_E0,
                    'k_d': pred.get('k_d', 0.0),
                    't_shift': 0.0,
                    'tau_probe': 0.0,
                }
                if nuisance_params:
                    sim_params.update(nuisance_params)
                
                pH_forecast = sim.simulate_forward(
                    sim_params, t_ref, return_totals=False, apply_probe_lag=False,
                    enable_measurement_effects=enable_meas,
                    enable_gas_exchange=enable_gas,
                    enable_mixing_ramp=enable_mix,
                )
                model_forecasts[model_name] = pH_forecast
            else:
                model_forecasts[model_name] = None
        except Exception as e:
            print(f"Warning: Forecast failed for {model_name}: {e}")
            model_forecasts[model_name] = None
    
    # Fit forecast
    if not np.isnan(fit_powder_frac):
        sim_params_fit = {
            'E_eff0': fit_E0_g_per_L,
            'k_d': fit_params.get('k_d', 0.0),
            't_shift': 0.0,
            'tau_probe': 0.0,
        }
        if fit_nuisance_params:
            fitted_nuisance = {k: fit_params.get(k, 0.0) for k in 
                              ['pH_offset', 'pH_drift_rate', 'tau_smoothing',
                               'gas_exchange_k', 'gas_exchange_C_eq', 'mixing_ramp_time_s']
                              if k in fit_params}
            sim_params_fit.update(fitted_nuisance)
        elif nuisance_params:
            sim_params_fit.update(nuisance_params)
        
        pH_fit_full = sim.simulate_forward(
            sim_params_fit, t_ref, return_totals=False, apply_probe_lag=False,
            enable_measurement_effects=enable_meas,
            enable_gas_exchange=enable_gas,
            enable_mixing_ramp=enable_mix,
        )
    else:
        pH_fit_full = None
    
    # Interpolate ground truth
    interp_func = interp1d(t_full, pH_true_full, kind='linear', bounds_error=False, fill_value='extrapolate')
    pH_true_interp = interp_func(t_ref)
    
    # Plot trajectory for each model (similar to evaluate_early_inference.py)
    plot_max_t = 1000.0
    plot_mask = t_ref <= plot_max_t
    t_plot = t_ref[plot_mask]
    pH_true_plot = pH_true_interp[plot_mask]
    
    # Interpolate prefix for plotting
    prefix_interp = interp1d(t_prefix_uniform, pH_prefix, kind='linear', 
                            bounds_error=False, fill_value='extrapolate')
    pH_prefix_plot = prefix_interp(t_plot)
    prefix_mask_plot = t_plot <= prefix_length
    
    # Create plots for each model in separate folders
    for model_name in models.keys():
        model_output_dir = output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        if model_forecasts.get(model_name) is not None:
            pH_model_plot = model_forecasts[model_name][plot_mask]
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot prefix (observed data)
            ax.plot(t_plot[prefix_mask_plot], pH_prefix_plot[prefix_mask_plot], 
                    'ko', markersize=4, label='Prefix (observed)', alpha=0.6, zorder=5)
            
            # Plot ground truth
            ax.plot(t_plot, pH_true_plot, 'b-', linewidth=2, label='Ground Truth', zorder=1)
            
            # Plot model forecast
            ax.plot(t_plot, pH_model_plot, 'r--', linewidth=2, label=f'{model_name.upper()} Forecast', zorder=2)
            
            # Plot fit forecast if available
            if pH_fit_full is not None:
                pH_fit_plot = pH_fit_full[plot_mask]
                ax.plot(t_plot, pH_fit_plot, 'g:', linewidth=2, label='Mechanistic Fit', zorder=3)
            
            # Vertical line at prefix end
            ax.axvline(prefix_length, color='gray', linestyle='--', alpha=0.5, 
                      label=f'Prefix end ({prefix_length}s)', zorder=4)
            
            ax.set_xlabel('Time [s]', fontsize=12)
            ax.set_ylabel('pH', fontsize=12)
            ax.set_title(f'{model_name.upper()} - Sample {sample_id:04d} - pH Trajectory Forecast (0-{plot_max_t:.0f}s)', fontsize=14)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, plot_max_t)
            
            # Add parameter info
            pred = model_predictions[model_name]
            pred_powder_frac = pred.get("powder_activity_frac", np.nan)
            pred_k_d = pred.get("k_d", np.nan)
            pred_E0 = pred_powder_frac * known_inputs['grams_urease_powder'] / known_inputs['volume_L'] if not np.isnan(pred_powder_frac) else np.nan
            
            param_text = (f"True: frac={true_powder_frac:.3f}, "
                         f"E0={true_E0_g_per_L:.4f} g/L, k_d={true_params.get('k_d', np.nan):.6f} 1/s\n"
                         f"{model_name.upper()}: frac={pred_powder_frac:.3f}, "
                         f"E0={pred_E0:.4f} g/L, k_d={pred_k_d:.6f} 1/s")
            if not np.isnan(fit_powder_frac):
                param_text += f"\nFit: frac={fit_powder_frac:.3f}, "
                param_text += f"E0={fit_E0_g_per_L:.4f} g/L, k_d={fit_params.get('k_d', np.nan):.6f} 1/s"
            
            ax.text(0.02, 0.98, param_text, transform=ax.transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            plt.tight_layout()
            plot_path = model_output_dir / f"sample_{sample_id:04d}_trajectory_plot.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
    
    # Compute metrics for all models
    summary = {
        'sample_id': sample_id,
        'true_powder_activity_frac': true_powder_frac,
        'true_k_d': true_params.get('k_d', np.nan),
        'true_E0_g_per_L': true_E0_g_per_L,
        'fit_powder_activity_frac': fit_powder_frac,
        'fit_k_d': fit_params.get('k_d', np.nan),
        'fit_E0_g_per_L': fit_E0_g_per_L,
    }
    
    # Add model predictions and metrics
    for model_name in models.keys():
        pred = model_predictions[model_name]
        pred_powder_frac = pred["powder_activity_frac"]
        pred_k_d = pred["k_d"]
        pred_E0 = pred_powder_frac * known_inputs['grams_urease_powder'] / known_inputs['volume_L'] if not np.isnan(pred_powder_frac) else np.nan
        
        summary[f'{model_name}_powder_activity_frac'] = pred_powder_frac
        summary[f'{model_name}_k_d'] = pred_k_d
        summary[f'{model_name}_E0_g_per_L'] = pred_E0
        
        # Parameter errors
        if not np.isnan(true_powder_frac):
            summary[f'{model_name}_powder_activity_frac_mae'] = abs(pred_powder_frac - true_powder_frac)
        if not np.isnan(true_params.get('k_d', np.nan)):
            summary[f'{model_name}_k_d_mae'] = abs(pred_k_d - true_params.get('k_d', np.nan))
        
        # Trajectory metrics
        if model_forecasts[model_name] is not None:
            traj_metrics = compute_metrics(pH_true_interp, model_forecasts[model_name])
            summary[f'{model_name}_rmse'] = traj_metrics['RMSE']
            summary[f'{model_name}_mae'] = traj_metrics['MAE']
            summary[f'{model_name}_r2'] = traj_metrics['R²']
        else:
            summary[f'{model_name}_rmse'] = np.nan
            summary[f'{model_name}_mae'] = np.nan
            summary[f'{model_name}_r2'] = np.nan
    
    # Fit parameter errors
    if not np.isnan(true_powder_frac) and not np.isnan(fit_powder_frac):
        summary['fit_powder_activity_frac_mae'] = abs(fit_powder_frac - true_powder_frac)
    if not np.isnan(true_params.get('k_d', np.nan)) and not np.isnan(fit_params.get('k_d', np.nan)):
        summary['fit_k_d_mae'] = abs(fit_params.get('k_d', np.nan) - true_params.get('k_d', np.nan))
    
    # Fit trajectory metrics
    if pH_fit_full is not None:
        traj_metrics_fit = compute_metrics(pH_true_interp, pH_fit_full)
        summary['fit_rmse'] = traj_metrics_fit['RMSE']
        summary['fit_mae'] = traj_metrics_fit['MAE']
        summary['fit_r2'] = traj_metrics_fit['R²']
    else:
        summary['fit_rmse'] = np.nan
        summary['fit_mae'] = np.nan
        summary['fit_r2'] = np.nan
    
    return summary


def main():
    """Main evaluation function."""
    models_dir = Path(CONFIG["models_dir"])
    data_dir = Path(CONFIG["data_dir"])
    output_dir = Path(CONFIG["output_dir"])
    n_test_samples = CONFIG["n_test_samples"]
    prefix_length = CONFIG["prefix_length"]
    t_max = CONFIG["t_max"]
    seed = CONFIG["seed"]
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("TRADITIONAL ML MODELS EVALUATION")
    print("="*80)
    print(f"Models directory: {models_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Prefix length: {prefix_length}s")
    
    # Load models
    print("\nLoading models...")
    models, scaler, training_metadata = load_models(models_dir, prefix_length)
    
    if len(models) == 0:
        raise ValueError("No models found! Train models first using train_traditional_ml.py")
    
    print(f"Loaded {len(models)} models: {list(models.keys())}")
    
    # Load data metadata
    with open(data_dir / "metadata.json", "r") as f:
        data_metadata = json.load(f)
    
    t_max_actual = data_metadata.get('t_max', t_max)
    if t_max_actual != t_max:
        print(f"Using t_max from metadata: {t_max_actual}")
        t_max = t_max_actual
    
    # Generate test cases
    print(f"\nGenerating {n_test_samples} test cases (seed={seed})...")
    test_params_dict = sample_parameters(n_test_samples, seed=seed)
    
    # Build time grid
    if 'time_grid_mode' in data_metadata and 'time_grid_config' in data_metadata:
        time_grid_mode = data_metadata['time_grid_mode']
        time_grid_config = data_metadata['time_grid_config']
        t_grid = build_time_grid(mode=time_grid_mode, t_max=t_max, config=time_grid_config)
    elif 't_grid' in data_metadata:
        t_grid = np.array(data_metadata['t_grid'])
    else:
        n_times = data_metadata.get('n_times', int(t_max / 1.0))
        t_grid = build_time_grid(mode="uniform", t_max=t_max, n_times=n_times)
    
    # Generate trajectories
    test_results_list = generate_trajectories(test_params_dict, t_grid, n_workers=4)
    test_results = [(i, r) for i, r in enumerate(test_results_list) if r is not None]
    print(f"Generated {len(test_results)} test trajectories")
    
    # Evaluate each test case
    print(f"\nEvaluating {len(test_results)} test cases...")
    all_summaries = []
    
    for sample_idx, (original_idx, result) in enumerate(tqdm(test_results[:n_test_samples], desc="Evaluating")):
        pH_true_full = result['pH_true']
        pH_meas_full = result['pH_meas']
        t_full = np.array(result.get('t_grid', t_grid))
        known_inputs = result['known_inputs']
        true_params = result['target_params']
        nuisance_params = result.get('nuisance_params', {})
        
        summary = evaluate_single_case(
            sample_idx, pH_meas_full, pH_true_full, t_full, known_inputs, true_params,
            models, scaler, data_metadata, prefix_length, t_max,
            CONFIG.get("reference_grid_dt", 5.0), output_dir,
            nuisance_params=nuisance_params,
            fit_nuisance_params=CONFIG.get("fit_nuisance_params", False),
        )
        
        if summary is not None:
            all_summaries.append(summary)
    
    print(f"\nEvaluated {len(all_summaries)} test cases")
    
    # Save results
    df_summary = pd.DataFrame(all_summaries)
    aggregate_csv_path = output_dir / "aggregate_metrics.csv"
    df_summary.to_csv(aggregate_csv_path, index=False)
    print(f"✓ Aggregate metrics saved to: {aggregate_csv_path}")
    
    # Print summary statistics
    print("\n" + "="*80)
    print("AGGREGATE METRICS")
    print("="*80)
    
    infer_params = data_metadata.get('infer_params', ['powder_activity_frac', 'k_d'])
    
    print("\nParameter Estimation (MAE):")
    for param_name in infer_params:
        print(f"\n{param_name}:")
        for model_name in models.keys():
            mae_col = f'{model_name}_{param_name}_mae'
            if mae_col in df_summary.columns:
                maes = df_summary[mae_col].dropna()
                if len(maes) > 0:
                    print(f"  {model_name:15s}: {maes.mean():.6f} ± {maes.std():.6f}")
        
        # Fit
        fit_mae_col = f'fit_{param_name}_mae'
        if fit_mae_col in df_summary.columns:
            fit_maes = df_summary[fit_mae_col].dropna()
            if len(fit_maes) > 0:
                print(f"  {'Fit':15s}: {fit_maes.mean():.6f} ± {fit_maes.std():.6f}")
    
    print(f"\nTrajectory Forecasting (Full Horizon 0-{t_max:.0f}s):")
    for model_name in models.keys():
        rmse_col = f'{model_name}_rmse'
        if rmse_col in df_summary.columns:
            rmses = df_summary[rmse_col].dropna()
            r2s = df_summary[f'{model_name}_r2'].dropna()
            if len(rmses) > 0:
                print(f"  {model_name:15s}: RMSE={rmses.mean():.6f} ± {rmses.std():.6f}, R²={r2s.mean():.4f}")
    
    # Fit trajectory
    if 'fit_rmse' in df_summary.columns:
        fit_rmses = df_summary['fit_rmse'].dropna()
        fit_r2s = df_summary['fit_r2'].dropna()
        if len(fit_rmses) > 0:
            print(f"  {'Fit':15s}: RMSE={fit_rmses.mean():.6f} ± {fit_rmses.std():.6f}, R²={fit_r2s.mean():.4f}")
    
    # Save summary JSON
    summary_json = {
        'prefix_length': prefix_length,
        't_max': t_max,
        'n_test_samples': len(all_summaries),
        'seed': seed,
        'models_evaluated': list(models.keys()),
        'parameter_metrics': {
            param: {
                model: {
                    'MAE_mean': float(df_summary[f'{model}_{param}_mae'].mean()),
                    'MAE_std': float(df_summary[f'{model}_{param}_mae'].std()),
                }
                for model in models.keys()
                if f'{model}_{param}_mae' in df_summary.columns
            }
            for param in infer_params
        },
        'trajectory_metrics': {
            model: {
                'RMSE_mean': float(df_summary[f'{model}_rmse'].mean()),
                'RMSE_std': float(df_summary[f'{model}_rmse'].std()),
                'R2_mean': float(df_summary[f'{model}_r2'].mean()),
            }
            for model in models.keys()
            if f'{model}_rmse' in df_summary.columns
        },
    }
    
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(summary_json, f, indent=2)
    
    print(f"\n✓ Evaluation complete!")
    print(f"  Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
