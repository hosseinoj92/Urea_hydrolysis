"""
Evaluate early inference model and compare with mechanistic parameter fitting.

Benchmark comparing:
- ML early inference → predicted parameters → forecast
- Mechanistic-only fitting → estimated parameters → forecast
"""

import torch
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import pearsonr

from forecast_early_inference import load_early_inference_model, normalize_inputs, denormalize_outputs
from fit_mechanistic import fit_mechanistic_parameters
from mechanistic_simulator import UreaseSimulator

# ╔══════════════════════════════════════════════════════════════╗
# ║                       USER CONFIG                             ║
# ╚══════════════════════════════════════════════════════════════╝
CONFIG = {
    # Model and data paths
    "model_path": "models_early_inference/best_model_prefix_30s.pt",
    "data_dir": "Generated_Data_EarlyInference_20000",
    "output_dir": "evaluation_early_inference",
    
    # Evaluation parameters
    "n_test_samples": 50,              # Number of test cases
    "prefix_length": 30.0,              # Length of prefix to reveal [s]
    "forecast_horizons": [300.0, 1000.0, 2000.0],  # Forecast horizons [s]
    "device": "auto",
    
    # Parameter fitting bounds
    "fit_bounds": {
        "activity_scale": (0.1, 2.0),
        "k_d": (0.0, 5e-3),
        "tau_probe": (0.0, 30.0),
        "pH_offset": (-0.1, 0.1),
    },
}


def compute_metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    """Compute RMSE, MAE, R², correlation."""
    mask = np.isfinite(true) & np.isfinite(pred)
    true_masked = true[mask]
    pred_masked = pred[mask]
    
    if len(true_masked) == 0:
        return {"RMSE": np.nan, "MAE": np.nan, "R²": np.nan, "Correlation": np.nan}
    
    rmse = np.sqrt(np.mean((pred_masked - true_masked) ** 2))
    mae = np.mean(np.abs(pred_masked - true_masked))
    
    ss_res = np.sum((pred_masked - true_masked) ** 2)
    ss_tot = np.sum((true_masked - np.mean(true_masked)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else np.nan
    
    corr, _ = pearsonr(pred_masked, true_masked) if len(pred_masked) > 1 else (np.nan, None)
    
    return {
        "RMSE": float(rmse),
        "MAE": float(mae),
        "R²": float(r2),
        "Correlation": float(corr) if not np.isnan(corr) else np.nan,
    }


def time_to_threshold(pH: np.ndarray, t: np.ndarray, threshold: float) -> float:
    """Find time when pH reaches threshold."""
    if len(pH) == 0:
        return np.nan
    
    # Find first time pH >= threshold
    mask = pH >= threshold
    if not np.any(mask):
        return np.nan
    
    idx = np.where(mask)[0][0]
    if idx == 0:
        return t[0]
    
    # Interpolate
    t1, t2 = t[idx-1], t[idx]
    pH1, pH2 = pH[idx-1], pH[idx]
    if pH2 == pH1:
        return t1
    
    w = (threshold - pH1) / (pH2 - pH1)
    return t1 + w * (t2 - t1)


def evaluate_single_case(
    pH_full: np.ndarray,
    t_full: np.ndarray,
    known_inputs: dict,
    true_params: dict,
    model: torch.nn.Module,
    metadata: dict,
    normalization_stats: dict,
    prefix_length: float,
    forecast_horizons: list,
    device: torch.device,
) -> dict:
    """
    Evaluate a single test case.
    
    Returns dict with:
    - ml_params: ML-estimated parameters
    - fit_params: Mechanistic-fit parameters
    - ml_forecasts: dict of ML forecasts for each horizon
    - fit_forecasts: dict of fit forecasts for each horizon
    - metrics: dict of metrics
    """
    # Extract prefix
    mask_prefix = t_full <= prefix_length
    t_prefix = t_full[mask_prefix]
    pH_prefix = pH_full[mask_prefix]
    
    if len(t_prefix) == 0:
        return None
    
    # Prepare known inputs array
    known_input_names = metadata.get('known_input_names', [])
    known_inputs_array = np.array([known_inputs[name] for name in known_input_names])
    
    # ML parameter estimation
    pH_seq_tensor, known_inputs_tensor = normalize_inputs(
        pH_prefix, known_inputs_array, normalization_stats
    )
    
    with torch.no_grad():
        pH_seq_tensor = pH_seq_tensor.to(device)
        known_inputs_tensor = known_inputs_tensor.to(device)
        mean, _ = model(pH_seq_tensor, known_inputs_tensor)
        params_norm = mean.cpu().numpy().squeeze()
    
    ml_params = {name: float(val) for name, val in 
                 zip(metadata['infer_params'], denormalize_outputs(params_norm, normalization_stats))}
    
    # Mechanistic parameter fitting
    try:
        fit_params = fit_mechanistic_parameters(
            pH_prefix, t_prefix, known_inputs,
            param_bounds=CONFIG["fit_bounds"]
        )
    except Exception as e:
        print(f"Warning: Fitting failed: {e}")
        fit_params = {k: 0.0 for k in CONFIG["fit_bounds"].keys()}
    
    # Build simulator
    S0 = known_inputs['substrate_mM'] / 1000.0
    T_K = known_inputs['temperature_C'] + 273.15
    E_loading_base_g_per_L = (known_inputs['grams_urease_powder'] * 
                             known_inputs['powder_activity_frac'] / 
                             known_inputs['volume_L'])
    
    sim = UreaseSimulator(
        S0=S0,
        N0=0.0,
        C0=0.0,
        Pt_total_M=0.0,
        T_K=T_K,
        initial_pH=known_inputs['initial_pH'],
        E_loading_base_g_per_L=E_loading_base_g_per_L,
        use_T_dependent_pH_activity=True,
    )
    
    # Forecasts for each horizon
    ml_forecasts = {}
    fit_forecasts = {}
    
    for horizon in forecast_horizons:
        # Time points for forecast (from prefix end to horizon)
        t_forecast = np.linspace(prefix_length, prefix_length + horizon, int(horizon / 2))
        
        # ML forecast
        sim_params_ml = {
            'a': ml_params.get('activity_scale', 1.0),
            'k_d': ml_params.get('k_d', 0.0),
            't_shift': 0.0,
            'tau_probe': ml_params.get('tau_probe', 0.0),
        }
        pH_ml = sim.simulate_forward(sim_params_ml, t_forecast, return_totals=False, apply_probe_lag=False)
        ml_forecasts[horizon] = {'t': t_forecast, 'pH': pH_ml}
        
        # Fit forecast
        sim_params_fit = {
            'a': fit_params.get('activity_scale', 1.0),
            'k_d': fit_params.get('k_d', 0.0),
            't_shift': 0.0,
            'tau_probe': fit_params.get('tau_probe', 0.0),
        }
        pH_fit = sim.simulate_forward(sim_params_fit, t_forecast, return_totals=False, apply_probe_lag=False)
        fit_forecasts[horizon] = {'t': t_forecast, 'pH': pH_fit}
    
    # Extract true future trajectory
    true_forecasts = {}
    for horizon in forecast_horizons:
        t_end = prefix_length + horizon
        mask = (t_full > prefix_length) & (t_full <= t_end)
        if np.any(mask):
            true_forecasts[horizon] = {
                't': t_full[mask],
                'pH': pH_full[mask],
            }
    
    # Compute metrics
    metrics = {}
    
    # Parameter metrics
    infer_params = metadata['infer_params']
    param_metrics_ml = {}
    param_metrics_fit = {}
    for param_name in infer_params:
        if param_name in true_params:
            true_val = true_params[param_name]
            ml_val = ml_params.get(param_name, 0.0)
            fit_val = fit_params.get(param_name, 0.0)
            
            param_metrics_ml[param_name] = {
                'MAE': float(np.abs(ml_val - true_val)),
                'RMSE': float(np.sqrt((ml_val - true_val) ** 2)),
                'RelativeError': float(np.abs(ml_val - true_val) / (np.abs(true_val) + 1e-8)),
            }
            param_metrics_fit[param_name] = {
                'MAE': float(np.abs(fit_val - true_val)),
                'RMSE': float(np.sqrt((fit_val - true_val) ** 2)),
                'RelativeError': float(np.abs(fit_val - true_val) / (np.abs(true_val) + 1e-8)),
            }
    
    metrics['parameters'] = {
        'ML': param_metrics_ml,
        'Fit': param_metrics_fit,
    }
    
    # Trajectory metrics
    trajectory_metrics = {}
    for horizon in forecast_horizons:
        if horizon not in true_forecasts:
            continue
        
        true_t = true_forecasts[horizon]['t']
        true_pH = true_forecasts[horizon]['pH']
        
        # Interpolate ML and fit forecasts to true time points
        ml_t = ml_forecasts[horizon]['t']
        ml_pH = ml_forecasts[horizon]['pH']
        fit_t = fit_forecasts[horizon]['t']
        fit_pH = fit_forecasts[horizon]['pH']
        
        from scipy.interpolate import interp1d
        ml_interp = interp1d(ml_t, ml_pH, kind='linear', fill_value='extrapolate')
        fit_interp = interp1d(fit_t, fit_pH, kind='linear', fill_value='extrapolate')
        
        ml_pH_interp = ml_interp(true_t)
        fit_pH_interp = fit_interp(true_t)
        
        trajectory_metrics[horizon] = {
            'ML': compute_metrics(true_pH, ml_pH_interp),
            'Fit': compute_metrics(true_pH, fit_pH_interp),
        }
        
        # Time to threshold (e.g., pH = 8.0)
        threshold = 8.0
        true_time_to_thresh = time_to_threshold(true_pH, true_t, threshold)
        ml_time_to_thresh = time_to_threshold(ml_pH_interp, true_t, threshold)
        fit_time_to_thresh = time_to_threshold(fit_pH_interp, true_t, threshold)
        
        if not np.isnan(true_time_to_thresh):
            trajectory_metrics[horizon]['TimeToThreshold'] = {
                'True': float(true_time_to_thresh),
                'ML': float(ml_time_to_thresh) if not np.isnan(ml_time_to_thresh) else np.nan,
                'Fit': float(fit_time_to_thresh) if not np.isnan(fit_time_to_thresh) else np.nan,
            }
    
    metrics['trajectory'] = trajectory_metrics
    
    return {
        'ml_params': ml_params,
        'fit_params': fit_params,
        'true_params': true_params,
        'ml_forecasts': ml_forecasts,
        'fit_forecasts': fit_forecasts,
        'true_forecasts': true_forecasts,
        'metrics': metrics,
    }


def main():
    # Use CONFIG
    model_path = Path(CONFIG["model_path"])
    data_dir = Path(CONFIG["data_dir"])
    output_dir = Path(CONFIG["output_dir"])
    n_test_samples = CONFIG["n_test_samples"]
    prefix_length = CONFIG["prefix_length"]
    
    # Device
    if CONFIG["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(CONFIG["device"])
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model, metadata, normalization_stats, model_prefix_length = load_early_inference_model(model_path, device)
    print("✓ Model loaded")
    
    # Load test data
    data_file = data_dir / "training_data.npz"
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    print(f"Loading test data from {data_file}...")
    data = np.load(data_file, allow_pickle=True)
    
    # We need full trajectories for evaluation, so we'll generate them from the original data generator
    # For now, use a subset of the training data as test
    # In practice, you'd want a separate test set with full trajectories
    
    # Load metadata to understand data structure
    with open(data_dir / "metadata.json", "r") as f:
        data_metadata = json.load(f)
    
    print(f"\nEvaluating on {n_test_samples} test cases...")
    print(f"Prefix length: {prefix_length}s")
    print(f"Forecast horizons: {CONFIG['forecast_horizons']}s")
    
    # For evaluation, we need to generate test cases with full trajectories
    # This is a simplified version - in practice, you'd load a separate test set
    print("\nNote: This evaluation uses synthetic test cases.")
    print("For real evaluation, generate a separate test set with full trajectories.")
    
    # Generate test cases (simplified - using training data structure)
    # In practice, you'd want a separate test generator
    results = []
    
    # Use a subset of training data as test (for demonstration)
    # In practice, generate a separate test set
    print("\nGenerating test cases...")
    from generate_early_inference_data import sample_parameters
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    
    # Sample test parameters
    test_params_dict = sample_parameters(n_test_samples, seed=12345)
    
    # Generate full trajectories for test
    t_max = 2000.0
    n_times = 2000
    t_grid = np.linspace(0.0, t_max, n_times)
    
    # Generate trajectories using the same worker function
    from generate_early_inference_data import _worker_generate_single_trajectory
    
    test_results = []
    for i in tqdm(range(n_test_samples), desc="Generating test trajectories"):
        params_row = {k: v[i] for k, v in test_params_dict.items()}
        try:
            _, result, _ = _worker_generate_single_trajectory(
                (i, params_row, t_grid, (), ())
            )
            if result is not None:
                test_results.append((params_row, result))
        except Exception as e:
            print(f"Warning: Failed to generate trajectory {i}: {e}")
            continue
    
    print(f"Generated {len(test_results)} test trajectories")
    
    # Evaluate each test case
    all_results = []
    for params_dict, result in tqdm(test_results[:n_test_samples], desc="Evaluating"):
        # Extract full trajectory
        pH_full = result['pH_meas']
        t_full = t_grid
        
        # Known inputs
        known_inputs = result['known_inputs']
        
        # True parameters
        true_params = result['target_params']
        
        # Evaluate
        eval_result = evaluate_single_case(
            pH_full, t_full, known_inputs, true_params,
            model, metadata, normalization_stats, prefix_length,
            CONFIG["forecast_horizons"], device
        )
        
        if eval_result is not None:
            all_results.append(eval_result)
    
    print(f"\nEvaluated {len(all_results)} test cases")
    
    # Aggregate metrics
    print("\n" + "="*60)
    print("AGGREGATE METRICS")
    print("="*60)
    
    # Parameter metrics
    infer_params = metadata['infer_params']
    param_metrics_agg = {}
    for param_name in infer_params:
        ml_maes = [r['metrics']['parameters']['ML'][param_name]['MAE'] 
                   for r in all_results if param_name in r['metrics']['parameters']['ML']]
        fit_maes = [r['metrics']['parameters']['Fit'][param_name]['MAE'] 
                    for r in all_results if param_name in r['metrics']['parameters']['Fit']]
        
        if ml_maes:
            param_metrics_agg[param_name] = {
                'ML_MAE_mean': float(np.mean(ml_maes)),
                'ML_MAE_std': float(np.std(ml_maes)),
                'Fit_MAE_mean': float(np.mean(fit_maes)),
                'Fit_MAE_std': float(np.std(fit_maes)),
            }
            print(f"\n{param_name}:")
            print(f"  ML MAE: {np.mean(ml_maes):.6f} ± {np.std(ml_maes):.6f}")
            print(f"  Fit MAE: {np.mean(fit_maes):.6f} ± {np.std(fit_maes):.6f}")
    
    # Trajectory metrics
    trajectory_metrics_agg = {}
    for horizon in CONFIG["forecast_horizons"]:
        ml_rmses = []
        fit_rmses = []
        ml_r2s = []
        fit_r2s = []
        
        for r in all_results:
            if horizon in r['metrics']['trajectory']:
                ml_rmses.append(r['metrics']['trajectory'][horizon]['ML']['RMSE'])
                fit_rmses.append(r['metrics']['trajectory'][horizon]['Fit']['RMSE'])
                ml_r2s.append(r['metrics']['trajectory'][horizon]['ML']['R²'])
                fit_r2s.append(r['metrics']['trajectory'][horizon]['Fit']['R²'])
        
        if ml_rmses:
            trajectory_metrics_agg[horizon] = {
                'ML_RMSE_mean': float(np.mean(ml_rmses)),
                'ML_RMSE_std': float(np.std(ml_rmses)),
                'Fit_RMSE_mean': float(np.mean(fit_rmses)),
                'Fit_RMSE_std': float(np.std(fit_rmses)),
                'ML_R2_mean': float(np.mean(ml_r2s)),
                'Fit_R2_mean': float(np.mean(fit_r2s)),
            }
            print(f"\nForecast horizon {horizon}s:")
            print(f"  ML RMSE: {np.mean(ml_rmses):.6f} ± {np.std(ml_rmses):.6f}")
            print(f"  Fit RMSE: {np.mean(fit_rmses):.6f} ± {np.std(fit_rmses):.6f}")
            print(f"  ML R²: {np.mean(ml_r2s):.4f}")
            print(f"  Fit R²: {np.mean(fit_r2s):.4f}")
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    metrics_summary = {
        'prefix_length': prefix_length,
        'forecast_horizons': CONFIG['forecast_horizons'],
        'n_test_samples': len(all_results),
        'parameter_metrics': param_metrics_agg,
        'trajectory_metrics': trajectory_metrics_agg,
    }
    
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)
    
    # Plot sample predictions
    n_plot = min(5, len(all_results))
    fig, axes = plt.subplots(n_plot, 1, figsize=(12, 3*n_plot))
    if n_plot == 1:
        axes = [axes]
    
    for i in range(n_plot):
        ax = axes[i]
        r = all_results[i]
        
        # Plot prefix (we need to get this from the original test data)
        # For now, plot from the first forecast horizon's true data
        if CONFIG['forecast_horizons'][0] in r['true_forecasts']:
            tf = r['true_forecasts'][CONFIG['forecast_horizons'][0]]
            prefix_mask = tf['t'] <= prefix_length
            if np.any(prefix_mask):
                ax.plot(tf['t'][prefix_mask], tf['pH'][prefix_mask], 
                        'ko', markersize=4, label='Prefix (observed)', alpha=0.6)
        
        # Plot true future
        for horizon in CONFIG['forecast_horizons']:
            if horizon in r['true_forecasts']:
                tf = r['true_forecasts'][horizon]
                ax.plot(tf['t'], tf['pH'], 'b-', linewidth=2, label=f'True (horizon {horizon}s)')
        
        # Plot ML forecast
        for horizon in CONFIG['forecast_horizons']:
            if horizon in r['ml_forecasts']:
                mf = r['ml_forecasts'][horizon]
                ax.plot(mf['t'], mf['pH'], 'r--', linewidth=2, label=f'ML forecast ({horizon}s)')
        
        # Plot fit forecast
        for horizon in CONFIG['forecast_horizons']:
            if horizon in r['fit_forecasts']:
                ff = r['fit_forecasts'][horizon]
                ax.plot(ff['t'], ff['pH'], 'g:', linewidth=2, label=f'Fit forecast ({horizon}s)')
        
        ax.axvline(prefix_length, color='gray', linestyle='--', alpha=0.5, label='Prefix end')
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('pH')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Test Case {i+1}')
    
    plt.tight_layout()
    plt.savefig(output_dir / "sample_predictions.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Evaluation complete!")
    print(f"  Results saved to: {output_dir}")
    print(f"  Metrics: {output_dir / 'metrics.json'}")
    print(f"  Plots: {output_dir / 'sample_predictions.png'}")


if __name__ == "__main__":
    main()
