"""
Evaluate early inference model and compare with mechanistic parameter fitting.

Unified parameterization: E0_g_per_L + k_d only (no activity_scale, powder_activity_frac, tau_probe, pH_offset).
"""

import torch
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import pearsonr
from scipy.interpolate import interp1d

from forecast_early_inference import load_early_inference_model, normalize_inputs, denormalize_outputs
from fit_mechanistic import fit_mechanistic_parameters
from generate_early_inference_data import extract_prefix, sample_parameters, build_time_grid
from mechanistic_simulator import UreaseSimulator

# ╔══════════════════════════════════════════════════════════════╗
# ║                       USER CONFIG                             ║
# ╚══════════════════════════════════════════════════════════════╝
CONFIG = {
    # Model and data paths
    "model_path": "models_early_inference_30s/best_model_prefix_30s.pt",
    "data_dir": "Generated_Data_EarlyInference_20000",
    "output_dir": "evaluation_early_inference_30s",
    
    # Evaluation parameters
    "n_test_samples": 100,              # Number of test cases
    "prefix_length": 30.0,              # Length of prefix to reveal [s]
    "t_max": 2000.0,                    # Full trajectory length [s] (should match data generation)
    "reference_grid_dt": 5.0,           # Reference grid spacing [s] for fair comparison (uniform)
    "device": "auto",
    "seed": 12345,                       # Deterministic seed for test set
    
    # Parameter fitting bounds (unified: E0_g_per_L and k_d only)
    "fit_bounds": {
        "E0_g_per_L": (5e-4, 1.25),  # Wide range covering slow to fast regimes [g/L]
        "k_d": (0.0, 5e-3),
    },
}


def compute_metrics(y_true, y_pred):
    """Compute regression metrics."""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # R²
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-12))
    
    # Correlation
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
    
    # Find first time pH >= threshold
    mask = pH >= threshold
    if not np.any(mask):
        return np.nan
    
    idx = np.argmax(mask)
    if idx == 0:
        return t[0]
    
    # Interpolate
    t1, t2 = t[idx-1], t[idx]
    pH1, pH2 = pH[idx-1], pH[idx]
    
    if pH2 == pH1:
        return t1
    
    t_thresh = t1 + (threshold - pH1) * (t2 - t1) / (pH2 - pH1)
    return float(t_thresh)


def evaluate_single_case(
    sample_id: int,
    pH_meas_full: np.ndarray,  # Measured pH (with noise) - what model sees
    pH_true_full: np.ndarray,  # True pH (no noise) - ground truth
    t_full: np.ndarray,  # Non-uniform time grid from simulation
    known_inputs: dict,
    true_params: dict,
    model,
    metadata: dict,
    normalization_stats: dict,
    prefix_length: float,
    t_max: float,
    reference_grid_dt: float,  # Uniform spacing for reference grid
    device: torch.device,
    output_dir: Path,
) -> dict:
    """
    Evaluate a single test case and write per-sample CSV.
    
    Unified parameterization: E0_g_per_L + k_d only.
    
    Parameters
    ----------
    pH_meas_full: Measured pH with noise (used for prefix - what model sees)
    pH_true_full: True pH without noise (used for ground truth comparison)
    
    Returns dict with metrics.
    """
    # Extract prefix and interpolate onto uniform grid (same as training)
    prefix_n_points = metadata.get('prefix_n_points', 100)
    t_prefix_uniform, pH_prefix = extract_prefix(
        pH_meas_full, t_full, prefix_length, prefix_n_points
    )
    
    if len(t_prefix_uniform) == 0:
        return None
    
    # For mechanistic fitting, use original non-uniform grid (better for optimization)
    mask_prefix = t_full <= prefix_length
    t_prefix_original = t_full[mask_prefix]
    pH_prefix_original = pH_meas_full[mask_prefix]
    
    # Prepare known inputs array (unified: 5 inputs, no powder_activity_frac)
    known_input_names = metadata.get('known_input_names', [])
    known_inputs_array = np.array([known_inputs[name] for name in known_input_names])
    
    # ML parameter estimation (uses uniform prefix grid)
    pH_seq_tensor, known_inputs_tensor = normalize_inputs(
        pH_prefix, known_inputs_array, normalization_stats
    )
    
    with torch.no_grad():
        # normalize_inputs already adds batch dimension, so just move to device
        pH_seq_tensor = pH_seq_tensor.to(device)
        known_inputs_tensor = known_inputs_tensor.to(device)
        mean, _ = model(pH_seq_tensor, known_inputs_tensor)
        params_norm = mean.cpu().numpy().squeeze()
    
    ml_params = {name: float(val) for name, val in 
                 zip(metadata['infer_params'], denormalize_outputs(params_norm, normalization_stats))}
    
    # Clamp ML predictions to valid bounds (prevents invalid negative/extreme values)
    ml_params['E0_g_per_L'] = float(np.clip(ml_params.get('E0_g_per_L', 0.5), 5e-4, 1.25))
    ml_params['k_d'] = float(np.clip(ml_params.get('k_d', 0.0), 0.0, 5e-3))
    
    # Compute derived powder_activity_frac for ML (for reporting only, not used in simulation)
    ml_E0 = ml_params.get('E0_g_per_L', np.nan)
    if not np.isnan(ml_E0) and ml_E0 > 0:
        ml_powder_activity_frac_derived = float(np.clip(
            ml_E0 * known_inputs['volume_L'] / known_inputs['grams_urease_powder'],
            0.0, 1.0
        ))
    else:
        ml_powder_activity_frac_derived = np.nan
    
    # Mechanistic parameter fitting (uses original non-uniform grid for better optimization)
    try:
        fit_params = fit_mechanistic_parameters(
            pH_prefix_original, t_prefix_original, known_inputs,
            param_bounds=CONFIG["fit_bounds"]
        )
    except Exception as e:
        print(f"Warning: Fitting failed for sample {sample_id}: {e}")
        fit_params = {k: np.nan for k in CONFIG["fit_bounds"].keys()}
    
    # Compute derived powder_activity_frac for Fit (for reporting only)
    fit_E0 = fit_params.get('E0_g_per_L', np.nan)
    if not np.isnan(fit_E0) and fit_E0 > 0:
        fit_powder_activity_frac_derived = float(np.clip(
            fit_E0 * known_inputs['volume_L'] / known_inputs['grams_urease_powder'],
            0.0, 1.0
        ))
    else:
        fit_powder_activity_frac_derived = np.nan
    
    # Build simulator (unified: use E_eff0 directly, no powder_activity_frac)
    S0 = known_inputs['substrate_mM'] / 1000.0
    T_K = known_inputs['temperature_C'] + 273.15
    
    sim = UreaseSimulator(
        S0=S0,
        N0=0.0,
        C0=0.0,
        Pt_total_M=0.0,
        T_K=T_K,
        initial_pH=known_inputs['initial_pH'],
        E_loading_base_g_per_L=1.0,  # Dummy value, overridden by E_eff0
        use_T_dependent_pH_activity=True,
    )
    
    # Create reference grid for fair comparison (uniform, doesn't bias toward early times)
    t_ref = np.arange(0.0, t_max + reference_grid_dt/2, reference_grid_dt)
    t_ref[-1] = t_max  # Ensure t_max is exactly included
    
    # ML forecast (full trajectory on reference grid) - use E_eff0 directly
    sim_params_ml = {
        'E_eff0': ml_params.get('E0_g_per_L', 0.5),  # Direct enzyme loading [g/L]
        'k_d': ml_params.get('k_d', 0.0),
        't_shift': 0.0,
        'tau_probe': 0.0,  # Not used (true pH space)
    }
    pH_ml_full = sim.simulate_forward(sim_params_ml, t_ref, return_totals=False, apply_probe_lag=False)
    
    # Fit forecast (full trajectory on reference grid) - use E_eff0 directly
    sim_params_fit = {
        'E_eff0': fit_params.get('E0_g_per_L', 0.5),  # Direct enzyme loading [g/L]
        'k_d': fit_params.get('k_d', 0.0),
        't_shift': 0.0,
        'tau_probe': 0.0,  # Not used (true pH space)
    }
    pH_fit_full = sim.simulate_forward(sim_params_fit, t_ref, return_totals=False, apply_probe_lag=False)
    
    # Interpolate ground truth to reference grid
    interp_func = interp1d(t_full, pH_true_full, kind='linear', bounds_error=False, fill_value='extrapolate')
    pH_true_interp = interp_func(t_ref)
    
    # Plot sample trajectory (first 1000 seconds)
    plot_max_t = 1000.0
    plot_mask = t_ref <= plot_max_t
    t_plot = t_ref[plot_mask]
    pH_true_plot = pH_true_interp[plot_mask]
    pH_ml_plot = pH_ml_full[plot_mask]
    pH_fit_plot = pH_fit_full[plot_mask]
    
    # Interpolate prefix for plotting
    prefix_interp = interp1d(t_prefix_uniform, pH_prefix, kind='linear', 
                            bounds_error=False, fill_value='extrapolate')
    pH_prefix_plot = prefix_interp(t_plot)
    prefix_mask_plot = t_plot <= prefix_length
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot prefix (observed data used for inference)
    ax.plot(t_plot[prefix_mask_plot], pH_prefix_plot[prefix_mask_plot], 
            'ko', markersize=4, label='Prefix (observed)', alpha=0.6, zorder=5)
    
    # Plot ground truth
    ax.plot(t_plot, pH_true_plot, 'b-', linewidth=2, label='Ground Truth', zorder=1)
    
    # Plot ML forecast
    ax.plot(t_plot, pH_ml_plot, 'r--', linewidth=2, label='ML Forecast', zorder=2)
    
    # Plot mechanistic fit forecast
    ax.plot(t_plot, pH_fit_plot, 'g:', linewidth=2, label='Mechanistic Fit', zorder=3)
    
    # Vertical line at prefix end
    ax.axvline(prefix_length, color='gray', linestyle='--', alpha=0.5, 
               label=f'Prefix end ({prefix_length}s)', zorder=4)
    
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('pH', fontsize=12)
    ax.set_title(f'Sample {sample_id:04d} - pH Trajectory Forecast (0-{plot_max_t:.0f}s)', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, plot_max_t)
    
    # Add parameter info as text
    param_text = (f"True: E0={true_params.get('E0_g_per_L', np.nan):.4f} g/L, "
                  f"k_d={true_params.get('k_d', np.nan):.6f} 1/s\n"
                  f"ML: E0={ml_params.get('E0_g_per_L', np.nan):.4f} g/L, "
                  f"k_d={ml_params.get('k_d', np.nan):.6f} 1/s\n"
                  f"Fit: E0={fit_params.get('E0_g_per_L', np.nan):.4f} g/L, "
                  f"k_d={fit_params.get('k_d', np.nan):.6f} 1/s")
    ax.text(0.02, 0.98, param_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plot_path = output_dir / f"sample_{sample_id:04d}_trajectory_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Compute metrics
    metrics = {}
    
    # Parameter metrics
    param_metrics_ml = {}
    param_metrics_fit = {}
    
    for param_name in metadata['infer_params']:
        if param_name in true_params:
            true_val = true_params[param_name]
            ml_val = ml_params.get(param_name, np.nan)
            fit_val = fit_params.get(param_name, np.nan)
            
            param_metrics_ml[param_name] = {
                'MAE': float(np.abs(ml_val - true_val)) if not np.isnan(ml_val) else np.nan,
                'RMSE': float(np.sqrt((ml_val - true_val) ** 2)) if not np.isnan(ml_val) else np.nan,
                'RelativeError': float(np.abs(ml_val - true_val) / (np.abs(true_val) + 1e-8)) if not np.isnan(ml_val) else np.nan,
            }
            param_metrics_fit[param_name] = {
                'MAE': float(np.abs(fit_val - true_val)) if not np.isnan(fit_val) else np.nan,
                'RMSE': float(np.sqrt((fit_val - true_val) ** 2)) if not np.isnan(fit_val) else np.nan,
                'RelativeError': float(np.abs(fit_val - true_val) / (np.abs(true_val) + 1e-8)) if not np.isnan(fit_val) else np.nan,
            }
    
    metrics['parameters'] = {
        'ML': param_metrics_ml,
        'Fit': param_metrics_fit,
    }
    
    # Trajectory metrics (full horizon)
    trajectory_metrics = compute_metrics(pH_true_interp, pH_ml_full)
    trajectory_metrics_fit = compute_metrics(pH_true_interp, pH_fit_full)
    
    metrics['trajectory'] = {
        'ML': trajectory_metrics,
        'Fit': trajectory_metrics_fit,
    }
    
    # Time to threshold (on reference grid)
    threshold = 8.0
    true_time_to_thresh = time_to_threshold(pH_true_interp, t_ref, threshold)
    ml_time_to_thresh = time_to_threshold(pH_ml_full, t_ref, threshold)
    fit_time_to_thresh = time_to_threshold(pH_fit_full, t_ref, threshold)
    
    if not np.isnan(true_time_to_thresh):
        metrics['time_to_threshold'] = {
            'True': float(true_time_to_thresh),
            'ML': float(ml_time_to_thresh) if not np.isnan(ml_time_to_thresh) else np.nan,
            'Fit': float(fit_time_to_thresh) if not np.isnan(fit_time_to_thresh) else np.nan,
        }
    
    # Write per-sample CSV (on reference grid)
    import csv
    sample_csv_path = output_dir / f"sample_{sample_id:04d}_trajectories.csv"
    
    # Interpolate prefix data for CSV (reuse prefix_interp from plotting)
    pH_prefix_on_ref = prefix_interp(t_ref)
    
    with open(sample_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time_s', 'pH_ground_truth', 'pH_prefix', 'pH_ml_forecast', 'pH_fit_forecast', 'is_prefix'])
        
        for i, t in enumerate(t_ref):
            is_prefix = t <= prefix_length
            
            writer.writerow([
                f"{t:.6f}",
                f"{pH_true_interp[i]:.6f}",
                f"{pH_prefix_on_ref[i]:.6f}" if is_prefix else "",
                f"{pH_ml_full[i]:.6f}",
                f"{pH_fit_full[i]:.6f}",
                "1" if is_prefix else "0",
            ])
    
    # Compute ground truth derived powder_activity_frac from true E0_g_per_L
    true_E0 = true_params.get('E0_g_per_L', np.nan)
    true_powder_activity_frac_derived = np.nan
    if not np.isnan(true_E0):
        true_powder_activity_frac_derived = float(np.clip(
            true_E0 * known_inputs['volume_L'] / known_inputs['grams_urease_powder'],
            0.0, 1.0
        ))
    
    # Return summary for aggregate CSV
    summary = {
        'sample_id': sample_id,
        # Inferred parameters (unified: E0_g_per_L and k_d only)
        'ml_E0_g_per_L': ml_params.get('E0_g_per_L', np.nan),
        'ml_k_d': ml_params.get('k_d', np.nan),
        'fit_E0_g_per_L': fit_params.get('E0_g_per_L', np.nan),
        'fit_k_d': fit_params.get('k_d', np.nan),
        'true_E0_g_per_L': true_params.get('E0_g_per_L', np.nan),
        'true_k_d': true_params.get('k_d', np.nan),
        # Parameter errors
        'ml_E0_g_per_L_mae': param_metrics_ml.get('E0_g_per_L', {}).get('MAE', np.nan),
        'ml_k_d_mae': param_metrics_ml.get('k_d', {}).get('MAE', np.nan),
        'fit_E0_g_per_L_mae': param_metrics_fit.get('E0_g_per_L', {}).get('MAE', np.nan),
        'fit_k_d_mae': param_metrics_fit.get('k_d', {}).get('MAE', np.nan),
        # Derived powder_activity_frac (for interpretability, not used in inference)
        'ml_powder_activity_frac_derived': ml_powder_activity_frac_derived,
        'fit_powder_activity_frac_derived': fit_powder_activity_frac_derived,
        'true_powder_activity_frac_derived': true_powder_activity_frac_derived,
        # Trajectory metrics
        'ml_rmse': trajectory_metrics.get('RMSE', np.nan),
        'ml_mae': trajectory_metrics.get('MAE', np.nan),
        'ml_r2': trajectory_metrics.get('R²', np.nan),
        'fit_rmse': trajectory_metrics_fit.get('RMSE', np.nan),
        'fit_mae': trajectory_metrics_fit.get('MAE', np.nan),
        'fit_r2': trajectory_metrics_fit.get('R²', np.nan),
    }
    
    if 'time_to_threshold' in metrics:
        summary['true_time_to_thresh'] = metrics['time_to_threshold']['True']
        summary['ml_time_to_thresh'] = metrics['time_to_threshold']['ML']
        summary['fit_time_to_thresh'] = metrics['time_to_threshold']['Fit']
    
    return summary


def main():
    # Use CONFIG
    model_path = Path(CONFIG["model_path"])
    data_dir = Path(CONFIG["data_dir"])
    output_dir = Path(CONFIG["output_dir"])
    n_test_samples = CONFIG["n_test_samples"]
    prefix_length = CONFIG["prefix_length"]
    t_max = CONFIG["t_max"]
    seed = CONFIG["seed"]
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    if CONFIG["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(CONFIG["device"])
    
    print(f"Using device: {device}")
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    model, metadata, normalization_stats, prefix_length_model = load_early_inference_model(model_path, device)
    
    print(f"Model loaded successfully")
    print(f"  Inferred parameters: {metadata['infer_params']}")
    print(f"  Known inputs: {metadata.get('known_input_names', [])}")
    print(f"  Prefix length: {prefix_length_model}s")
    
    # Load data directory
    data_file = data_dir / "training_data.npz"
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    data = np.load(data_file, allow_pickle=True)
    
    # Load metadata to understand data structure
    with open(data_dir / "metadata.json", "r") as f:
        data_metadata = json.load(f)
    
    # Use t_max from metadata if available (should match training data)
    t_max_actual = data_metadata.get('t_max', t_max)
    if t_max_actual != t_max:
        print(f"Using t_max from metadata: {t_max_actual} (config had {t_max})")
        t_max = t_max_actual
    
    # Generate test cases with deterministic seed
    print(f"\nGenerating {n_test_samples} test cases (seed={seed})...")
    
    test_params_dict = sample_parameters(n_test_samples, seed=seed)
    
    # Use same time grid as training data (from metadata if available, else build)
    if 'time_grid_mode' in data_metadata and 'time_grid_config' in data_metadata:
        time_grid_mode = data_metadata['time_grid_mode']
        time_grid_config = data_metadata['time_grid_config']
        t_grid = build_time_grid(mode=time_grid_mode, t_max=t_max, config=time_grid_config)
        print(f"Using time grid from metadata: {len(t_grid)} points ({time_grid_mode} mode)")
    elif 't_grid' in data_metadata:
        # Backwards compatibility: use stored t_grid directly
        t_grid = np.array(data_metadata['t_grid'])
        print(f"Using stored time grid from metadata: {len(t_grid)} points (uniform grid)")
    else:
        # Fallback: build uniform grid with reasonable default
        n_times = data_metadata.get('n_times', int(t_max / 1.0))
        t_grid = build_time_grid(mode="uniform", t_max=t_max, n_times=n_times)
        print(f"Using default uniform time grid: {len(t_grid)} points (dt={t_max/(n_times-1):.1f}s)")
    
    # Generate trajectories
    from generate_early_inference_data import generate_trajectories
    test_results_list = generate_trajectories(test_params_dict, t_grid, n_workers=4)
    
    # Filter successful results and pair with indices
    test_results = [(i, r) for i, r in enumerate(test_results_list) if r is not None]
    print(f"Generated {len(test_results)} test trajectories")
    
    # Save test sample IDs
    test_sample_ids = [i for i, _ in test_results]
    with open(output_dir / "test_sample_ids.json", 'w') as f:
        json.dump({
            'seed': seed,
            'n_samples': len(test_sample_ids),
            'sample_ids': test_sample_ids,
            'prefix_length': prefix_length,
            't_max': t_max,
            'reference_grid_dt': CONFIG.get("reference_grid_dt", 5.0),
            'time_grid_mode': data_metadata.get('time_grid_mode', 'uniform'),  # Fallback to uniform for old data
        }, f, indent=2)
    
    # Evaluate each test case
    print(f"\nEvaluating {len(test_results)} test cases...")
    all_summaries = []
    
    for sample_idx, (original_idx, result) in enumerate(tqdm(test_results[:n_test_samples], desc="Evaluating")):
        # Extract full trajectory
        pH_true_full = result['pH_true']  # True pH (no noise) - ground truth
        pH_meas_full = result['pH_meas']  # Measured pH (with noise) - what model sees
        
        # Use stored time grid from result (each trajectory stores its own grid)
        # Fallback to default grid if not stored (for backwards compatibility)
        t_full = np.array(result.get('t_grid', t_grid))
        
        # Known inputs
        known_inputs = result['known_inputs']
        
        # True parameters
        true_params = result['target_params']
        
        # Evaluate
        summary = evaluate_single_case(
            sample_idx,
            pH_meas_full,
            pH_true_full,
            t_full, known_inputs, true_params,
            model, metadata, normalization_stats, prefix_length,
            t_max, CONFIG.get("reference_grid_dt", 5.0), device, output_dir
        )
        
        if summary is not None:
            all_summaries.append(summary)
    
    print(f"\nEvaluated {len(all_summaries)} test cases")
    
    # Write aggregate CSV
    import pandas as pd
    df_summary = pd.DataFrame(all_summaries)
    aggregate_csv_path = output_dir / "aggregate_metrics.csv"
    df_summary.to_csv(aggregate_csv_path, index=False)
    print(f"✓ Aggregate metrics saved to: {aggregate_csv_path}")
    
    # Compute aggregate statistics
    print("\n" + "="*60)
    print("AGGREGATE METRICS (Unified: E0_g_per_L + k_d)")
    print("="*60)
    
    # Parameter metrics
    infer_params = metadata['infer_params']
    print("\nParameter Estimation (MAE):")
    for param_name in infer_params:
        ml_maes = df_summary[f'ml_{param_name}_mae'].dropna()
        fit_maes = df_summary[f'fit_{param_name}_mae'].dropna()
        
        if len(ml_maes) > 0:
            print(f"\n{param_name}:")
            print(f"  ML:   {ml_maes.mean():.6f} ± {ml_maes.std():.6f}")
            print(f"  Fit:  {fit_maes.mean():.6f} ± {fit_maes.std():.6f}")
    
    # Trajectory metrics
    print(f"\nTrajectory Forecasting (Full Horizon 0-{t_max:.0f}s on uniform reference grid):")
    ml_rmses = df_summary['ml_rmse'].dropna()
    fit_rmses = df_summary['fit_rmse'].dropna()
    ml_r2s = df_summary['ml_r2'].dropna()
    fit_r2s = df_summary['fit_r2'].dropna()
    
    if len(ml_rmses) > 0:
        print(f"  ML RMSE:  {ml_rmses.mean():.6f} ± {ml_rmses.std():.6f}")
        print(f"  Fit RMSE: {fit_rmses.mean():.6f} ± {fit_rmses.std():.6f}")
        print(f"  ML R²:    {ml_r2s.mean():.4f}")
        print(f"  Fit R²:   {fit_r2s.mean():.4f}")
    
    # Save summary JSON
    summary_json = {
        'prefix_length': prefix_length,
        't_max': t_max,
        'reference_grid_dt': CONFIG.get("reference_grid_dt", 5.0),
        'n_test_samples': len(all_summaries),
        'seed': seed,
        'parameterization': 'unified_E0_g_per_L_k_d',
        'parameter_metrics': {
            param: {
                'ML_MAE_mean': float(df_summary[f'ml_{param}_mae'].mean()),
                'ML_MAE_std': float(df_summary[f'ml_{param}_mae'].std()),
                'Fit_MAE_mean': float(df_summary[f'fit_{param}_mae'].mean()),
                'Fit_MAE_std': float(df_summary[f'fit_{param}_mae'].std()),
            }
            for param in infer_params
        },
        'trajectory_metrics': {
            'ML_RMSE_mean': float(ml_rmses.mean()),
            'ML_RMSE_std': float(ml_rmses.std()),
            'Fit_RMSE_mean': float(fit_rmses.mean()),
            'Fit_RMSE_std': float(fit_rmses.std()),
            'ML_R2_mean': float(ml_r2s.mean()),
            'Fit_R2_mean': float(fit_r2s.mean()),
        },
    }
    
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(summary_json, f, indent=2)
    
    print(f"\n✓ Evaluation complete!")
    print(f"  Results saved to: {output_dir}")
    print(f"  Per-sample CSVs: {output_dir}/sample_*_trajectories.csv")
    print(f"  Aggregate CSV: {aggregate_csv_path}")
    print(f"  Test IDs: {output_dir}/test_sample_ids.json")


if __name__ == "__main__":
    main()
