"""
Industrial calibration: fit E_eff0 and k_d (and optionally t_shift/tau_probe) to
experimental pH(t) curves using trained DeepONet as fast forward model.

Demonstrates rapid batch-to-batch enzyme deactivation drift estimation from inline pH.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

from deeponet import create_deeponet
from mechanistic_simulator import UreaseSimulator

# ╔══════════════════════════════════════════════════════════════╗
# ║                       USER CONFIG                             ║
# ╚══════════════════════════════════════════════════════════════╝
CONFIG = {
    # Model and data paths
    "model_path": "models/best_model.pt",  # Path to trained model checkpoint
    "output_dir": "calibration",           # Directory to save calibration results
    
    # Experimental data files
    "exp_files": [
        # "replicate1.csv",
        # "replicate2.csv",
        # "replicate3.csv",
    ],
    "exp_labels": None,  # If None, auto-generate: exp1, exp2, ...
    
    # Fitting options
    "fit_E_eff0": False,      # Fit effective enzyme loading (usually fixed from grams)
    "fit_k_d": True,          # Fit deactivation rate [1/s]
    "fit_t_shift": False,     # Fit time shift [s]
    "fit_tau_probe": False,   # Fit probe lag time [s]
    
    # Device
    "device": "auto",         # Device: "auto", "cpu", or "cuda"
}


def load_model(model_path: Path, device: torch.device):
    """Load trained DeepONet model."""
    checkpoint = torch.load(model_path, map_location=device)
    metadata = checkpoint.get('metadata', {})
    args = checkpoint.get('args', {})
    
    use_totals = metadata.get('use_totals', True)
    n_outputs = 3 if use_totals else 1
    
    input_dim = len(metadata.get('param_names', ['a', 'E_eff0', 'k_d', 't_shift', 'tau_probe']))
    hidden_dims = args.get('hidden_dims', [128, 128, 128])
    branch_output_dim = args.get('branch_output_dim', 128)
    
    model = create_deeponet(
        input_dim=input_dim,
        n_outputs=n_outputs,
        hidden_dims=hidden_dims,
        branch_output_dim=branch_output_dim,
        use_bias=True
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, metadata


def deeponet_forward(model, params_dict, t_grid, metadata, device, use_totals=True):
    """
    Forward prediction using DeepONet.
    
    Parameters:
    -----------
    params_dict: dict with keys ['a', 'E_eff0', 'k_d', 't_shift', 'tau_probe']
    t_grid: time points [s]
    metadata: model metadata
    device: torch device
    use_totals: if True, return totals; else return pH
    
    Returns:
    --------
    outputs: (n_times, n_outputs) array
    """
    param_names = metadata.get('param_names', ['a', 'E_eff0', 'k_d', 't_shift', 'tau_probe'])
    x = np.array([params_dict.get(name, 0.0) for name in param_names])
    x_tensor = torch.FloatTensor(x).unsqueeze(0).to(device)  # (1, n_params)
    
    outputs = []
    with torch.no_grad():
        for t in t_grid:
            t_tensor = torch.FloatTensor([[t]]).to(device)  # (1, 1)
            y_pred = model(x_tensor, t_tensor).cpu().numpy()  # (1, n_outputs)
            outputs.append(y_pred[0, :])
    
    outputs = np.array(outputs)  # (n_times, n_outputs)
    
    if use_totals:
        return outputs  # (n_times, 3) = [S, Ntot, Ctot]
    else:
        return outputs[:, 0]  # (n_times,) = pH


def convert_totals_to_pH(S, Ntot, Ctot, metadata):
    """Convert totals to pH using mechanistic speciation."""
    sim = UreaseSimulator(
        S0=metadata.get('S0', 0.020),
        N0=metadata.get('N0', 0.0),
        C0=metadata.get('C0', 0.0),
        T_K=metadata.get('T_K', 313.15),
        initial_pH=metadata.get('initial_pH', 7.36),
        E_loading_base_g_per_L=metadata.get('E_loading_base_g_per_L', 0.5)
    )
    
    pH = []
    for s, n, c in zip(S, Ntot, Ctot):
        sp = sim.compute_speciation(n, c, 0.0)
        pH.append(sp['pH'])
    return np.array(pH)


def load_experimental_data(csv_path: Path, time_col_hints=None, ph_col_hints=None, time_to_seconds=1.0):
    """
    Load experimental pH(t) data from CSV.
    
    Parameters:
    -----------
    csv_path: path to CSV file
    time_col_hints: list of possible time column names
    ph_col_hints: list of possible pH column names
    time_to_seconds: multiplier to convert time to seconds
    
    Returns:
    --------
    times, pH: arrays
    """
    if time_col_hints is None:
        time_col_hints = ["time", "t", "Time", "T"]
    if ph_col_hints is None:
        ph_col_hints = ["ph", "pH", "PH"]
    
    df = pd.read_csv(csv_path, sep=None, engine="python")
    
    # Find time column
    time_col = None
    for hint in time_col_hints:
        for col in df.columns:
            if hint.lower() in str(col).lower():
                time_col = col
                break
        if time_col:
            break
    
    # Find pH column
    ph_col = None
    for hint in ph_col_hints:
        for col in df.columns:
            if hint.lower() in str(col).lower():
                ph_col = col
                break
        if ph_col:
            break
    
    if time_col is None or ph_col is None:
        raise ValueError(f"Could not find time/pH columns. Columns: {list(df.columns)}")
    
    times = np.asarray(df[time_col], dtype=float) * time_to_seconds
    pH = np.asarray(df[ph_col], dtype=float)
    
    # Remove NaN/inf
    mask = np.isfinite(times) & np.isfinite(pH)
    times = times[mask]
    pH = pH[mask]
    
    # Sort by time
    order = np.argsort(times)
    times = times[order]
    pH = pH[order]
    
    return times, pH


def fit_experiment(
    model,
    metadata,
    exp_times,
    exp_pH,
    device,
    fit_E_eff0=True,
    fit_k_d=True,
    fit_t_shift=False,
    fit_tau_probe=False,
    E_eff0_bounds=(0.1, 1.0),
    k_d_bounds=(0.0, 5e-3),
    t_shift_bounds=(-10.0, 30.0),
    tau_probe_bounds=(0.0, 30.0),
    E_eff0_init=0.5,
    k_d_init=0.0,
    t_shift_init=0.0,
    tau_probe_init=0.0
):
    """
    Fit parameters to experimental pH(t) using DeepONet as forward model.
    
    Returns:
    --------
    result: dict with fitted parameters and metrics
    """
    use_totals = metadata.get('use_totals', True)
    
    # Build parameter vector
    param_names = metadata.get('param_names', ['a', 'E_eff0', 'k_d', 't_shift', 'tau_probe'])
    params0 = []
    bounds_lower = []
    bounds_upper = []
    param_indices = {}
    idx = 0
    
    # Fixed parameters (not fitted)
    fixed_params = {
        'a': 1.0,  # Activity scale (not fitted, use E_eff0 directly)
    }
    
    if fit_E_eff0:
        params0.append(E_eff0_init)
        bounds_lower.append(E_eff0_bounds[0])
        bounds_upper.append(E_eff0_bounds[1])
        param_indices['E_eff0'] = idx
        idx += 1
    else:
        fixed_params['E_eff0'] = E_eff0_init
    
    if fit_k_d:
        params0.append(k_d_init)
        bounds_lower.append(k_d_bounds[0])
        bounds_upper.append(k_d_bounds[1])
        param_indices['k_d'] = idx
        idx += 1
    else:
        fixed_params['k_d'] = k_d_init
    
    if fit_t_shift:
        params0.append(t_shift_init)
        bounds_lower.append(t_shift_bounds[0])
        bounds_upper.append(t_shift_bounds[1])
        param_indices['t_shift'] = idx
        idx += 1
    else:
        fixed_params['t_shift'] = t_shift_init
    
    if fit_tau_probe:
        params0.append(tau_probe_init)
        bounds_lower.append(tau_probe_bounds[0])
        bounds_upper.append(tau_probe_bounds[1])
        param_indices['tau_probe'] = idx
        idx += 1
    else:
        fixed_params['tau_probe'] = tau_probe_init
    
    def residuals(x):
        """Residuals for least_squares."""
        # Build full parameter dict
        params_dict = fixed_params.copy()
        for name, i in param_indices.items():
            params_dict[name] = x[i]
        
        # Forward prediction
        try:
            if use_totals:
                outputs = deeponet_forward(model, params_dict, exp_times, metadata, device, use_totals=True)
                S, Ntot, Ctot = outputs[:, 0], outputs[:, 1], outputs[:, 2]
                pH_pred = convert_totals_to_pH(S, Ntot, Ctot, metadata)
            else:
                pH_pred = deeponet_forward(model, params_dict, exp_times, metadata, device, use_totals=False)
            
            return pH_pred - exp_pH
        except Exception as e:
            print(f"Warning: Forward prediction failed: {e}")
            return np.full_like(exp_pH, 1e6)
    
    # Optimize
    print(f"Fitting {len(params0)} parameters...")
    result = least_squares(
        residuals,
        x0=np.array(params0),
        bounds=(np.array(bounds_lower), np.array(bounds_upper)),
        method='trf',
        xtol=1e-8,
        ftol=1e-8,
        verbose=0
    )
    
    # Extract fitted parameters
    fitted_params = fixed_params.copy()
    for name, i in param_indices.items():
        fitted_params[name] = result.x[i]
    
    # Evaluate final fit
    if use_totals:
        outputs = deeponet_forward(model, fitted_params, exp_times, metadata, device, use_totals=True)
        S, Ntot, Ctot = outputs[:, 0], outputs[:, 1], outputs[:, 2]
        pH_fit = convert_totals_to_pH(S, Ntot, Ctot, metadata)
    else:
        pH_fit = deeponet_forward(model, fitted_params, exp_times, metadata, device, use_totals=False)
    
    rmse = np.sqrt(np.mean((pH_fit - exp_pH)**2))
    mae = np.mean(np.abs(pH_fit - exp_pH))
    
    return {
        'fitted_params': fitted_params,
        'rmse': rmse,
        'mae': mae,
        'pH_fit': pH_fit,
        'optimization_result': result
    }


def main():
    # Use CONFIG dictionary for all parameters
    model_path = Path(CONFIG["model_path"])
    output_dir = Path(CONFIG["output_dir"])
    exp_files = CONFIG["exp_files"]
    exp_labels = CONFIG["exp_labels"]
    fit_E_eff0 = CONFIG["fit_E_eff0"]
    fit_k_d = CONFIG["fit_k_d"]
    fit_t_shift = CONFIG["fit_t_shift"]
    fit_tau_probe = CONFIG["fit_tau_probe"]
    
    # Check if experimental files are provided
    if not exp_files or len(exp_files) == 0:
        raise ValueError("No experimental files specified in CONFIG['exp_files']. Please add CSV file paths.")
    
    # Device
    if CONFIG["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(CONFIG["device"])
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    model, metadata = load_model(model_path, device)
    print("✓ Model loaded")
    
    # Labels
    if exp_labels is None:
        exp_labels = [f"exp{i+1}" for i in range(len(exp_files))]
    
    if len(exp_labels) != len(exp_files):
        raise ValueError("Number of labels must match number of experiment files")
    
    # Load and fit each experiment
    results = []
    
    print(f"\nCalibrating {len(exp_files)} experiments...")
    print("="*60)
    
    for exp_file, label in zip(exp_files, exp_labels):
        print(f"\n{label}: {exp_file}")
        
        # Load experimental data
        exp_times, exp_pH = load_experimental_data(Path(exp_file))
        print(f"  Loaded {len(exp_times)} data points")
        print(f"  Time range: [{exp_times[0]:.1f}, {exp_times[-1]:.1f}] s")
        print(f"  pH range: [{exp_pH.min():.3f}, {exp_pH.max():.3f}]")
        
        # Fit
        fit_result = fit_experiment(
            model,
            metadata,
            exp_times,
            exp_pH,
            device,
            fit_E_eff0=fit_E_eff0,
            fit_k_d=fit_k_d,
            fit_t_shift=fit_t_shift,
            fit_tau_probe=fit_tau_probe
        )
        
        print(f"  Fitted E_eff0: {fit_result['fitted_params']['E_eff0']:.6f} g/L")
        print(f"  Fitted k_d: {fit_result['fitted_params']['k_d']:.6e} s^-1")
        if fit_t_shift:
            print(f"  Fitted t_shift: {fit_result['fitted_params']['t_shift']:.3f} s")
        if fit_tau_probe:
            print(f"  Fitted tau_probe: {fit_result['fitted_params']['tau_probe']:.3f} s")
        print(f"  RMSE: {fit_result['rmse']:.4f} pH units")
        print(f"  MAE: {fit_result['mae']:.4f} pH units")
        
        results.append({
            'label': label,
            'file': exp_file,
            'exp_times': exp_times,
            'exp_pH': exp_pH,
            **fit_result
        })
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Summary table
    summary_data = []
    for r in results:
        summary_data.append({
            'Experiment': r['label'],
            'E_eff0 (g/L)': r['fitted_params']['E_eff0'],
            'k_d (1/s)': r['fitted_params']['k_d'],
            'k_d (1/h)': r['fitted_params']['k_d'] * 3600,
            'RMSE (pH)': r['rmse'],
            'MAE (pH)': r['mae']
        })
        if fit_t_shift:
            summary_data[-1]['t_shift (s)'] = r['fitted_params']['t_shift']
        if fit_tau_probe:
            summary_data[-1]['tau_probe (s)'] = r['fitted_params']['tau_probe']
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / "calibration_summary.csv", index=False)
    
    print("\n" + "="*60)
    print("CALIBRATION SUMMARY")
    print("="*60)
    print(summary_df.to_string(index=False))
    
    # Save detailed results
    with open(output_dir / "calibration_results.json", "w") as f:
        json.dump({
            r['label']: {
                'file': r['file'],
                'fitted_params': {k: float(v) if isinstance(v, (np.floating, float)) else v
                                 for k, v in r['fitted_params'].items()},
                'rmse': float(r['rmse']),
                'mae': float(r['mae'])
            }
            for r in results
        }, f, indent=2)
    
    # Overlay plots
    n_exps = len(results)
    fig, axes = plt.subplots(n_exps, 1, figsize=(10, 4*n_exps))
    if n_exps == 1:
        axes = [axes]
    
    for i, r in enumerate(results):
        ax = axes[i]
        ax.plot(r['exp_times'], r['exp_pH'], 'o', label='Experiment', markersize=4, alpha=0.7)
        ax.plot(r['exp_times'], r['pH_fit'], '-', label='DeepONet Fit', linewidth=2)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('pH')
        ax.set_title(f"{r['label']} — E_eff0={r['fitted_params']['E_eff0']:.4f} g/L, "
                    f"k_d={r['fitted_params']['k_d']:.2e} s⁻¹, RMSE={r['rmse']:.3f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "calibration_overlays.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Calibration complete!")
    print(f"  Results saved to: {output_dir}")
    print(f"  Summary: {output_dir / 'calibration_summary.csv'}")
    print(f"  Plots: {output_dir / 'calibration_overlays.png'}")
    print("\nStory: ML-trained DeepONet enables real-time digital-twin speed for")
    print("rapid batch-to-batch enzyme deactivation drift estimation from inline pH.")


if __name__ == "__main__":
    main()
