"""
Evaluate DeepONet surrogate model on held-out data and benchmark speed vs ODE solver.
"""

import torch
import numpy as np
import time
from pathlib import Path
import json
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from tqdm import tqdm
import sys
import os

# Check if we're in an interactive terminal that supports progress bars
def is_interactive_terminal():
    """Check if we're in an interactive terminal that supports tqdm."""
    return hasattr(sys.stdout, 'isatty') and sys.stdout.isatty() and os.getenv('TERM') != 'dumb'

from deeponet import create_deeponet
from mechanistic_simulator import UreaseSimulator, simulate_forward

# ╔══════════════════════════════════════════════════════════════╗
# ║                       USER CONFIG                             ║
# ╚══════════════════════════════════════════════════════════════╝
CONFIG = {
    # Model and data paths
    "model_path": "models/best_model.pt",  # Path to trained model checkpoint
    "data_dir": "Generated_Data_5params_10000",                     # Directory containing training_data.npz
    "output_dir": "evaluation",            # Directory to save evaluation results
    
    # Evaluation parameters
    "n_test_samples": 100,                  # Number of test samples to evaluate
    "device": "auto",                       # Device: "auto", "cpu", or "cuda"
}


def load_model(model_path: Path, device: torch.device):
    """Load trained DeepONet model."""
    checkpoint = torch.load(model_path, map_location=device)
    metadata = checkpoint.get('metadata', {})
    args = checkpoint.get('args', {})
    
    use_totals = metadata.get('use_totals', True)
    n_outputs = 3 if use_totals else 1
    
    # Reconstruct model
    # Use variable_params if available (new format), otherwise fall back to param_names (old format)
    variable_params = metadata.get('variable_params', None)
    if variable_params is not None:
        input_dim = len(variable_params)  # New format: only variable parameters
    else:
        # Old format: use param_names or default
        param_names = metadata.get('param_names', ['a', 'E_eff0', 'k_d', 't_shift', 'tau_probe'])
        input_dim = len(param_names)
    
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


def evaluate_model(
    model: torch.nn.Module,
    params: np.ndarray,
    t_grid: np.ndarray,
    true_outputs: np.ndarray,
    metadata: dict,
    device: torch.device,
    use_totals: bool = True,
    normalize: bool = True
) -> dict:
    """
    Evaluate model predictions vs true outputs.
    
    Returns:
    --------
    metrics: dict with RMSE, MAE, R², correlation, etc.
    """
    model.eval()
    n_samples, n_times, n_outputs = true_outputs.shape
    
    predictions = np.zeros_like(true_outputs)
    
    # Get normalization stats if available
    norm_stats = metadata.get('normalization_stats', {})
    if normalize and norm_stats:
        param_mean = np.array(norm_stats.get('param_mean', [0.0] * params.shape[1]))
        param_std = np.array(norm_stats.get('param_std', [1.0] * params.shape[1]))
        t_max = norm_stats.get('t_max', np.max(t_grid))
        output_mean = np.array(norm_stats.get('output_mean', [0.0] * n_outputs))
        output_std = np.array(norm_stats.get('output_std', [1.0] * n_outputs))
    else:
        param_mean = np.zeros(params.shape[1])
        param_std = np.ones(params.shape[1])
        t_max = np.max(t_grid)
        output_mean = np.zeros(n_outputs)
        output_std = np.ones(n_outputs)
    
    # Normalize inputs
    params_norm = (params - param_mean) / (param_std + 1e-8)
    t_grid_norm = t_grid / t_max
    
    # Convert t_grid to tensor once
    t_tensor_all = torch.FloatTensor(t_grid_norm).unsqueeze(1).to(device)  # (n_times, 1)
    
    with torch.no_grad():
        # Single progress bar for samples - configured to update in place
        pbar = tqdm(
            range(n_samples), 
            desc="Evaluating", 
            unit="sample", 
            ncols=120, 
            dynamic_ncols=False,
            file=sys.stdout,
            mininterval=0.1,  # Update at most every 0.1 seconds
            maxinterval=1.0,   # Update at least every 1 second
            smoothing=0.1      # Smooth progress updates
        )
        for i in pbar:
            x = torch.FloatTensor(params[i:i+1]).to(device)  # (1, n_params)
            
            # Batch predict all time points at once for this sample
            # Expand x to match all time points: (1, n_params) -> (n_times, n_params)
            x_expanded = x.expand(n_times, -1)  # (n_times, n_params)
            
            # Predict all time points at once
            y_pred = model(x_expanded, t_tensor_all).cpu().numpy()  # (n_times, n_outputs)
            predictions[i, :, :] = y_pred
            
            # Update progress bar (only update postfix, not the bar itself on every iteration)
            if (i + 1) % max(1, n_samples // 100) == 0 or i == n_samples - 1:
                pbar.set_postfix({
                    'sample': f'{i+1}/{n_samples}',
                    'progress': f'{(i+1)/n_samples*100:.1f}%'
                })
    
    # Compute metrics per output
    metrics = {}
    
    if use_totals:
        output_names = ['S (urea)', 'Ntot (ammonia)', 'Ctot (carbon)']
    else:
        output_names = ['pH']
    
    for k, name in enumerate(output_names):
        true_k = true_outputs[:, :, k].flatten()
        pred_k = predictions[:, :, k].flatten()
        
        # Remove any NaN/inf
        mask = np.isfinite(true_k) & np.isfinite(pred_k)
        true_k = true_k[mask]
        pred_k = pred_k[mask]
        
        rmse = np.sqrt(np.mean((pred_k - true_k)**2))
        mae = np.mean(np.abs(pred_k - true_k))
        r2 = 1 - np.sum((pred_k - true_k)**2) / np.sum((true_k - np.mean(true_k))**2)
        corr, _ = pearsonr(pred_k, true_k)
        
        metrics[name] = {
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'Correlation': corr
        }
    
    # Overall metrics
    metrics['overall'] = {
        'mean_RMSE': np.mean([m['RMSE'] for m in metrics.values() if isinstance(m, dict) and 'RMSE' in m]),
        'mean_R²': np.mean([m['R²'] for m in metrics.values() if isinstance(m, dict) and 'R²' in m])
    }
    
    return metrics, predictions


def benchmark_speed(
    model: torch.nn.Module,
    params: np.ndarray,
    t_grid: np.ndarray,
    metadata: dict,
    device: torch.device,
    n_runs: int = 100,
    variable_params: list = None,
    fixed_params: dict = None
) -> dict:
    """Benchmark DeepONet speed vs ODE solver."""
    
    # DeepONet benchmark
    model.eval()
    n_samples = min(10, params.shape[0])  # Use subset for speed test
    test_params = params[:n_samples]
    
    # Warmup
    with torch.no_grad():
        x = torch.FloatTensor(test_params[0:1]).to(device)
        t = torch.FloatTensor([[t_grid[0]]]).to(device)
        _ = model(x, t)
    
    # Time DeepONet
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.time()
    
    # Prepare tensors once
    t_tensor_all = torch.FloatTensor(t_grid).unsqueeze(1).to(device)  # (n_times, 1)
    
    with torch.no_grad():
        total_iterations = n_runs * n_samples
        # Use disable=False to ensure progress bar is shown, and position=0 to ensure it stays in place
        # Disable if not in interactive terminal to prevent stacking
        pbar = tqdm(
            total=total_iterations, 
            desc="Benchmarking DeepONet", 
            unit="iter", 
            ncols=120, 
            leave=False,
            file=sys.stdout,
            mininterval=0.5,  # Update at most every 0.5 seconds
            maxinterval=2.0,  # Update at least every 2 seconds
            smoothing=0.1,
            disable=not is_interactive_terminal(),
            position=0,
            dynamic_ncols=False
        )
        iteration = 0
        update_interval = max(1, total_iterations // 100)  # Update ~100 times total
        for run_idx in range(n_runs):
            for i in range(n_samples):
                x = torch.FloatTensor(test_params[i:i+1]).to(device)
                x_expanded = x.expand(len(t_grid), -1)  # (n_times, n_params)
                _ = model(x_expanded, t_tensor_all)  # Batch predict all time points
                iteration += 1
                # Update progress bar periodically
                if iteration % update_interval == 0 or iteration == total_iterations:
                    pbar.n = iteration
                    pbar.refresh()
                    pbar.set_postfix({'samples': n_samples, 'runs': n_runs})
        pbar.n = total_iterations
        pbar.refresh()
        pbar.close()
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    deeponet_time = time.time() - start
    deeponet_time_per_sample = deeponet_time / (n_runs * n_samples)
    
    # ODE solver benchmark - use per-sample parameters
    start = time.time()
    total_iterations = n_runs * n_samples
    # Use disable=False to ensure progress bar is shown, and position=0 to ensure it stays in place
    # Disable if not in interactive terminal to prevent stacking
    pbar = tqdm(
        total=total_iterations, 
        desc="Benchmarking ODE solver", 
        unit="iter", 
        ncols=120, 
        leave=False,
        file=sys.stdout,
        mininterval=0.5,  # Update at most every 0.5 seconds
        maxinterval=2.0,  # Update at least every 2 seconds
        smoothing=0.1,
        disable=not is_interactive_terminal(),
        position=0,
        dynamic_ncols=False
    )
    iteration = 0
    update_interval = max(1, total_iterations // 100)  # Update ~100 times total
    for run_idx in range(n_runs):
        for i in range(n_samples):
            # Reconstruct full parameter set for this sample
            sample_params = {}
            if variable_params is not None:
                for j, param_name in enumerate(variable_params):
                    sample_params[param_name] = test_params[i, j]
                if fixed_params is not None:
                    sample_params.update(fixed_params)
            else:
                # Fallback to old format (backward compatibility)
                sample_params = {
                    'a': test_params[i, 0] if test_params.shape[1] > 0 else 1.0,
                    'k_d': test_params[i, 2] if test_params.shape[1] > 2 else 0.0,
                    't_shift': 0.0,
                    'tau_probe': 0.0,
                }
            
            # Extract physical parameters
            substrate_mM = sample_params.get("substrate_mM", 20.0)
            temperature_C = sample_params.get("temperature_C", 40.0)
            initial_pH_i = sample_params.get("initial_pH", 7.0)
            grams_urease_powder = sample_params.get("grams_urease_powder", 0.10)
            powder_activity_frac = sample_params.get("powder_activity_frac", 1.0)
            volume_L = sample_params.get("volume_L", 0.2)
            k_d = sample_params.get("k_d", 0.0)
            
            S0 = substrate_mM / 1000.0
            T_K = temperature_C + 273.15
            E_loading_base_g_per_L = grams_urease_powder * powder_activity_frac / volume_L
            
            sim = UreaseSimulator(
                S0=S0,
                N0=0.0,
                C0=0.0,
                T_K=T_K,
                initial_pH=initial_pH_i,
                E_loading_base_g_per_L=E_loading_base_g_per_L,
                use_T_dependent_pH_activity=True,
            )
            
            # Parameters for ODE solver
            p = {
                'a': 1.0,  # Activity already in powder_activity_frac
                'k_d': k_d,
                't_shift': 0.0,
                'tau_probe': 0.0,
            }
            _ = sim.simulate_forward(p, t_grid, return_totals=True)
            iteration += 1
            # Update progress bar periodically
            if iteration % update_interval == 0 or iteration == total_iterations:
                pbar.n = iteration
                pbar.refresh()
                pbar.set_postfix({'samples': n_samples, 'runs': n_runs})
    pbar.n = total_iterations
    pbar.refresh()
    pbar.close()
    
    ode_time = time.time() - start
    ode_time_per_sample = ode_time / (n_runs * n_samples)
    
    speedup = ode_time_per_sample / deeponet_time_per_sample
    
    return {
        'deeponet_time_per_sample': deeponet_time_per_sample,
        'ode_time_per_sample': ode_time_per_sample,
        'speedup': speedup,
        'n_runs': n_runs,
        'n_samples': n_samples
    }


def main():
    # Use CONFIG dictionary for all parameters
    model_path = Path(CONFIG["model_path"])
    data_dir = Path(CONFIG["data_dir"])
    output_dir = Path(CONFIG["output_dir"])
    n_test_samples = CONFIG["n_test_samples"]
    
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
    
    # Load test data
    data_file = data_dir / "training_data.npz"
    
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    data = np.load(data_file)
    params = data["params"]
    outputs = data["outputs"]
    t_grid = data["t_grid"]
    pH_saved = data.get("pH", None)  # Load saved pH if available
    
    # Load metadata to get parameter structure
    with open(data_dir / "metadata.json", "r") as f:
        data_metadata = json.load(f)
    
    variable_params = data_metadata.get("variable_params", [])
    fixed_params = data_metadata.get("fixed_params", {})
    
    # Use held-out test set (last N samples)
    n_test = min(n_test_samples, params.shape[0])
    test_params = params[-n_test:]
    test_outputs = outputs[-n_test:]
    test_pH_saved = pH_saved[-n_test:] if pH_saved is not None else None
    
    print(f"\nEvaluating on {n_test} test samples...")
    
    # Evaluate
    use_totals = metadata.get('use_totals', True)
    metrics, predictions = evaluate_model(
        model, test_params, t_grid, test_outputs, metadata, device, use_totals
    )
    
    # Print metrics
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    for name, m in metrics.items():
        if isinstance(m, dict):
            print(f"\n{name}:")
            for key, val in m.items():
                if isinstance(val, float):
                    print(f"  {key}: {val:.6f}")
    
    # Benchmark speed
    print("\n" + "="*60)
    print("SPEED BENCHMARK")
    print("="*60)
    speed_metrics = benchmark_speed(
        model, test_params, t_grid, metadata, device,
        variable_params=variable_params,
        fixed_params=fixed_params
    )
    print(f"DeepONet time per sample: {speed_metrics['deeponet_time_per_sample']*1000:.2f} ms")
    print(f"ODE solver time per sample: {speed_metrics['ode_time_per_sample']*1000:.2f} ms")
    print(f"Speedup: {speed_metrics['speedup']:.1f}x")
    
    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump({
            'metrics': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                           for kk, vv in v.items()} if isinstance(v, dict) else v
                       for k, v in metrics.items()},
            'speed_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v
                            for k, v in speed_metrics.items()}
        }, f, indent=2)
    
    # Plot sample predictions
    n_plot = min(5, n_test)
    fig, axes = plt.subplots(n_plot, 1, figsize=(10, 2*n_plot))
    if n_plot == 1:
        axes = [axes]
    
    for i in range(n_plot):
        ax = axes[i]
        if use_totals:
            # Use saved pH if available (most accurate)
            if test_pH_saved is not None:
                pH_true = test_pH_saved[i, :]
            else:
                # Fallback: compute pH from totals using per-sample parameters
                from mechanistic_simulator import UreaseSimulator
                
                # Reconstruct full parameter set for this sample
                sample_params = {}
                for j, param_name in enumerate(variable_params):
                    sample_params[param_name] = test_params[i, j]
                sample_params.update(fixed_params)
                
                # Extract physical parameters
                substrate_mM = sample_params.get("substrate_mM", 20.0)
                temperature_C = sample_params.get("temperature_C", 40.0)
                initial_pH_i = sample_params.get("initial_pH", 7.0)
                grams_urease_powder = sample_params.get("grams_urease_powder", 0.10)
                powder_activity_frac = sample_params.get("powder_activity_frac", 1.0)
                volume_L = sample_params.get("volume_L", 0.2)
                
                S0 = substrate_mM / 1000.0
                T_K = temperature_C + 273.15
                E_loading_base_g_per_L = grams_urease_powder * powder_activity_frac / volume_L
                
                sim = UreaseSimulator(
                    S0=S0,
                    N0=0.0,
                    C0=0.0,
                    T_K=T_K,
                    initial_pH=initial_pH_i,
                    E_loading_base_g_per_L=E_loading_base_g_per_L,
                    use_T_dependent_pH_activity=True,
                )
                
                # True pH from totals using correct simulator
                pH_true = []
                for j in range(len(t_grid)):
                    sp = sim.compute_speciation(test_outputs[i, j, 1], test_outputs[i, j, 2], 0.0)
                    pH_true.append(sp['pH'])
                pH_true = np.array(pH_true)
            
            # Predicted pH from predicted totals (use same simulator setup)
            from mechanistic_simulator import UreaseSimulator
            
            # Reconstruct full parameter set for this sample
            sample_params = {}
            for j, param_name in enumerate(variable_params):
                sample_params[param_name] = test_params[i, j]
            sample_params.update(fixed_params)
            
            # Extract physical parameters
            substrate_mM = sample_params.get("substrate_mM", 20.0)
            temperature_C = sample_params.get("temperature_C", 40.0)
            initial_pH_i = sample_params.get("initial_pH", 7.0)
            grams_urease_powder = sample_params.get("grams_urease_powder", 0.10)
            powder_activity_frac = sample_params.get("powder_activity_frac", 1.0)
            volume_L = sample_params.get("volume_L", 0.2)
            
            S0 = substrate_mM / 1000.0
            T_K = temperature_C + 273.15
            E_loading_base_g_per_L = grams_urease_powder * powder_activity_frac / volume_L
            
            sim = UreaseSimulator(
                S0=S0,
                N0=0.0,
                C0=0.0,
                T_K=T_K,
                initial_pH=initial_pH_i,
                E_loading_base_g_per_L=E_loading_base_g_per_L,
                use_T_dependent_pH_activity=True,
            )
            
            # Predicted pH from predicted totals
            pH_pred = []
            for j in range(len(t_grid)):
                sp = sim.compute_speciation(predictions[i, j, 1], predictions[i, j, 2], 0.0)
                pH_pred.append(sp['pH'])
            pH_pred = np.array(pH_pred)
            
            ax.plot(t_grid, pH_true, 'o-', label='True pH', markersize=3)
            ax.plot(t_grid, pH_pred, 's-', label='Predicted pH', markersize=3)
            ax.set_ylabel('pH')
        else:
            ax.plot(t_grid, test_outputs[i, :, 0], 'o-', label='True', markersize=3)
            ax.plot(t_grid, predictions[i, :, 0], 's-', label='Predicted', markersize=3)
            ax.set_ylabel('pH')
        
        ax.set_xlabel('Time [s]')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Sample {i+1}')
    
    plt.tight_layout()
    plt.savefig(output_dir / "sample_predictions.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Evaluation complete!")
    print(f"  Results saved to: {output_dir}")
    print(f"  Metrics: {output_dir / 'metrics.json'}")
    print(f"  Plots: {output_dir / 'sample_predictions.png'}")


if __name__ == "__main__":
    main()
