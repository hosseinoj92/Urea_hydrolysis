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

from deeponet import create_deeponet
from mechanistic_simulator import UreaseSimulator, simulate_forward

# ╔══════════════════════════════════════════════════════════════╗
# ║                       USER CONFIG                             ║
# ╚══════════════════════════════════════════════════════════════╝
CONFIG = {
    # Model and data paths
    "model_path": "models/best_model.pt",  # Path to trained model checkpoint
    "data_dir": "data",                     # Directory containing training_data.npz
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


def evaluate_model(
    model: torch.nn.Module,
    params: np.ndarray,
    t_grid: np.ndarray,
    true_outputs: np.ndarray,
    metadata: dict,
    device: torch.device,
    use_totals: bool = True
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
    
    with torch.no_grad():
        for i in range(n_samples):
            x = torch.FloatTensor(params[i:i+1]).to(device)  # (1, n_params)
            
            for j, t in enumerate(t_grid):
                t_tensor = torch.FloatTensor([[t]]).to(device)  # (1, 1)
                y_pred = model(x, t_tensor).cpu().numpy()  # (1, n_outputs)
                predictions[i, j, :] = y_pred[0, :]
    
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
    n_runs: int = 100
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
    
    with torch.no_grad():
        for _ in range(n_runs):
            for i in range(n_samples):
                x = torch.FloatTensor(test_params[i:i+1]).to(device)
                for t_val in t_grid:
                    t = torch.FloatTensor([[t_val]]).to(device)
                    _ = model(x, t)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    deeponet_time = time.time() - start
    deeponet_time_per_sample = deeponet_time / (n_runs * n_samples)
    
    # ODE solver benchmark
    sim = UreaseSimulator(
        S0=metadata.get('S0', 0.020),
        N0=metadata.get('N0', 0.0),
        C0=metadata.get('C0', 0.0),
        T_K=metadata.get('T_K', 313.15),
        initial_pH=metadata.get('initial_pH', 7.36),
        E_loading_base_g_per_L=metadata.get('E_loading_base_g_per_L', 0.5)
    )
    
    start = time.time()
    for _ in range(n_runs):
        for i in range(n_samples):
            p = {
                'a': test_params[i, 0],
                'E_eff0': test_params[i, 1],
                'k_d': test_params[i, 2],
                't_shift': test_params[i, 3],
                'tau_probe': test_params[i, 4],
            }
            _ = sim.simulate_forward(p, t_grid, return_totals=True)
    
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
    
    # Use held-out test set (last N samples)
    n_test = min(n_test_samples, params.shape[0])
    test_params = params[-n_test:]
    test_outputs = outputs[-n_test:]
    
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
    speed_metrics = benchmark_speed(model, test_params, t_grid, metadata, device)
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
            # Plot pH computed from totals
            from mechanistic_simulator import UreaseSimulator
            sim = UreaseSimulator(
                S0=metadata.get('S0', 0.020),
                N0=metadata.get('N0', 0.0),
                C0=metadata.get('C0', 0.0),
                T_K=metadata.get('T_K', 313.15),
                initial_pH=metadata.get('initial_pH', 7.36),
                E_loading_base_g_per_L=metadata.get('E_loading_base_g_per_L', 0.5)
            )
            
            # True pH from totals
            pH_true = []
            for j in range(len(t_grid)):
                sp = sim.compute_speciation(test_outputs[i, j, 1], test_outputs[i, j, 2], 0.0)
                pH_true.append(sp['pH'])
            pH_true = np.array(pH_true)
            
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
