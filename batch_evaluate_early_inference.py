"""
Batch evaluation script for evaluating multiple trained models.

This script scans a directory of trained models (from batch training) and evaluates
each one, saving results in a parallel directory structure.
"""

import torch
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from datetime import datetime
import glob

# Import evaluation functions
from evaluate_early_inference import (
    evaluate_single_case,
    compute_metrics,
    time_to_threshold,
)
from forecast_early_inference import load_early_inference_model
from generate_early_inference_data import (
    extract_prefix,
    sample_parameters,
    build_time_grid,
    generate_trajectories,
)
from fit_mechanistic import fit_mechanistic_parameters
from mechanistic_simulator import UreaseSimulator
from scipy.interpolate import interp1d
import pandas as pd

# ╔══════════════════════════════════════════════════════════════╗
# ║                   BATCH EVALUATION CONFIG                     ║
# ╚══════════════════════════════════════════════════════════════╝

CONFIG = {
    # Paths
    "models_dir": r"C:\Users\vt4ho\Simulations\simulation_data\models\imperfect\version_2\batch_ML_training_models",
    "data_dir": r"C:\Users\vt4ho\Simulations\simulation_data\generated_data\imperfect\version_2\Generated_Data_EarlyInference_100000",
    "output_base_dir": r"C:\Users\vt4ho\Simulations\simulation_data\evaluation\imperfect\version_2\batch_evaluation_results",
    
    # Evaluation parameters
    "n_test_samples": 100,              # Number of test cases per model
    "t_max": 2000.0,                    # Full trajectory length [s]
    "reference_grid_dt": 5.0,           # Reference grid spacing [s] for fair comparison
    "device": "auto",
    "seed": 12345,                       # Deterministic seed for test set
    
    # Parameter fitting bounds (unified: E0_g_per_L and k_d only)
    "fit_bounds": {
        "E0_g_per_L": (5e-2, 1.25),  # Wide range covering slow to fast regimes [g/L]
        "k_d": (1e-5, 5e-3),
    },
    
    # Fitting configuration
    "fit_nuisance_params": True,  # If True, fit nuisance parameters; if False, use ground truth
}


def find_model_files(models_dir: Path):
    """
    Find all best_model files in the models directory.
    
    Returns
    -------
    List of tuples: (model_folder_name, model_file_path, prefix_length)
    """
    models_dir = Path(models_dir)
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    
    model_files = []
    
    # Find all directories in models_dir
    for folder in models_dir.iterdir():
        if not folder.is_dir():
            continue
        
        # Look for best_model_prefix_*.pt files
        best_model_files = list(folder.glob("best_model_prefix_*.pt"))
        
        if not best_model_files:
            # Try to find any .pt file as fallback
            pt_files = list(folder.glob("*.pt"))
            if pt_files:
                print(f"Warning: No best_model found in {folder.name}, using {pt_files[0].name}")
                best_model_files = [pt_files[0]]
            else:
                print(f"Warning: No model files found in {folder.name}, skipping")
                continue
        
        # Use the first best_model file found
        model_file = best_model_files[0]
        
        # Extract prefix_length from filename (e.g., "best_model_prefix_30s.pt" -> 30.0)
        filename = model_file.stem  # "best_model_prefix_30s"
        try:
            # Extract number from "best_model_prefix_30s"
            prefix_str = filename.replace("best_model_prefix_", "").replace("s", "")
            prefix_length = float(prefix_str)
        except ValueError:
            # Fallback: try to get from checkpoint
            try:
                checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
                prefix_length = checkpoint.get('prefix_length', 30.0)
                print(f"Warning: Could not extract prefix_length from filename, using {prefix_length} from checkpoint")
            except Exception as e:
                print(f"Warning: Could not determine prefix_length for {folder.name}, using default 30.0")
                prefix_length = 30.0
        
        model_files.append((folder.name, model_file, prefix_length))
    
    return model_files


def evaluate_model(
    model_folder_name: str,
    model_path: Path,
    prefix_length: float,
    data_dir: Path,
    output_dir: Path,
    config: dict,
    device: torch.device,
):
    """
    Evaluate a single model.
    
    Parameters
    ----------
    model_folder_name: Name of the model folder (for display)
    model_path: Path to the best_model file
    prefix_length: Prefix length used for this model
    data_dir: Directory containing training data
    output_dir: Directory to save evaluation results
    config: Evaluation configuration
    device: Torch device
    """
    print(f"\n{'='*80}")
    print(f"EVALUATING MODEL: {model_folder_name}")
    print(f"{'='*80}")
    print(f"Model path: {model_path}")
    print(f"Prefix length: {prefix_length}s")
    print(f"Output directory: {output_dir}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f"\nLoading model...")
    model, metadata, normalization_stats, prefix_length_model = load_early_inference_model(
        model_path, device
    )
    
    # Use prefix_length from model if available
    if prefix_length_model is not None:
        prefix_length = prefix_length_model
    
    print(f"Model loaded successfully")
    print(f"  Inferred parameters: {metadata['infer_params']}")
    print(f"  Known inputs: {metadata.get('known_input_names', [])}")
    print(f"  Prefix length: {prefix_length}s")
    
    # Load data directory
    data_file = data_dir / "training_data.npz"
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    data = np.load(data_file, allow_pickle=True)
    
    # Load metadata to understand data structure
    with open(data_dir / "metadata.json", "r") as f:
        data_metadata = json.load(f)
    
    # Use t_max from metadata if available
    t_max = data_metadata.get('t_max', config['t_max'])
    
    # Generate test cases with deterministic seed
    print(f"\nGenerating {config['n_test_samples']} test cases (seed={config['seed']})...")
    
    test_params_dict = sample_parameters(config['n_test_samples'], seed=config['seed'])
    
    # Use same time grid as training data
    if 'time_grid_mode' in data_metadata and 'time_grid_config' in data_metadata:
        time_grid_mode = data_metadata['time_grid_mode']
        time_grid_config = data_metadata['time_grid_config']
        t_grid = build_time_grid(mode=time_grid_mode, t_max=t_max, config=time_grid_config)
        print(f"Using time grid from metadata: {len(t_grid)} points ({time_grid_mode} mode)")
    elif 't_grid' in data_metadata:
        t_grid = np.array(data_metadata['t_grid'])
        print(f"Using stored time grid from metadata: {len(t_grid)} points")
    else:
        n_times = data_metadata.get('n_times', int(t_max / 1.0))
        t_grid = build_time_grid(mode="uniform", t_max=t_max, n_times=n_times)
        print(f"Using default uniform time grid: {len(t_grid)} points")
    
    # Generate trajectories
    test_results_list = generate_trajectories(test_params_dict, t_grid, n_workers=4)
    
    # Filter successful results
    test_results = [(i, r) for i, r in enumerate(test_results_list) if r is not None]
    print(f"Generated {len(test_results)} test trajectories")
    
    # Save test sample IDs
    test_sample_ids = [i for i, _ in test_results]
    with open(output_dir / "test_sample_ids.json", 'w') as f:
        json.dump({
            'seed': config['seed'],
            'n_samples': len(test_sample_ids),
            'sample_ids': test_sample_ids,
            'prefix_length': prefix_length,
            't_max': t_max,
            'reference_grid_dt': config.get("reference_grid_dt", 5.0),
            'model_folder': model_folder_name,
            'model_path': str(model_path),
        }, f, indent=2)
    
    # Evaluate each test case
    print(f"\nEvaluating {len(test_results)} test cases...")
    all_summaries = []
    
    for sample_idx, (original_idx, result) in enumerate(tqdm(test_results[:config['n_test_samples']], 
                                                           desc=f"Evaluating {model_folder_name}")):
        # Extract full trajectory
        pH_true_full = result['pH_true']
        pH_meas_full = result['pH_meas']
        t_full = np.array(result.get('t_grid', t_grid))
        known_inputs = result['known_inputs']
        true_params = result['target_params']
        nuisance_params = result.get('nuisance_params', {})
        
        # Evaluate
        summary = evaluate_single_case(
            sample_idx,
            pH_meas_full,
            pH_true_full,
            t_full, known_inputs, true_params,
            model, metadata, normalization_stats, prefix_length,
            t_max, config.get("reference_grid_dt", 5.0), device, output_dir,
            nuisance_params=nuisance_params,
            fit_nuisance_params=config.get("fit_nuisance_params", False),
        )
        
        if summary is not None:
            all_summaries.append(summary)
    
    print(f"\nEvaluated {len(all_summaries)} test cases")
    
    # Write aggregate CSV
    df_summary = pd.DataFrame(all_summaries)
    aggregate_csv_path = output_dir / "aggregate_metrics.csv"
    df_summary.to_csv(aggregate_csv_path, index=False)
    print(f"✓ Aggregate metrics saved to: {aggregate_csv_path}")
    
    # Compute aggregate statistics
    print("\n" + "="*60)
    print(f"AGGREGATE METRICS: {model_folder_name}")
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
    print(f"\nTrajectory Forecasting (Full Horizon 0-{t_max:.0f}s):")
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
        'model_folder': model_folder_name,
        'model_path': str(model_path),
        'prefix_length': prefix_length,
        't_max': t_max,
        'reference_grid_dt': config.get("reference_grid_dt", 5.0),
        'n_test_samples': len(all_summaries),
        'seed': config['seed'],
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
    
    print(f"\n✓ Evaluation complete for {model_folder_name}!")
    print(f"  Results saved to: {output_dir}")
    
    return summary_json


def main():
    """Main function to run batch evaluation."""
    print("="*80)
    print("BATCH EVALUATION FOR EARLY INFERENCE MODELS")
    print("="*80)
    
    # Get paths from CONFIG
    models_dir = Path(CONFIG["models_dir"])
    data_dir = Path(CONFIG["data_dir"])
    output_base_dir = Path(CONFIG["output_base_dir"])
    
    # Device
    if CONFIG["device"] == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(CONFIG["device"])
    
    print(f"Using device: {device}")
    print(f"Models directory: {models_dir}")
    print(f"Data directory: {data_dir}")
    print(f"Output base directory: {output_base_dir}")
    
    # Find all model files
    print(f"\nScanning for models in {models_dir}...")
    model_files = find_model_files(models_dir)
    
    if len(model_files) == 0:
        print("No model files found!")
        return
    
    print(f"\nFound {len(model_files)} models to evaluate:")
    for i, (folder_name, model_path, prefix_length) in enumerate(model_files, 1):
        print(f"  {i}. {folder_name} (prefix: {prefix_length}s)")
    
    # Ask for confirmation
    response = input(f"\nProceed with evaluating {len(model_files)} models? (yes/no): ").strip().lower()
    if response not in ['yes', 'y']:
        print("Batch evaluation cancelled.")
        return
    
    # Create output base directory
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate each model
    successful = []
    failed = []
    all_summaries = []
    
    start_time = datetime.now()
    
    for i, (folder_name, model_path, prefix_length) in enumerate(model_files, 1):
        print(f"\n[{i}/{len(model_files)}] Processing: {folder_name}")
        
        # Create output directory with same name as model folder
        output_dir = output_base_dir / folder_name
        
        try:
            summary = evaluate_model(
                folder_name,
                model_path,
                prefix_length,
                data_dir,
                output_dir,
                CONFIG,
                device,
            )
            successful.append(folder_name)
            all_summaries.append({
                'model_folder': folder_name,
                'prefix_length': prefix_length,
                **summary['parameter_metrics'],
                **summary['trajectory_metrics'],
            })
        except Exception as e:
            print(f"\n✗ Evaluation failed for {folder_name}")
            print(f"  Error: {str(e)}")
            failed.append(folder_name)
            
            # Save error to file
            error_file = output_dir / "evaluation_error.txt"
            error_file.parent.mkdir(parents=True, exist_ok=True)
            with open(error_file, 'w') as f:
                f.write(f"Evaluation failed for model: {folder_name}\n")
                f.write(f"Error: {str(e)}\n")
                import traceback
                f.write(traceback.format_exc())
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Print summary
    print("\n" + "="*80)
    print("BATCH EVALUATION SUMMARY")
    print("="*80)
    print(f"Total models: {len(model_files)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Total time: {duration}")
    
    if successful:
        print(f"\nSuccessful evaluations:")
        for name in successful:
            print(f"  ✓ {name}")
    
    if failed:
        print(f"\nFailed evaluations:")
        for name in failed:
            print(f"  ✗ {name}")
    
    # Save batch summary
    batch_summary_file = output_base_dir / "batch_evaluation_summary.json"
    batch_summary = {
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "duration_seconds": duration.total_seconds(),
        "total_models": len(model_files),
        "successful": successful,
        "failed": failed,
        "model_summaries": all_summaries,
    }
    with open(batch_summary_file, 'w') as f:
        json.dump(batch_summary, f, indent=2)
    print(f"\nBatch summary saved to: {batch_summary_file}")
    
    # Create comparison CSV (if multiple models succeeded)
    if len(all_summaries) > 1:
        comparison_df = pd.DataFrame(all_summaries)
        comparison_csv = output_base_dir / "model_comparison.csv"
        comparison_df.to_csv(comparison_csv, index=False)
        print(f"Model comparison CSV saved to: {comparison_csv}")


if __name__ == "__main__":
    main()
