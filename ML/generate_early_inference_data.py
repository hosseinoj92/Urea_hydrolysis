"""
Generate training dataset for early inference model.

This generates realistic training examples by:
1. Sampling parameter vectors from ranges in CONFIG
2. Running the mechanistic model to produce full trajectories
3. Applying a measurement model to pH (probe time constant, noise, offset/drift)
4. Extracting many prefixes (e.g., first 10s/30s/60s) as inputs with targets being the underlying parameters
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
import os
from mechanistic_simulator import UreaseSimulator

# ╔══════════════════════════════════════════════════════════════╗
# ║                       USER CONFIG                             ║
# ╚══════════════════════════════════════════════════════════════╝
CONFIG = {
    # Dataset generation parameters
    "n_samples": 20000,           # Number of full trajectories to generate
    "t_max": 2000.0,              # Maximum time [s] for full trajectories
    "n_times": 2000,              # Number of time points in full trajectory
    "seed": 42,                    # Random seed for reproducibility
    "output_dir": "Generated_Data_EarlyInference_20000",  # Output directory
    
    # Prefix extraction (multiple prefix lengths for training)
    "prefix_lengths": [10.0, 30.0, 60.0, 120.0],  # Prefix lengths in seconds [s]
    "prefix_n_points": 50,        # Number of points to extract from each prefix (uniform sampling)
    
    # Parameters to infer (latent kinetic/deactivation parameters)
    "infer_params": [
        "activity_scale",  # Effective activity scaling (multiplies base enzyme loading)
        "k_d",             # Deactivation rate [1/s]
        "tau_probe",       # Probe time constant [s] (optional)
        "pH_offset",       # pH measurement offset (optional)

    ],
    
    # Parameter sampling ranges
    "param_ranges": {
        # Physical conditions (known inputs)
        "substrate_mM": [1.0, 100.0],
        "grams_urease_powder": [0.01, 0.5],
        "temperature_C": [20.0, 40.0],
        "initial_pH": [6.5, 7.5],
        "powder_activity_frac": [0.01, 0.5],
        "volume_L": [0.1, 1.0],
        
        # Latent parameters to infer
        "activity_scale": [0.1, 2.0],      # Activity multiplier
        "k_d": [0.0, 5e-3],                 # Deactivation rate [1/s]
        "tau_probe": [0.0, 30.0],           # Probe time constant [s]
        "pH_offset": [-0.1, 0.1],           # pH measurement offset
    },
    
    # Fixed parameters
    "fixed_params": {
        "volume_L": 0.2,  # Fixed volume
    },
    
    # Measurement model parameters
    "measurement_model": {
        "add_noise": True,
        "noise_std": 0.01,          # pH measurement noise std [pH units]
        "use_probe_lag": True,      # Apply probe time constant
        "use_offset": True,         # Apply pH offset
    },
    
    # Parallel processing
    "n_workers": 8,
}

N0 = 0.0
C0 = 0.0


def sample_parameters(n_samples: int, seed: int = 42) -> np.ndarray:
    """
    Sample parameter vectors for full trajectories.
    
    Returns
    -------
    params_dict: dict with keys for all parameters
    """
    rng = np.random.default_rng(seed)
    
    params = {}
    for param_name, (lo, hi) in CONFIG["param_ranges"].items():
        if param_name not in CONFIG.get("fixed_params", {}):
            params[param_name] = rng.uniform(lo, hi, n_samples)
        else:
            # Fixed value for all samples
            params[param_name] = np.full(n_samples, CONFIG["fixed_params"][param_name])
    
    return params


def apply_measurement_model(pH_true: np.ndarray, t_grid: np.ndarray, 
                           tau_probe: float, pH_offset: float, 
                           noise_std: float, use_probe_lag: bool, 
                           use_offset: bool, add_noise: bool) -> np.ndarray:
    """
    Apply measurement model to true pH trajectory.
    
    Parameters
    ----------
    pH_true: (n_times,) array of true pH values
    t_grid: (n_times,) array of time points
    tau_probe: probe time constant [s]
    pH_offset: pH measurement offset
    noise_std: standard deviation of measurement noise
    use_probe_lag: if True, apply probe lag
    use_offset: if True, apply offset
    add_noise: if True, add noise
    
    Returns
    -------
    pH_meas: (n_times,) array of measured pH values
    """
    pH_meas = pH_true.copy()
    
    # Apply probe lag (first-order filter)
    if use_probe_lag and tau_probe > 0.0:
        import math
        for i in range(1, len(t_grid)):
            dt = t_grid[i] - t_grid[i-1]
            a = math.exp(-dt / max(tau_probe, 1e-12))
            pH_meas[i] = a * pH_meas[i-1] + (1 - a) * pH_true[i]
    
    # Apply offset
    if use_offset:
        pH_meas = pH_meas + pH_offset
    
    # Add noise
    if add_noise and noise_std > 0.0:
        rng = np.random.default_rng()
        noise = rng.normal(0.0, noise_std, size=pH_meas.shape)
        pH_meas = pH_meas + noise
    
    return pH_meas


def extract_prefix(pH_meas: np.ndarray, t_grid: np.ndarray, 
                   prefix_length: float, n_points: int) -> tuple:
    """
    Extract prefix of specified length from measured pH trajectory.
    
    Parameters
    ----------
    pH_meas: (n_times,) array of measured pH
    t_grid: (n_times,) array of time points
    prefix_length: length of prefix to extract [s]
    n_points: number of points to sample from prefix
    
    Returns
    -------
    t_prefix: (n_points,) array of time points
    pH_prefix: (n_points,) array of pH values
    """
    # Find indices within prefix_length
    mask = t_grid <= prefix_length
    if not np.any(mask):
        # If no points within prefix, use first point
        t_prefix = np.array([t_grid[0]])
        pH_prefix = np.array([pH_meas[0]])
    else:
        t_prefix_full = t_grid[mask]
        pH_prefix_full = pH_meas[mask]
        
        # Sample n_points uniformly from prefix
        if len(t_prefix_full) <= n_points:
            t_prefix = t_prefix_full
            pH_prefix = pH_prefix_full
        else:
            indices = np.linspace(0, len(t_prefix_full) - 1, n_points, dtype=int)
            t_prefix = t_prefix_full[indices]
            pH_prefix = pH_prefix_full[indices]
    
    return t_prefix, pH_prefix


def _worker_generate_single_trajectory(args_tuple):
    """
    Worker function for parallel trajectory generation.
    
    Returns: (index, result_dict, error)
    """
    index, params_dict, t_grid, constants = args_tuple
    
    try:
        # Extract physical parameters
        substrate_mM = params_dict["substrate_mM"]
        grams_urease_powder = params_dict["grams_urease_powder"]
        temperature_C = params_dict["temperature_C"]
        initial_pH_i = params_dict["initial_pH"]
        powder_activity_frac = params_dict["powder_activity_frac"]
        volume_L = params_dict["volume_L"]
        
        # Extract latent parameters
        activity_scale = params_dict["activity_scale"]
        k_d = params_dict["k_d"]
        tau_probe = params_dict["tau_probe"]
        pH_offset = params_dict["pH_offset"]
        
        # Convert to physical quantities
        S0 = substrate_mM / 1000.0  # mM → M
        T_K = temperature_C + 273.15
        E_loading_base_g_per_L = grams_urease_powder * powder_activity_frac / volume_L
        
        # Build simulator
        sim = UreaseSimulator(
            S0=S0,
            N0=N0,
            C0=C0,
            Pt_total_M=0.0,
            T_K=T_K,
            initial_pH=initial_pH_i,
            E_loading_base_g_per_L=E_loading_base_g_per_L,
            use_T_dependent_pH_activity=True,
        )
        
        # Parameters for ODE solver
        params = {
            "a": activity_scale,  # Activity scaling
            "k_d": k_d,
            "t_shift": 0.0,
            "tau_probe": tau_probe,
        }
        
        # Simulate forward to get pH
        pH_true = sim.simulate_forward(params, t_grid, return_totals=False, apply_probe_lag=False)
        
        # Apply measurement model
        mm = CONFIG["measurement_model"]
        pH_meas = apply_measurement_model(
            pH_true, t_grid, tau_probe, pH_offset,
            mm["noise_std"], mm["use_probe_lag"], mm["use_offset"], mm["add_noise"]
        )
        
        # Extract prefixes
        prefix_data = {}
        for prefix_length in CONFIG["prefix_lengths"]:
            t_prefix, pH_prefix = extract_prefix(
                pH_meas, t_grid, prefix_length, CONFIG["prefix_n_points"]
            )
            prefix_data[prefix_length] = {
                "t": t_prefix,
                "pH": pH_prefix,
            }
        
        # Known inputs (for model input)
        known_inputs = {
            "substrate_mM": substrate_mM,
            "grams_urease_powder": grams_urease_powder,
            "temperature_C": temperature_C,
            "initial_pH": initial_pH_i,
            "powder_activity_frac": powder_activity_frac,
            "volume_L": volume_L,
        }
        
        # Target parameters (to infer)
        target_params = {}
        for param_name in CONFIG["infer_params"]:
            target_params[param_name] = params_dict[param_name]
        
        result = {
            "pH_true": pH_true,
            "pH_meas": pH_meas,
            "prefix_data": prefix_data,
            "known_inputs": known_inputs,
            "target_params": target_params,
        }
        
        return (index, result, None)
        
    except Exception as e:
        return (index, None, str(e))


def generate_trajectories(params_dict: dict, t_grid: np.ndarray, n_workers: int = None) -> list:
    """
    Generate trajectories for given parameter samples using parallel processing.
    
    Returns
    -------
    results: list of result dictionaries (one per sample)
    """
    n_samples = len(params_dict[list(params_dict.keys())[0]])
    
    if n_workers is None:
        n_workers = mp.cpu_count()
    n_workers = min(n_workers, n_samples)
    
    print(f"Using {n_workers} worker processes for parallel generation...")
    
    # Prepare arguments
    constants = ()  # No constants needed for now
    worker_args = [
        (i, {k: v[i] for k, v in params_dict.items()}, t_grid, constants)
        for i in range(n_samples)
    ]
    
    results = [None] * n_samples
    failed_indices = []
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_index = {
            executor.submit(_worker_generate_single_trajectory, args): args[0]
            for args in worker_args
        }
        
        for future in tqdm(as_completed(future_to_index), total=n_samples, desc="Generating trajectories"):
            index = future_to_index[future]
            try:
                result_index, result, error = future.result()
                
                if result_index != index:
                    raise RuntimeError(f"Index mismatch! Expected {index}, got {result_index}")
                
                if error is not None:
                    print(f"\nWarning: Failed at sample {index}: {error}")
                    failed_indices.append(index)
                else:
                    results[index] = result
                    
            except Exception as e:
                print(f"\nWarning: Exception processing sample {index}: {e}")
                failed_indices.append(index)
    
    if failed_indices:
        print(f"\nFailed {len(failed_indices)} samples out of {n_samples}")
    
    return results


def main():
    """Generate and save early inference training dataset."""
    n_samples = CONFIG["n_samples"]
    t_max = CONFIG["t_max"]
    n_times = CONFIG["n_times"]
    seed = CONFIG["seed"]
    output_dir = Path(CONFIG["output_dir"])
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate time grid (uniform for full trajectory)
    t_grid = np.linspace(0.0, t_max, n_times)
    
    print("="*60)
    print("EARLY INFERENCE TRAINING DATA GENERATION")
    print("="*60)
    print(f"Configuration:")
    print(f"  • Number of samples: {n_samples:,}")
    print(f"  • Time grid: {t_max} s, {n_times} points")
    print(f"  • Prefix lengths: {CONFIG['prefix_lengths']} s")
    print(f"  • Parameters to infer: {CONFIG['infer_params']}")
    print(f"  • Output directory: {output_dir}")
    
    # Determine number of workers
    n_workers = CONFIG.get("n_workers", None)
    if n_workers is None:
        n_workers = mp.cpu_count()
        print(f"  • Using all available CPU cores: {n_workers}")
    else:
        print(f"  • Using {n_workers} CPU cores")
    
    print("\n" + "="*60)
    print("STEP 1: Sampling parameters")
    print("="*60)
    
    params_dict = sample_parameters(n_samples, seed=seed)
    print(f"✓ Sampled {n_samples:,} parameter vectors")
    
    print("\n" + "="*60)
    print("STEP 2: Generating trajectories")
    print("="*60)
    
    results = generate_trajectories(params_dict, t_grid, n_workers=n_workers)
    print(f"\n✓ Generated {len([r for r in results if r is not None]):,} trajectories")
    
    # Step 3: Organize data for training
    print("\n" + "="*60)
    print("STEP 3: Organizing data for training")
    print("="*60)
    
    # For each prefix length, create training examples
    training_data = {}
    for prefix_length in CONFIG["prefix_lengths"]:
        training_data[prefix_length] = {
            "pH_prefix": [],
            "t_prefix": [],
            "known_inputs": [],
            "target_params": [],
        }
    
    for result in results:
        if result is None:
            continue
        
        for prefix_length in CONFIG["prefix_lengths"]:
            prefix_info = result["prefix_data"][prefix_length]
            training_data[prefix_length]["pH_prefix"].append(prefix_info["pH"])
            training_data[prefix_length]["t_prefix"].append(prefix_info["t"])
            training_data[prefix_length]["known_inputs"].append(result["known_inputs"])
            training_data[prefix_length]["target_params"].append(result["target_params"])
    
    # Convert to arrays (pad sequences to fixed length)
    print("Converting to arrays and padding sequences...")
    processed_data = {}
    for prefix_length in CONFIG["prefix_lengths"]:
        n_examples = len(training_data[prefix_length]["pH_prefix"])
        
        # Find max sequence length
        max_len = max(len(seq) for seq in training_data[prefix_length]["pH_prefix"])
        max_len = max(max_len, CONFIG["prefix_n_points"])
        
        # Pad sequences
        pH_prefix_padded = []
        t_prefix_padded = []
        for i in range(n_examples):
            pH_seq = training_data[prefix_length]["pH_prefix"][i]
            t_seq = training_data[prefix_length]["t_prefix"][i]
            
            # Pad to max_len
            if len(pH_seq) < max_len:
                pad_len = max_len - len(pH_seq)
                pH_seq = np.pad(pH_seq, (0, pad_len), mode='edge')
                t_seq = np.pad(t_seq, (0, pad_len), mode='edge')
            
            pH_prefix_padded.append(pH_seq[:max_len])
            t_prefix_padded.append(t_seq[:max_len])
        
        # Convert known inputs to array
        known_inputs_array = []
        for ki in training_data[prefix_length]["known_inputs"]:
            known_inputs_array.append([
                ki["substrate_mM"],
                ki["grams_urease_powder"],
                ki["temperature_C"],
                ki["initial_pH"],
                ki["powder_activity_frac"],
                ki["volume_L"],
            ])
        
        # Convert target params to array
        target_params_array = []
        for tp in training_data[prefix_length]["target_params"]:
            target_params_array.append([
                tp[param_name] for param_name in CONFIG["infer_params"]
            ])
        
        processed_data[prefix_length] = {
            "pH_prefix": np.array(pH_prefix_padded),  # (n_examples, max_len)
            "t_prefix": np.array(t_prefix_padded),   # (n_examples, max_len)
            "known_inputs": np.array(known_inputs_array),  # (n_examples, 6)
            "target_params": np.array(target_params_array),  # (n_examples, n_infer_params)
        }
    
    # Step 4: Save data
    print("\n" + "="*60)
    print("STEP 4: Saving data to disk")
    print("="*60)
    
    # Save as NPZ
    print("  → Saving training_data.npz...")
    np.savez_compressed(
        output_dir / "training_data.npz",
        **{f"prefix_{int(pl)}s": processed_data[pl] for pl in CONFIG["prefix_lengths"]}
    )
    print("  ✓ training_data.npz saved")
    
    # Save metadata
    print("  → Saving metadata.json...")
    metadata = {
        "n_samples": n_samples,
        "t_max": t_max,
        "n_times": n_times,
        "t_grid": t_grid.tolist(),
        "prefix_lengths": CONFIG["prefix_lengths"],
        "prefix_n_points": CONFIG["prefix_n_points"],
        "infer_params": CONFIG["infer_params"],
        "param_ranges": CONFIG["param_ranges"],
        "fixed_params": CONFIG.get("fixed_params", {}),
        "measurement_model": CONFIG["measurement_model"],
        "known_input_names": [
            "substrate_mM",
            "grams_urease_powder",
            "temperature_C",
            "initial_pH",
            "powder_activity_frac",
            "volume_L",
        ],
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("  ✓ metadata.json saved")
    
    # Print summary
    print("\n" + "="*60)
    print("✓ DATA GENERATION COMPLETE!")
    print("="*60)
    print(f"All files saved to: {output_dir.resolve()}")
    print(f"\nGenerated files:")
    print(f"  • training_data.npz - Main dataset")
    print(f"  • metadata.json - Dataset metadata")
    print(f"\nDataset statistics:")
    for prefix_length in CONFIG["prefix_lengths"]:
        n_examples = processed_data[prefix_length]["target_params"].shape[0]
        print(f"  • Prefix {prefix_length}s: {n_examples:,} examples")


if __name__ == "__main__":
    main()
