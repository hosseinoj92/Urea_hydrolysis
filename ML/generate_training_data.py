"""
Generate synthetic training dataset for DeepONet by sampling parameter ranges
around the experimental regime (40°C batch).

This version uses physically meaningful knobs consistent with simulation5_batch:
- Substrate concentration (urea) in mM
- Grams of urease powder (converted to activity multiplier a)
- Temperature in °C (with T-dependent pH-activity)
- Optional enzyme deactivation rate k_d

E_eff0, t_shift and tau_probe are held fixed; variation comes from:
    (substrate_mM, grams_urease_powder, temperature_C, k_d)
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

# Default conditions matching 40°C batch experiments / simulation5_batch
VOLUME_L = 0.2
POWDER_ACTIVITY_FRAC = 1.0
GRAMS_UREASE_POWDER_REF = 0.10  # reference enzyme mass used in original 40 °C runs

N0 = 0.0
C0 = 0.0

# Default initial pH (used if not overridden by sampled initial_pH values)
INITIAL_PH_DEFAULT = 7.0

# From reference: E_loading_base = grams_ref * frac / volume
E_LOADING_BASE_G_PER_L = GRAMS_UREASE_POWDER_REF * POWDER_ACTIVITY_FRAC / VOLUME_L

# Fixed nuisance parameters for this first dataset
FIXED_T_SHIFT_S = 0.0
FIXED_TAU_PROBE_S = 0.0

# ╔══════════════════════════════════════════════════════════════╗
# ║                       USER CONFIG                             ║
# ╚══════════════════════════════════════════════════════════════╝
CONFIG = {
    # Dataset generation parameters
    "n_samples": 200000,           # Number of parameter samples to generate
    "t_max": 18000.0,              # Maximum time [s] for trajectories
    "n_times": 2000,                # Number of time points in trajectory
    "seed": 42,                    # Random seed for reproducibility
    "output_dir": "Generated_Data_5params",          # Output directory for generated data
    "use_totals": True,            # If True, output (S, Ntot, Ctot); else pH only
    
    # Parallel processing
    "n_workers": 8,             # Number of CPU cores to use (None = use all available)
    # Note: Intel Xeon W-2245 has 8 cores (16 threads), but for CPU-bound tasks
    # using 8 workers (one per physical core) is optimal
    
    # Parameter sampling ranges (uniform sampling)
    "param_ranges": {
        "substrate_mM": [1.0, 100.0],        # Urea concentration [mM]
        "grams_urease_powder": [0.01, 6.0],    # Enzyme powder mass [g]
        "temperature_C": [20.0, 40.0],         # Temperature [°C]
        "k_d": [0.0, 5e-3],                    # Deactivation rate [1/s]
        "initial_pH": [6.5, 7.5],              # Initial pH range
    },
}


def sample_parameters(n_samples: int, seed: int = 42) -> np.ndarray:
    """
    Sample parameter vectors from realistic ranges (uniform sampling).

    Parameters (columns of returned array):
    - substrate_mM: initial urea concentration [mM]  (1–1000 mM)
    - grams_urease_powder: total urease powder mass [g]  (0.01–1.0 g)
    - temperature_C: isothermal temperature [°C]  (20–40 °C; default grid in batch code)
    - k_d: deactivation rate [1/s]  (0–5e-3)

    Returns
    -------
    params_array: (n_samples, 5) array
    """
    rng = np.random.default_rng(seed)
    
    # Get ranges from CONFIG
    ranges = CONFIG["param_ranges"]

    # Substrate in mM
    substrate_mM = rng.uniform(ranges["substrate_mM"][0], ranges["substrate_mM"][1], n_samples)

    # Grams of urease powder
    grams_urease_powder = rng.uniform(ranges["grams_urease_powder"][0], ranges["grams_urease_powder"][1], n_samples)

    # Temperature in °C
    temperature_C = rng.uniform(ranges["temperature_C"][0], ranges["temperature_C"][1], n_samples)

    # First-order deactivation rate
    k_d = rng.uniform(ranges["k_d"][0], ranges["k_d"][1], n_samples)

    # Initial pH
    initial_pH_samples = rng.uniform(ranges["initial_pH"][0], ranges["initial_pH"][1], n_samples)

    return np.column_stack([substrate_mM, grams_urease_powder, temperature_C, k_d, initial_pH_samples])


def _worker_generate_single_trajectory(args_tuple):
    """
    Worker function for parallel trajectory generation.
    
    This function is called by each worker process. It takes a tuple containing:
    - index: sample index (to maintain correct ordering)
    - params_row: single parameter vector [substrate_mM, grams_urease_powder, temperature_C, k_d]
    - t_grid: time grid array
    - return_totals: boolean flag
    - global constants: N0, C0, initial_pH, E_LOADING_BASE_G_PER_L, GRAMS_UREASE_POWDER_REF, FIXED_T_SHIFT_S, FIXED_TAU_PROBE_S
    
    Returns:
    -------
    (index, result): tuple where result is either (S, Ntot, Ctot) or pH array
    """
    index, params_row, t_grid, return_totals, constants = args_tuple
    
    # Unpack constants
    N0, C0, E_LOADING_BASE_G_PER_L, GRAMS_UREASE_POWDER_REF, FIXED_T_SHIFT_S, FIXED_TAU_PROBE_S = constants
    
    substrate_mM, grams_urease_powder, temperature_C, k_d, initial_pH_i = params_row

    # Convert to physical quantities used in the simulator
    S0 = substrate_mM / 1000.0  # mM → M
    T_K = temperature_C + 273.15

    # Activity multiplier a, defined relative to reference grams
    a = grams_urease_powder / GRAMS_UREASE_POWDER_REF

    # Build a simulator instance for this (S0, T_K) with T-dependent pH-activity
    sim = UreaseSimulator(
        S0=S0,
        N0=N0,
        C0=C0,
        Pt_total_M=0.0,
        T_K=T_K,
        initial_pH=initial_pH_i,
        E_loading_base_g_per_L=E_LOADING_BASE_G_PER_L,
        use_T_dependent_pH_activity=True,
    )

    # Parameters passed to the ODE solver
    params = {
        "a": a,
        "k_d": k_d,
        "t_shift": FIXED_T_SHIFT_S,
        "tau_probe": FIXED_TAU_PROBE_S,
    }

    try:
        if return_totals:
            S, Ntot, Ctot = sim.simulate_forward(
                params, t_grid, return_totals=True
            )
            result = np.column_stack([S, Ntot, Ctot])  # (n_times, 3)
        else:
            pH = sim.simulate_forward(
                params, t_grid, return_totals=False
            )
            result = pH.reshape(-1, 1)  # (n_times, 1)
        
        return (index, result, None)  # (index, result, error)
    except Exception as e:
        # Return error info so main process can handle it
        return (index, None, str(e))


def generate_trajectories(
    params_array: np.ndarray,
    t_grid: np.ndarray,
    return_totals: bool = True,
    n_workers: int = None,
) -> np.ndarray:
    """
    Generate trajectories for given parameter samples using parallel processing.

    Parameters
    ----------
    params_array : (n_samples, 4) array
        Columns: [substrate_mM, grams_urease_powder, temperature_C, k_d]
    t_grid : array_like
        Time points [s]
    return_totals : bool, default True
        If True, return (S, Ntot, Ctot); else return pH.
    n_workers : int, optional
        Number of worker processes. If None, uses all available CPU cores.

    Returns
    -------
    outputs : (n_samples, n_times, n_outputs) array
        If return_totals:
            outputs[:, :, 0] = S, outputs[:, :, 1] = Ntot, outputs[:, :, 2] = Ctot
        Else:
            outputs[:, :, 0] = pH
        
    Note: outputs[i] corresponds EXACTLY to params_array[i] regardless of which
    worker processed it. Order is preserved via index tracking.
    """
    n_samples = params_array.shape[0]
    n_times = len(t_grid)
    
    # Determine number of workers
    if n_workers is None:
        n_workers = mp.cpu_count()
    n_workers = min(n_workers, n_samples)  # Don't use more workers than samples
    
    print(f"Using {n_workers} worker processes for parallel generation...")

    if return_totals:
        outputs = np.zeros((n_samples, n_times, 3))
    else:
        outputs = np.zeros((n_samples, n_times, 1))

    # Prepare constants tuple for worker processes
    # (These are passed to avoid pickling issues with global variables)
    constants = (
        N0,
        C0,
        E_LOADING_BASE_G_PER_L,
        GRAMS_UREASE_POWDER_REF,
        FIXED_T_SHIFT_S,
        FIXED_TAU_PROBE_S,
    )

    # Prepare arguments for each worker: (index, params_row, t_grid, return_totals, constants)
    worker_args = [
        (i, params_array[i, :], t_grid, return_totals, constants)
        for i in range(n_samples)
    ]

    failed_indices = []
    
    # Use ProcessPoolExecutor for parallel processing
    # This ensures correct ordering via index tracking
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(_worker_generate_single_trajectory, args): args[0]
            for args in worker_args
        }
        
        # Process completed tasks with tqdm progress bar
        # tqdm automatically handles multiprocessing progress tracking
        for future in tqdm(as_completed(future_to_index), total=n_samples, desc="Generating trajectories"):
            index = future_to_index[future]
            try:
                result_index, result, error = future.result()
                
                # CRITICAL: Verify index matches (safety check)
                if result_index != index:
                    raise RuntimeError(f"Index mismatch! Expected {index}, got {result_index}")
                
                if error is not None:
                    print(f"\nWarning: Failed at sample {index}: {error}")
                    failed_indices.append(index)
                    # Fill with NaN
                    outputs[index] = np.nan
                else:
                    # Store result at correct index
                    outputs[index] = result
                    
            except Exception as e:
                print(f"\nWarning: Exception processing sample {index}: {e}")
                failed_indices.append(index)
                outputs[index] = np.nan

    if failed_indices:
        print(f"\nFailed {len(failed_indices)} samples out of {n_samples}")
        print(f"Failed indices: {failed_indices[:10]}..." if len(failed_indices) > 10 else f"Failed indices: {failed_indices}")

    return outputs


def _worker_convert_totals_to_pH(args_tuple):
    """
    Worker function for parallel pH conversion from totals.
    
    Parameters:
    -----------
    args_tuple: (index, Ntot_row, Ctot_row, initial_pH_i, constants)
        - index: sample index
        - Ntot_row: (n_times,) array of total ammonia
        - Ctot_row: (n_times,) array of total carbon
        - constants: (N0, C0, initial_pH, E_LOADING_BASE_G_PER_L, GRAMS_UREASE_POWDER_REF)
    
    Returns:
    --------
    (index, pH_row): tuple with sample index and pH array
    """
    index, Ntot_row, Ctot_row, initial_pH_i, constants = args_tuple
    
    # Unpack constants
    N0, C0, E_LOADING_BASE_G_PER_L, GRAMS_UREASE_POWDER_REF = constants
    
    # Create simulator for speciation (only needs to compute pH, not kinetics)
    sim = UreaseSimulator(
        S0=0.0,
        N0=N0,
        C0=C0,
        Pt_total_M=0.0,
        T_K=313.15,  # reference; speciation is not very T-sensitive over 20–40 °C
        initial_pH=initial_pH_i,
        E_loading_base_g_per_L=E_LOADING_BASE_G_PER_L,
        use_T_dependent_pH_activity=True,
    )
    
    n_times = len(Ntot_row)
    pH_row = np.zeros(n_times)
    
    for j in range(n_times):
        sp = sim.compute_speciation(Ntot_row[j], Ctot_row[j], 0.0)
        pH_row[j] = sp['pH']
    
    return (index, pH_row)


def convert_totals_to_pH(
    S: np.ndarray,
    Ntot: np.ndarray,
    Ctot: np.ndarray,
    initial_pH_array: np.ndarray = None,
    n_workers: int = None,
) -> np.ndarray:
    """
    Convert totals (S, Ntot, Ctot) to pH using the same equilibrium routine.
    This ensures physical consistency in the hybrid surrogate.
    
    Uses parallel processing for faster conversion.
    """
    n_samples, n_times = S.shape
    
    # Determine number of workers
    if n_workers is None:
        n_workers = mp.cpu_count()
    n_workers = min(n_workers, n_samples)
    
    print(f"Converting totals to pH using {n_workers} workers...")
    
    # Prepare constants
    constants = (
        N0,
        C0,
        E_LOADING_BASE_G_PER_L,
        GRAMS_UREASE_POWDER_REF,
    )
    
    # If no per-sample initial pH is provided, use the default for all
    if initial_pH_array is None:
        initial_pH_array = np.full(n_samples, INITIAL_PH_DEFAULT, dtype=float)
    
    # Prepare arguments for each worker
    worker_args = [
        (i, Ntot[i, :], Ctot[i, :], float(initial_pH_array[i]), constants)
        for i in range(n_samples)
    ]
    
    pH = np.zeros((n_samples, n_times))
    
    # Use parallel processing for pH conversion
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_index = {
            executor.submit(_worker_convert_totals_to_pH, args): args[0]
            for args in worker_args
        }
        
        # Process with progress bar
        for future in tqdm(as_completed(future_to_index), total=n_samples, desc="Converting totals → pH"):
            index = future_to_index[future]
            try:
                result_index, pH_row = future.result()
                
                # Verify index matches
                if result_index != index:
                    raise RuntimeError(f"Index mismatch in pH conversion! Expected {index}, got {result_index}")
                
                pH[index, :] = pH_row
                
            except Exception as e:
                print(f"\nWarning: Exception converting pH for sample {index}: {e}")
                pH[index, :] = np.nan
    
    return pH


def main():
    """Generate and save training dataset."""
    # Use CONFIG dictionary for all parameters
    n_samples = CONFIG["n_samples"]
    t_max = CONFIG["t_max"]
    n_times = CONFIG["n_times"]
    seed = CONFIG["seed"]
    output_dir = Path(CONFIG["output_dir"])
    use_totals = CONFIG["use_totals"]
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Time grid
    t_grid = np.linspace(0.0, t_max, n_times)
    
    print("="*60)
    print("DEEPONET TRAINING DATA GENERATION")
    print("="*60)
    print(f"Configuration:")
    print(f"  • Number of samples: {n_samples:,}")
    print(f"  • Time grid: {t_max} s ({t_max/3600:.1f} hours), {n_times} points")
    print(f"  • Output mode: {'totals (S, Ntot, Ctot)' if use_totals else 'pH'}")
    print(f"  • Output directory: {output_dir}")
    
    # Determine number of workers
    n_workers = CONFIG.get("n_workers", None)
    if n_workers is None:
        n_workers = mp.cpu_count()
        print(f"  • Using all available CPU cores: {n_workers}")
    else:
        print(f"  • Using {n_workers} CPU cores (configured in CONFIG)")
    
    print("\n" + "="*60)
    print("STEP 1: Sampling parameters")
    print("="*60)
    print("Generating random parameter vectors from specified ranges...")
    params_array = sample_parameters(n_samples, seed=seed)
    print(f"✓ Sampled {n_samples:,} parameter vectors")
    print(f"  Parameter ranges:")
    for name, (lo, hi) in CONFIG["param_ranges"].items():
        print(f"    • {name}: [{lo}, {hi}]")
    
    print("\n" + "="*60)
    print("STEP 2: Generating trajectories (ODE integration)")
    print("="*60)
    print("This step runs the mechanistic simulator for each parameter set.")
    print("Each simulation integrates ODEs to compute (S, Ntot, Ctot) over time.")
    print("This is the most time-consuming step (~7-8 hours for 100k samples).\n")
    
    outputs = generate_trajectories(
        params_array,
        t_grid,
        return_totals=use_totals,
        n_workers=n_workers
    )
    print(f"\n✓ Generated {n_samples:,} trajectories")
    
    # Step 3: Convert totals to pH for validation (with progress bar)
    if use_totals:
        print("\n" + "="*60)
        print("STEP 3: Converting totals to pH for validation")
        print("="*60)
        print("This step converts (S, Ntot, Ctot) → pH using the same")
        print("equilibrium/charge-balance routine as the mechanistic model.")
        print("This ensures physical consistency in the hybrid surrogate.\n")
        
        # Extract per-sample initial pH from params_array (last column)
        initial_pH_array = params_array[:, 4]
        
        pH_from_totals = convert_totals_to_pH(
            outputs[:, :, 0],  # S
            outputs[:, :, 1],  # Ntot
            outputs[:, :, 2],  # Ctot
            initial_pH_array=initial_pH_array,
            n_workers=n_workers  # Use same number of workers
        )
    else:
        print("\n" + "="*60)
        print("STEP 3: Using pH directly (no conversion needed)")
        print("="*60)
        pH_from_totals = outputs[:, :, 0]
    
    # Step 4: Save data (with progress indicators)
    print("\n" + "="*60)
    print("STEP 4: Saving data to disk")
    print("="*60)
    print(f"Saving to: {output_dir}")
    
    # Save as compressed NPZ (this can take a while for large datasets)
    print("  → Saving training_data.npz (compressed)...")
    with tqdm(total=1, desc="Saving NPZ file") as pbar:
        np.savez_compressed(
            output_dir / "training_data.npz",
            params=params_array,
            outputs=outputs,
            t_grid=t_grid,
            pH=pH_from_totals
        )
        pbar.update(1)
    print("  ✓ training_data.npz saved")
    
    # Save metadata
    print("  → Saving metadata.json...")
    metadata = {
        "n_samples": n_samples,
        "n_times": n_times,
        "t_max": t_max,
        "t_grid": t_grid.tolist(),
        "N0": N0,
        "C0": C0,
        "initial_pH_default": INITIAL_PH_DEFAULT,
        "volume_L": VOLUME_L,
        "powder_activity_frac": POWDER_ACTIVITY_FRAC,
        "grams_urease_powder_ref": GRAMS_UREASE_POWDER_REF,
        "E_loading_base_g_per_L": E_LOADING_BASE_G_PER_L,
        "use_totals": use_totals,
        "param_names": [
            "substrate_mM",
            "grams_urease_powder",
            "temperature_C",
            "k_d",
            "initial_pH",
        ],
        "param_ranges": CONFIG["param_ranges"],
        "fixed_params": {
            "t_shift_s": FIXED_T_SHIFT_S,
            "tau_probe_s": FIXED_TAU_PROBE_S,
        },
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("  ✓ metadata.json saved")
    
    # Save a sample as CSV for inspection
    print("  → Saving sample_params.csv (first 100 samples)...")
    sample_df = pd.DataFrame({
        "substrate_mM": params_array[:100, 0],
        "grams_urease_powder": params_array[:100, 1],
        "temperature_C": params_array[:100, 2],
        "k_d": params_array[:100, 3],
        "initial_pH": params_array[:100, 4],
    })
    sample_df.to_csv(output_dir / "sample_params.csv", index=False)
    print("  ✓ sample_params.csv saved")
    
    # Final summary
    print("\n" + "="*60)
    print("✓ DATA GENERATION COMPLETE!")
    print("="*60)
    print(f"All files saved to: {output_dir.resolve()}")
    print(f"\nGenerated files:")
    print(f"  • training_data.npz - Main dataset (params, outputs, t_grid, pH)")
    print(f"  • metadata.json - Dataset metadata and configuration")
    print(f"  • sample_params.csv - First 100 parameter samples for inspection")
    print(f"\nDataset statistics:")
    print(f"  • Parameters shape: {params_array.shape}")
    print(f"  • Outputs shape: {outputs.shape}")
    print(f"  • pH range: [{pH_from_totals.min():.3f}, {pH_from_totals.max():.3f}]")
    print(f"  • Estimated file size: ~{outputs.nbytes / 1e9:.2f} GB (uncompressed)")


if __name__ == "__main__":
    main()
