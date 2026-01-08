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
# Note: volume_L and powder_activity_frac are now variable parameters in CONFIG
# These defaults are only used if not specified in CONFIG
VOLUME_L_DEFAULT = 0.2
POWDER_ACTIVITY_FRAC_DEFAULT = 1.0

N0 = 0.0
C0 = 0.0

# Default initial pH (used if not overridden by sampled initial_pH values)
INITIAL_PH_DEFAULT = 7.0

# Fixed nuisance parameters for this first dataset
FIXED_T_SHIFT_S = 0.0
FIXED_TAU_PROBE_S = 0.0

# ╔══════════════════════════════════════════════════════════════╗
# ║                       USER CONFIG                             ║
# ╚══════════════════════════════════════════════════════════════╝
CONFIG = {
    # Dataset generation parameters
    "n_samples": 50000,           # Number of parameter samples to generate
    "t_max": 1000.0,              # Maximum time [s] for trajectories (e.g., 200 s for short runs)
    "n_times": 500,                # Number of time points in trajectory (e.g., 1000 points for 200 s = 0.2 s resolution)
    "seed": 42,                    # Random seed for reproducibility
    "output_dir": "Generated_Data_6params_50000",          # Output directory for generated data
    "use_totals": True,            # If True, output (S, Ntot, Ctot); else pH only
    
    # Time grid configuration
    # Options: "uniform", "concentrated", "piecewise"
    # - "uniform": Uniformly spaced time points (default, current behavior)
    # - "concentrated": Power-law distribution with more points at the beginning
    #   (useful when most dynamics happen early in the reaction)
    # - "piecewise": Specify early/late windows with different point densities
    "time_grid_mode": "concentrated",   # "uniform", "concentrated", or "piecewise"
    
    # Parameters for "concentrated" mode:
    # Time points are distributed as: t = t_max * (u^alpha) where u is uniform in [0,1]
    # - alpha < 1: concentrates points at beginning (smaller = more concentration)
    # - alpha = 1: uniform spacing
    # - alpha = 0.5: square-root distribution (good default for early dynamics)
    "time_grid_alpha": 0.4,        # Power-law exponent (0 < alpha <= 1)
    
    # Parameters for "piecewise" mode:
    # Divides time into early window (0 to t_early) and late window (t_early to t_max)
    # with different point densities
    "time_grid_t_early": 200.0,    # Early time window boundary [s] (where most dynamics occur)
    "time_grid_n_early": 400,      # Number of points in early window (0 to t_early)
    "time_grid_n_late": 100,       # Number of points in late window (t_early to t_max)
    
    # Parallel processing
    "n_workers": 8,             # Number of CPU cores to use (None = use all available)
    # Note: Intel Xeon W-2245 has 8 cores (16 threads), but for CPU-bound tasks
    # using 8 workers (one per physical core) is optimal
    
    # Parameter configuration: specify which parameters are VARIABLE vs FIXED
    # Variable parameters will be sampled from ranges; fixed parameters use constant values
    "variable_params": [
        "substrate_mM",
        "grams_urease_powder", 
        "temperature_C",
        "k_d",
        "initial_pH",
        "powder_activity_frac",
    ],  # List of parameter names to vary (order determines column order in saved array)
    
    # Parameter sampling ranges (only used for variable parameters)
    "param_ranges": {
        "substrate_mM": [1.0, 100.0],        # Urea concentration [mM]
        "grams_urease_powder": [0.01, 0.5],    # Enzyme powder mass [g]
        "temperature_C": [20.0, 40.0],         # Temperature [°C]
        "k_d": [0.0, 5e-3],                    # Deactivation rate [1/s]
        "initial_pH": [6.5, 7.5],              # Initial pH range
        "powder_activity_frac": [0.01, 0.5],    # Fraction of powder that is active (0-1)
        "volume_L": [0.1, 1.0],                # Volume of solution [L]
    },
    
    # Fixed parameter values (only used if parameter is NOT in variable_params)
    "fixed_params": {
        # Example: if you want to fix some parameters, uncomment and set values:
        # "substrate_mM": 20.0,              # Fixed at 20 mM
        # "temperature_C": 40.0,             # Fixed at 40°C
        # "powder_activity_frac": 1.0,       # Fixed at 100% active
         "volume_L": 0.2,                   # Fixed at 0.2 L
    },
}


def generate_time_grid(t_max: float, n_times: int, mode: str = "uniform", 
                       alpha: float = 0.5, t_early: float = None, 
                       n_early: int = None, n_late: int = None) -> np.ndarray:
    """
    Generate time grid with optional non-uniform spacing.
    
    Parameters
    ----------
    t_max : float
        Maximum time [s]
    n_times : int
        Total number of time points
    mode : str, default "uniform"
        Time grid mode: "uniform", "concentrated", or "piecewise"
    alpha : float, default 0.5
        Power-law exponent for "concentrated" mode (0 < alpha <= 1)
        - alpha < 1: concentrates points at beginning
        - alpha = 1: uniform spacing
        - alpha = 0.5: square-root distribution (good default)
    t_early : float, optional
        Early time window boundary [s] for "piecewise" mode
    n_early : int, optional
        Number of points in early window for "piecewise" mode
    n_late : int, optional
        Number of points in late window for "piecewise" mode
    
    Returns
    -------
    t_grid : (n_times,) array
        Time points [s], always includes t=0 and t=t_max
    """
    if mode == "uniform":
        # Uniform spacing (current default behavior)
        t_grid = np.linspace(0.0, t_max, n_times)
        
    elif mode == "concentrated":
        # Power-law distribution: t = t_max * (u^alpha) where u is uniform in [0,1]
        # This concentrates points at the beginning when alpha < 1
        if not (0 < alpha <= 1):
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        
        # Generate uniform samples in [0, 1]
        u = np.linspace(0.0, 1.0, n_times)
        # Apply power-law transformation
        t_grid = t_max * (u ** alpha)
        # Ensure first point is exactly 0 and last is exactly t_max
        t_grid[0] = 0.0
        t_grid[-1] = t_max
        
    elif mode == "piecewise":
        # Piecewise: early window gets more points, late window gets fewer
        if t_early is None or n_early is None or n_late is None:
            raise ValueError("piecewise mode requires t_early, n_early, and n_late")
        if t_early <= 0 or t_early >= t_max:
            raise ValueError(f"t_early must be in (0, t_max), got {t_early}")
        if n_early + n_late != n_times:
            raise ValueError(f"n_early + n_late must equal n_times, got {n_early} + {n_late} = {n_early + n_late} != {n_times}")
        
        # Early window: uniform spacing from 0 to t_early
        t_early_grid = np.linspace(0.0, t_early, n_early)
        # Late window: uniform spacing from t_early to t_max
        t_late_grid = np.linspace(t_early, t_max, n_late + 1)[1:]  # Skip t_early (already in early grid)
        # Combine
        t_grid = np.concatenate([t_early_grid, t_late_grid])
        # Ensure first point is exactly 0 and last is exactly t_max
        t_grid[0] = 0.0
        t_grid[-1] = t_max
        
    else:
        raise ValueError(f"Unknown time_grid_mode: {mode}. Must be 'uniform', 'concentrated', or 'piecewise'")
    
    return t_grid


def sample_parameters(n_samples: int, seed: int = 42) -> np.ndarray:
    """
    Sample parameter vectors from realistic ranges (uniform sampling).
    Only samples VARIABLE parameters as specified in CONFIG["variable_params"].
    
    Returns
    -------
    params_array: (n_samples, n_variable_params) array
        Columns correspond to CONFIG["variable_params"] in order
    """
    rng = np.random.default_rng(seed)
    
    # Get variable parameter names and ranges
    variable_params = CONFIG["variable_params"]
    ranges = CONFIG["param_ranges"]
    
    # Validate that all variable params have ranges defined
    missing = [p for p in variable_params if p not in ranges]
    if missing:
        raise ValueError(f"Missing param_ranges for variable parameters: {missing}")
    
    # Sample each variable parameter
    sampled_params = []
    for param_name in variable_params:
        if param_name not in ranges:
            raise ValueError(f"Parameter '{param_name}' in variable_params but no range defined in param_ranges")
        lo, hi = ranges[param_name]
        sampled = rng.uniform(lo, hi, n_samples)
        sampled_params.append(sampled)
    
    # Stack into array: (n_samples, n_variable_params)
    return np.column_stack(sampled_params)


def _worker_generate_single_trajectory(args_tuple):
    """
    Worker function for parallel trajectory generation.
    
    This function is called by each worker process. It takes a tuple containing:
    - index: sample index (to maintain correct ordering)
    - params_row: single parameter vector (only variable parameters, in order of variable_params)
    - t_grid: time grid array
    - return_totals: boolean flag
    - constants: (N0, C0, FIXED_T_SHIFT_S, FIXED_TAU_PROBE_S, variable_params, fixed_params)
    
    Returns:
    -------
    (index, result): tuple where result is either (S, Ntot, Ctot) or pH array
    """
    index, params_row, t_grid, return_totals, constants = args_tuple
    
    # Unpack constants
    N0, C0, FIXED_T_SHIFT_S, FIXED_TAU_PROBE_S, variable_params, fixed_params = constants
    
    # Reconstruct full parameter set from variable + fixed
    # Create a dict with all parameters
    all_params = fixed_params.copy()  # Start with fixed values
    
    # Add variable parameters (in order of variable_params list)
    for i, param_name in enumerate(variable_params):
        all_params[param_name] = params_row[i]
    
    # Extract individual parameters (with defaults if missing)
    substrate_mM = all_params.get("substrate_mM", 20.0)
    grams_urease_powder = all_params.get("grams_urease_powder", 0.10)
    temperature_C = all_params.get("temperature_C", 40.0)
    k_d = all_params.get("k_d", 0.0)
    initial_pH_i = all_params.get("initial_pH", 7.0)
    powder_activity_frac = all_params.get("powder_activity_frac", 1.0)
    volume_L = all_params.get("volume_L", 0.2)

    # Convert to physical quantities used in the simulator
    S0 = substrate_mM / 1000.0  # mM → M
    T_K = temperature_C + 273.15

    # Compute E_loading_base following notebook logic:
    # E_loading_base_g_per_L = grams_urease_powder * powder_activity_frac / volume_L
    E_loading_base_g_per_L = grams_urease_powder * powder_activity_frac / volume_L

    # Build a simulator instance for this (S0, T_K) with T-dependent pH-activity
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

    # Parameters passed to the ODE solver
    # activity_scale 'a' = 1.0 (we already account for activity via powder_activity_frac)
    params = {
        "a": 1.0,  # No additional scaling needed since activity is in powder_activity_frac
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
    params_array : (n_samples, n_variable_params) array
        Columns correspond to CONFIG["variable_params"] in order
        Only contains VARIABLE parameters; fixed parameters are handled internally
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
    variable_params = CONFIG["variable_params"]
    fixed_params = CONFIG.get("fixed_params", {})
    
    constants = (
        N0,
        C0,
        FIXED_T_SHIFT_S,
        FIXED_TAU_PROBE_S,
        variable_params,  # List of variable parameter names (in order)
        fixed_params,     # Dict of fixed parameter values
    )

    # Prepare arguments for each worker: (index, params_row, t_grid, return_totals, constants)
    # params_row: [substrate_mM, grams_urease_powder, temperature_C, k_d, initial_pH, powder_activity_frac, volume_L]
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
        - constants: (N0, C0)
    
    Returns:
    --------
    (index, pH_row): tuple with sample index and pH array
    """
    index, Ntot_row, Ctot_row, initial_pH_i, constants = args_tuple
    
    # Unpack constants
    N0, C0 = constants
    
    # Create simulator for speciation (only needs to compute pH, not kinetics)
    # E_loading_base_g_per_L doesn't affect speciation, so we use a dummy value
    sim = UreaseSimulator(
        S0=0.0,
        N0=N0,
        C0=C0,
        Pt_total_M=0.0,
        T_K=313.15,  # reference; speciation is not very T-sensitive over 20–40 °C
        initial_pH=initial_pH_i,
        E_loading_base_g_per_L=0.5,  # dummy value (not used for speciation)
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
    
    # Prepare constants (minimal set needed for speciation)
    constants = (
        N0,
        C0,
    )
    
    # If no per-sample initial pH is provided, use the default for all
    if initial_pH_array is None:
        initial_pH_array = np.full(n_samples, INITIAL_PH_DEFAULT, dtype=float)
    
    # Prepare arguments for each worker
    # Note: We use a reference T_K for speciation (not very T-sensitive)
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
    
    # Time grid generation
    time_grid_mode = CONFIG.get("time_grid_mode", "uniform")
    time_grid_alpha = CONFIG.get("time_grid_alpha", 0.5)
    time_grid_t_early = CONFIG.get("time_grid_t_early", 200.0)
    time_grid_n_early = CONFIG.get("time_grid_n_early", 400)
    time_grid_n_late = CONFIG.get("time_grid_n_late", 100)
    
    t_grid = generate_time_grid(
        t_max=t_max,
        n_times=n_times,
        mode=time_grid_mode,
        alpha=time_grid_alpha,
        t_early=time_grid_t_early,
        n_early=time_grid_n_early,
        n_late=time_grid_n_late
    )
    
    print("="*60)
    print("DEEPONET TRAINING DATA GENERATION")
    print("="*60)
    print(f"Configuration:")
    print(f"  • Number of samples: {n_samples:,}")
    print(f"  • Time grid: {t_max} s ({t_max/3600:.1f} hours), {n_times} points")
    print(f"  • Time grid mode: {time_grid_mode}")
    if time_grid_mode == "concentrated":
        print(f"    - Power-law exponent (alpha): {time_grid_alpha}")
        print(f"    - More points concentrated at beginning (smaller alpha = more concentration)")
    elif time_grid_mode == "piecewise":
        print(f"    - Early window (0 to {time_grid_t_early} s): {time_grid_n_early} points")
        print(f"    - Late window ({time_grid_t_early} to {t_max} s): {time_grid_n_late} points")
    print(f"  • Output mode: {'totals (S, Ntot, Ctot)' if use_totals else 'pH'}")
    print(f"  • Output directory: {output_dir}")
    
    # Print time grid statistics
    dt = np.diff(t_grid)
    print(f"\nTime grid statistics:")
    print(f"    - First 10% of time: {np.sum(t_grid <= 0.1*t_max)} points")
    print(f"    - First 25% of time: {np.sum(t_grid <= 0.25*t_max)} points")
    print(f"    - First 50% of time: {np.sum(t_grid <= 0.5*t_max)} points")
    print(f"    - Min time step: {dt.min():.4f} s")
    print(f"    - Max time step: {dt.max():.4f} s")
    print(f"    - Mean time step: {dt.mean():.4f} s")
    print(f"    - Time step ratio (max/min): {dt.max()/dt.min():.2f}x")
    
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
    
    # Validate configuration
    variable_params = CONFIG["variable_params"]
    fixed_params = CONFIG.get("fixed_params", {})
    
    # Check for conflicts (parameter in both variable and fixed)
    conflicts = set(variable_params) & set(fixed_params.keys())
    if conflicts:
        raise ValueError(f"Parameter(s) cannot be both variable and fixed: {conflicts}")
    
    # Check all variable params have ranges
    missing_ranges = [p for p in variable_params if p not in CONFIG["param_ranges"]]
    if missing_ranges:
        raise ValueError(f"Variable parameters missing ranges: {missing_ranges}")
    
    print(f"Variable parameters ({len(variable_params)}):")
    for name in variable_params:
        lo, hi = CONFIG["param_ranges"][name]
        print(f"    • {name}: [{lo}, {hi}]")
    
    if fixed_params:
        print(f"\nFixed parameters ({len(fixed_params)}):")
        for name, value in fixed_params.items():
            print(f"    • {name}: {value}")
    else:
        print(f"\nFixed parameters: None")
    
    print(f"\nGenerating random parameter vectors...")
    params_array = sample_parameters(n_samples, seed=seed)
    print(f"✓ Sampled {n_samples:,} parameter vectors")
    print(f"  Parameter array shape: {params_array.shape} (n_samples, n_variable_params)")
    
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
        
        # Extract per-sample initial pH from params_array
        # Find index of "initial_pH" in variable_params
        variable_params = CONFIG["variable_params"]
        fixed_params = CONFIG.get("fixed_params", {})
        
        if "initial_pH" in variable_params:
            initial_pH_idx = variable_params.index("initial_pH")
            initial_pH_array = params_array[:, initial_pH_idx]
        elif "initial_pH" in fixed_params:
            # Fixed value for all samples
            initial_pH_array = np.full(n_samples, fixed_params["initial_pH"], dtype=float)
        else:
            # Use default
            initial_pH_array = np.full(n_samples, INITIAL_PH_DEFAULT, dtype=float)
        
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
    # Compute time grid statistics for metadata
    dt = np.diff(t_grid)
    time_grid_stats = {
        "min_dt": float(dt.min()),
        "max_dt": float(dt.max()),
        "mean_dt": float(dt.mean()),
        "dt_ratio": float(dt.max() / dt.min()),
        "points_in_first_10pct": int(np.sum(t_grid <= 0.1*t_max)),
        "points_in_first_25pct": int(np.sum(t_grid <= 0.25*t_max)),
        "points_in_first_50pct": int(np.sum(t_grid <= 0.5*t_max)),
    }
    
    metadata = {
        "n_samples": n_samples,
        "n_times": n_times,
        "t_max": t_max,
        "t_grid": t_grid.tolist(),
        "time_grid_config": {
            "mode": time_grid_mode,
            "alpha": time_grid_alpha if time_grid_mode == "concentrated" else None,
            "t_early": time_grid_t_early if time_grid_mode == "piecewise" else None,
            "n_early": time_grid_n_early if time_grid_mode == "piecewise" else None,
            "n_late": time_grid_n_late if time_grid_mode == "piecewise" else None,
        },
        "time_grid_stats": time_grid_stats,
        "N0": N0,
        "C0": C0,
        "initial_pH_default": INITIAL_PH_DEFAULT,
        "volume_L_default": VOLUME_L_DEFAULT,
        "powder_activity_frac_default": POWDER_ACTIVITY_FRAC_DEFAULT,
        "use_totals": use_totals,
        "variable_params": CONFIG["variable_params"],  # List of variable parameter names (in order)
        "fixed_params": CONFIG.get("fixed_params", {}),  # Dict of fixed parameter values
        "param_ranges": CONFIG["param_ranges"],  # Ranges for variable parameters
        "n_variable_params": len(CONFIG["variable_params"]),  # Dimensionality of saved params array
        "all_param_names": [
            "substrate_mM",
            "grams_urease_powder",
            "temperature_C",
            "k_d",
            "initial_pH",
            "powder_activity_frac",
            "volume_L",
        ],  # All possible parameter names (for reference)
        "fixed_simulation_params": {
            "t_shift_s": FIXED_T_SHIFT_S,
            "tau_probe_s": FIXED_TAU_PROBE_S,
        },
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("  ✓ metadata.json saved")
    
    # Save a sample as CSV for inspection
    print("  → Saving sample_params.csv (first 100 samples)...")
    variable_params = CONFIG["variable_params"]
    sample_dict = {}
    for i, param_name in enumerate(variable_params):
        sample_dict[param_name] = params_array[:100, i]
    sample_df = pd.DataFrame(sample_dict)
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
