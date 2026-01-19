"""
Train traditional ML models for early inference parameter estimation.

Supports multiple algorithms:
- Gaussian Process Regression (GPR)
- XGBoost
- LightGBM
- CatBoost
- Random Forest
- Extra Trees
- Multi-layer Perceptron (MLP)

Models can be enabled/disabled via boolean flags in CONFIG.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import pickle
from tqdm import tqdm
import time
from datetime import datetime

# Traditional ML imports
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb
import lightgbm as lgb
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("Warning: CatBoost not available. Install with: pip install catboost")

# ╔══════════════════════════════════════════════════════════════╗
# ║                       USER CONFIG                             ║
# ╚══════════════════════════════════════════════════════════════╝
CONFIG = {
    # Data paths
    "data_dir": r"C:\Users\vt4ho\Simulations\simulation_data\generated_data\imperfect\version_experiment\Generated_Data_EarlyInference_50000",
    "output_dir": r"C:\Users\vt4ho\Simulations\simulation_data\models\imperfect\version_experiment\traditional_ml_models",
    "prefix_length": 30.0,  # Which prefix length to train on
    
    # Model selection (set to True to train, False to skip)
    "train_gpr": False,
    "train_xgboost": False,
    "train_lightgbm": False,
    "train_catboost": False,
    "train_random_forest": False,
    "train_extra_trees": False,
    "train_mlp": True,
    
    # Training parameters
    "val_split": 0.2,
    "test_split": 0.1,  # Additional test set for final evaluation
    "seed": 42,
    "n_jobs": -1,  # Number of parallel jobs (-1 = all cores)
    
    # Model-specific hyperparameters
    "gpr": {
        "kernel": "RBF",  # "RBF", "Matern", or "RBF+White"
        "alpha": 1e-6,  # Noise level
        "n_restarts_optimizer": 2,  # Reduced for faster training (GPR is very slow, O(n³) complexity)
    },
    "xgboost": {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    },
    "lightgbm": {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
    },
    "catboost": {
        "iterations": 500,
        "depth": 6,
        "learning_rate": 0.05,
        "random_seed": 42,
        "verbose": False,
    },
    "random_forest": {
        "n_estimators": 500,
        "max_depth": 20,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
    },
    "extra_trees": {
        "n_estimators": 500,
        "max_depth": 20,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "random_state": 42,
    },
    "mlp": {
        "hidden_layer_sizes": (256, 128, 64),
        "activation": "relu",
        "solver": "adam",
        "alpha": 1e-4,  # L2 regularization
        "learning_rate": "adaptive",
        "max_iter": 2000,
        "early_stopping": True,
        "validation_fraction": 0.1,
        "random_state": 42,
    },
}


def load_training_data(data_dir: Path, prefix_length: float):
    """Load training data from NPZ file."""
    data_file = data_dir / "training_data.npz"
    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")
    
    print(f"Loading data from {data_file}...")
    data = np.load(data_file, allow_pickle=True)
    
    # Load data for specified prefix length
    prefix_key = f"prefix_{int(prefix_length)}s"
    if prefix_key not in data:
        available = [k for k in data.keys() if k.startswith('prefix_')]
        raise ValueError(f"Prefix length {prefix_length}s not found in data. Available: {available}")
    
    prefix_data = data[prefix_key].item()
    pH_prefix = prefix_data["pH_prefix"]  # (n_samples, seq_len)
    t_prefix = prefix_data["t_prefix"]     # (n_samples, seq_len)
    known_inputs = prefix_data["known_inputs"]  # (n_samples, 5)
    target_params = prefix_data["target_params"]  # (n_samples, 2)
    
    # Load metadata
    with open(data_dir / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    print(f"Data shape: pH_prefix={pH_prefix.shape}, known_inputs={known_inputs.shape}, target_params={target_params.shape}")
    print(f"Infer params: {metadata['infer_params']}")
    
    return pH_prefix, t_prefix, known_inputs, target_params, metadata


def prepare_features(pH_prefix: np.ndarray, t_prefix: np.ndarray, known_inputs: np.ndarray) -> np.ndarray:
    """
    Prepare feature vector from pH sequence, time sequence, and known inputs.
    
    Features include:
    - Statistical features from pH sequence (mean, std, min, max, initial, final, etc.)
    - Statistical features from time sequence
    - Known inputs (substrate_mM, grams_urease_powder, temperature_C, initial_pH, volume_L)
    """
    n_samples = pH_prefix.shape[0]
    features = []
    
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
    
    # pH rate of change (approximate derivative)
    if pH_prefix.shape[1] > 1:
        pH_diff = np.diff(pH_prefix, axis=1)
        t_diff = np.diff(t_prefix, axis=1)
        # Avoid division by zero
        t_diff_safe = np.where(t_diff > 1e-8, t_diff, 1e-8)
        pH_rate = pH_diff / t_diff_safe
        pH_rate_mean = np.mean(pH_rate, axis=1)
        pH_rate_max = np.max(pH_rate, axis=1)
    else:
        pH_rate_mean = np.zeros(n_samples)
        pH_rate_max = np.zeros(n_samples)
    
    # Early vs late pH (first 30% vs last 30%)
    n_early = max(1, int(0.3 * pH_prefix.shape[1]))
    n_late = max(1, int(0.3 * pH_prefix.shape[1]))
    pH_early_mean = np.mean(pH_prefix[:, :n_early], axis=1)
    pH_late_mean = np.mean(pH_prefix[:, -n_late:], axis=1)
    pH_early_late_diff = pH_late_mean - pH_early_mean
    
    # Combine all features
    features = np.column_stack([
        # pH statistics
        pH_mean, pH_std, pH_min, pH_max, pH_initial, pH_final, pH_range, pH_change,
        # Time statistics
        t_mean, t_std, t_max, t_final,
        # Rate features
        pH_rate_mean, pH_rate_max,
        # Early/late comparison
        pH_early_mean, pH_late_mean, pH_early_late_diff,
        # Known inputs
        known_inputs,  # (n_samples, 5)
    ])
    
    return features


def train_gpr(X_train, y_train, config: dict, pbar=None):
    """Train Gaussian Process Regressor."""
    import threading
    import time as time_module
    
    if pbar:
        pbar.set_description("Training GPR")
    
    # Select kernel
    kernel_type = config.get("kernel", "RBF")
    if kernel_type == "RBF":
        kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    elif kernel_type == "Matern":
        kernel = Matern(length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5)
    elif kernel_type == "RBF+White":
        kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)) + WhiteKernel(noise_level=1e-3)
    else:
        kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    
    # Reduce restarts for faster training (GPR is very slow)
    n_restarts = min(config.get("n_restarts_optimizer", 3), 2)  # Cap at 2 for speed
    
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=config.get("alpha", 1e-6),
        n_restarts_optimizer=n_restarts,
        random_state=config.get("seed", 42),
    )
    
    # GPR can be slow for large datasets, so use smaller subset
    max_samples = 2000  # Reduced for faster training (GPR is O(n³))
    if len(X_train) > max_samples:
        if pbar:
            pbar.write(f"  Using subset of {max_samples} samples for GPR (full dataset: {len(X_train)})")
        indices = np.random.RandomState(config.get("seed", 42)).choice(len(X_train), max_samples, replace=False)
        X_train_subset = X_train[indices]
        y_train_subset = y_train[indices]
    else:
        X_train_subset = X_train
        y_train_subset = y_train
    
    # Show progress with elapsed time updates
    if pbar:
        pbar.write("  Fitting GPR (this may take 5-15 minutes)...")
        start_time = time_module.time()
        elapsed_done = threading.Event()
        
        # Update elapsed time in a separate thread
        def update_elapsed():
            while not elapsed_done.is_set():
                elapsed = time_module.time() - start_time
                mins = int(elapsed // 60)
                secs = int(elapsed % 60)
                if pbar:
                    pbar.set_postfix({"elapsed": f"{mins}m{secs}s", "status": "fitting..."})
                if elapsed_done.wait(5):  # Wait 5 seconds or until event is set
                    break
        
        elapsed_thread = threading.Thread(target=update_elapsed, daemon=True)
        elapsed_thread.start()
    
    try:
        model.fit(X_train_subset, y_train_subset)
    finally:
        if pbar:
            elapsed_done.set()
            elapsed = time_module.time() - start_time
            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            pbar.set_postfix({"elapsed": f"{mins}m{secs}s", "status": "complete"})
            pbar.update(1)
    
    return model


def train_xgboost(X_train, y_train, config: dict, pbar=None):
    """Train XGBoost model with multi-output support."""
    import threading
    import time as time_module
    
    n_estimators = config.get("n_estimators", 500)
    
    if pbar:
        pbar.set_description("Training XGBoost")
        pbar.total = n_estimators
        pbar.unit = "trees"
        pbar.n = 0
        pbar.refresh()
    
    # XGBoost base model (single output)
    base_model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=config.get("max_depth", 6),
        learning_rate=config.get("learning_rate", 0.05),
        subsample=config.get("subsample", 0.8),
        colsample_bytree=config.get("colsample_bytree", 0.8),
        random_state=config.get("random_state", 42),
        n_jobs=config.get("n_jobs", -1),
        verbosity=0,
    )
    
    # Wrap in MultiOutputRegressor for multi-output support
    model = MultiOutputRegressor(base_model, n_jobs=config.get("n_jobs", -1))
    
    # Estimate progress based on time (since MultiOutputRegressor doesn't expose callbacks)
    if pbar:
        start_time = time_module.time()
        progress_done = threading.Event()
        
        def update_progress():
            while not progress_done.is_set():
                elapsed = time_module.time() - start_time
                # Estimate: assume linear progress (rough approximation)
                # For 2 outputs, training takes roughly 2x time
                estimated_time_per_tree = 0.1  # Rough estimate, will be refined
                if elapsed > 1.0:  # After 1 second, refine estimate
                    estimated_time_per_tree = elapsed / (n_estimators * 2)  # 2 outputs
                
                estimated_trees = min(int(elapsed / max(estimated_time_per_tree, 1e-6)), n_estimators)
                if pbar:
                    pbar.n = estimated_trees
                    pbar.set_postfix({"trees": f"{estimated_trees}/{n_estimators}"})
                    pbar.refresh()
                progress_done.wait(0.5)  # Update every 0.5 seconds
        
        progress_thread = threading.Thread(target=update_progress, daemon=True)
        progress_thread.start()
    
    model.fit(X_train, y_train)
    
    if pbar:
        progress_done.set()
        pbar.n = n_estimators
        pbar.set_postfix({"trees": f"{n_estimators}/{n_estimators}"})
        pbar.update(0)  # Refresh display
    
    return model


def train_lightgbm(X_train, y_train, config: dict, pbar=None):
    """Train LightGBM model with multi-output support."""
    import threading
    import time as time_module
    
    n_estimators = config.get("n_estimators", 500)
    
    if pbar:
        pbar.set_description("Training LightGBM")
        pbar.total = n_estimators
        pbar.unit = "trees"
        pbar.n = 0
        pbar.refresh()
    
    # LightGBM base model (single output)
    base_model = lgb.LGBMRegressor(
        n_estimators=n_estimators,
        max_depth=config.get("max_depth", 6),
        learning_rate=config.get("learning_rate", 0.05),
        subsample=config.get("subsample", 0.8),
        colsample_bytree=config.get("colsample_bytree", 0.8),
        random_state=config.get("random_state", 42),
        n_jobs=1,  # Set to 1 for MultiOutputRegressor (it handles parallelism)
        verbose=-1,  # Disable built-in verbose
    )
    
    # Wrap in MultiOutputRegressor for multi-output support
    model = MultiOutputRegressor(base_model, n_jobs=config.get("n_jobs", -1))
    
    # Estimate progress based on time
    if pbar:
        start_time = time_module.time()
        progress_done = threading.Event()
        
        def update_progress():
            while not progress_done.is_set():
                elapsed = time_module.time() - start_time
                estimated_time_per_tree = 0.05  # Rough estimate
                if elapsed > 1.0:
                    estimated_time_per_tree = elapsed / (n_estimators * 2)
                
                estimated_trees = min(int(elapsed / max(estimated_time_per_tree, 1e-6)), n_estimators)
                if pbar:
                    pbar.n = estimated_trees
                    pbar.set_postfix({"trees": f"{estimated_trees}/{n_estimators}"})
                    pbar.refresh()
                progress_done.wait(0.5)
        
        progress_thread = threading.Thread(target=update_progress, daemon=True)
        progress_thread.start()
    
    model.fit(X_train, y_train)
    
    if pbar:
        progress_done.set()
        pbar.n = n_estimators
        pbar.set_postfix({"trees": f"{n_estimators}/{n_estimators}"})
        pbar.update(0)
    
    return model


def train_catboost(X_train, y_train, config: dict, pbar=None):
    """Train CatBoost model with multi-output support."""
    import threading
    import time as time_module
    
    if not CATBOOST_AVAILABLE:
        raise ImportError("CatBoost not available. Install with: pip install catboost")
    
    iterations = config.get("iterations", 500)
    
    if pbar:
        pbar.set_description("Training CatBoost")
        pbar.total = iterations
        pbar.unit = "iter"
        pbar.n = 0
        pbar.refresh()
    
    # CatBoost base model (single output)
    base_model = cb.CatBoostRegressor(
        iterations=iterations,
        depth=config.get("depth", 6),
        learning_rate=config.get("learning_rate", 0.05),
        random_seed=config.get("random_seed", 42),
        verbose=False,  # Suppress CatBoost's own progress
        thread_count=1,  # Set to 1 for MultiOutputRegressor (it handles parallelism)
    )
    
    # Wrap in MultiOutputRegressor for multi-output support
    model = MultiOutputRegressor(base_model, n_jobs=config.get("n_jobs", -1))
    
    # Estimate progress based on time
    if pbar:
        start_time = time_module.time()
        progress_done = threading.Event()
        
        def update_progress():
            while not progress_done.is_set():
                elapsed = time_module.time() - start_time
                estimated_time_per_iter = 0.05
                if elapsed > 1.0:
                    estimated_time_per_iter = elapsed / (iterations * 2)
                
                estimated_iters = min(int(elapsed / max(estimated_time_per_iter, 1e-6)), iterations)
                if pbar:
                    pbar.n = estimated_iters
                    pbar.set_postfix({"iter": f"{estimated_iters}/{iterations}"})
                    pbar.refresh()
                progress_done.wait(0.5)
        
        progress_thread = threading.Thread(target=update_progress, daemon=True)
        progress_thread.start()
    
    model.fit(X_train, y_train)
    
    if pbar:
        progress_done.set()
        pbar.n = iterations
        pbar.set_postfix({"iter": f"{iterations}/{iterations}"})
        pbar.update(0)
    
    return model


def train_random_forest(X_train, y_train, config: dict, pbar=None):
    """Train Random Forest model."""
    import threading
    import time as time_module
    
    n_estimators = config.get("n_estimators", 500)
    
    if pbar:
        pbar.set_description("Training Random Forest")
        pbar.total = n_estimators
        pbar.unit = "trees"
        pbar.n = 0
        pbar.refresh()
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=config.get("max_depth", 20),
        min_samples_split=config.get("min_samples_split", 5),
        min_samples_leaf=config.get("min_samples_leaf", 2),
        random_state=config.get("random_state", 42),
        n_jobs=config.get("n_jobs", -1),
        verbose=0,
    )
    
    # Estimate progress based on time
    if pbar:
        start_time = time_module.time()
        progress_done = threading.Event()
        
        def update_progress():
            while not progress_done.is_set():
                elapsed = time_module.time() - start_time
                estimated_time_per_tree = 0.1
                if elapsed > 1.0:
                    estimated_time_per_tree = elapsed / n_estimators
                
                estimated_trees = min(int(elapsed / max(estimated_time_per_tree, 1e-6)), n_estimators)
                if pbar:
                    pbar.n = estimated_trees
                    pbar.set_postfix({"trees": f"{estimated_trees}/{n_estimators}"})
                    pbar.refresh()
                progress_done.wait(0.5)
        
        progress_thread = threading.Thread(target=update_progress, daemon=True)
        progress_thread.start()
    
    model.fit(X_train, y_train)
    
    if pbar:
        progress_done.set()
        pbar.n = n_estimators
        pbar.set_postfix({"trees": f"{n_estimators}/{n_estimators}"})
        pbar.update(0)
    
    return model


def train_extra_trees(X_train, y_train, config: dict, pbar=None):
    """Train Extra Trees model."""
    import threading
    import time as time_module
    
    n_estimators = config.get("n_estimators", 500)
    
    if pbar:
        pbar.set_description("Training Extra Trees")
        pbar.total = n_estimators
        pbar.unit = "trees"
        pbar.n = 0
        pbar.refresh()
    
    model = ExtraTreesRegressor(
        n_estimators=n_estimators,
        max_depth=config.get("max_depth", 20),
        min_samples_split=config.get("min_samples_split", 5),
        min_samples_leaf=config.get("min_samples_leaf", 2),
        random_state=config.get("random_state", 42),
        n_jobs=config.get("n_jobs", -1),
        verbose=0,
    )
    
    # Estimate progress based on time
    if pbar:
        start_time = time_module.time()
        progress_done = threading.Event()
        
        def update_progress():
            while not progress_done.is_set():
                elapsed = time_module.time() - start_time
                estimated_time_per_tree = 0.1
                if elapsed > 1.0:
                    estimated_time_per_tree = elapsed / n_estimators
                
                estimated_trees = min(int(elapsed / max(estimated_time_per_tree, 1e-6)), n_estimators)
                if pbar:
                    pbar.n = estimated_trees
                    pbar.set_postfix({"trees": f"{estimated_trees}/{n_estimators}"})
                    pbar.refresh()
                progress_done.wait(0.5)
        
        progress_thread = threading.Thread(target=update_progress, daemon=True)
        progress_thread.start()
    
    model.fit(X_train, y_train)
    
    if pbar:
        progress_done.set()
        pbar.n = n_estimators
        pbar.set_postfix({"trees": f"{n_estimators}/{n_estimators}"})
        pbar.update(0)
    
    return model


def train_mlp(X_train, y_train, config: dict, pbar=None):
    """
    Train Multi-layer Perceptron with iteration tracking.
    
    Uses warm_start to train in chunks and show real progress.
    """
    max_iter = config.get("max_iter", 500)
    chunk_size = 50  # Train in chunks of 50 iterations (larger to avoid convergence warnings)
    
    if pbar:
        pbar.set_description("Training MLP")
        pbar.total = max_iter
        pbar.unit = "iter"
        pbar.n = 0
        pbar.refresh()
    
    # Use warm_start to train in chunks and track progress
    # Start with a reasonable max_iter to avoid convergence warnings
    model = MLPRegressor(
        hidden_layer_sizes=config.get("hidden_layer_sizes", (256, 128, 64)),
        activation=config.get("activation", "relu"),
        solver=config.get("solver", "adam"),
        alpha=config.get("alpha", 1e-4),
        learning_rate=config.get("learning_rate", "adaptive"),  # See explanation below
        max_iter=chunk_size,  # Will be updated in chunks
        early_stopping=config.get("early_stopping", True),
        validation_fraction=config.get("validation_fraction", 0.1),
        random_state=config.get("random_state", 42),
        warm_start=True,  # Enable warm start for chunk training
        verbose=False,
    )
    
    # Train in chunks to show progress
    total_trained = 0
    n_chunks = (max_iter + chunk_size - 1) // chunk_size
    
    for chunk in range(n_chunks):
        # Determine how many iterations for this chunk
        remaining = max_iter - total_trained
        if remaining <= 0:
            break
        
        iterations_this_chunk = min(chunk_size, remaining)
        model.max_iter = iterations_this_chunk
        
        # Train this chunk
        model.fit(X_train, y_train)
        
        # Get actual iterations trained (may be less if converged early)
        actual_iters_this_chunk = getattr(model, 'n_iter_', iterations_this_chunk)
        total_trained += actual_iters_this_chunk
        
        # Update progress
        if pbar:
            pbar.n = total_trained
            pbar.set_postfix({"iter": f"{total_trained}/{max_iter}"})
            pbar.refresh()
        
        # Check if training stopped early (converged or early stopping)
        if actual_iters_this_chunk < iterations_this_chunk:
            # Training converged or stopped early
            if pbar:
                pbar.set_postfix({"iter": f"{total_trained}/{max_iter} (converged)"})
                pbar.refresh()
            break
        
        if total_trained >= max_iter:
            break
    
    # Final update
    if pbar:
        actual_iters = getattr(model, 'n_iter_', total_trained)
        pbar.n = actual_iters
        pbar.set_postfix({"iter": f"{actual_iters}/{max_iter}"})
        pbar.update(0)
    
    return model


def main():
    """Main training function."""
    data_dir = Path(CONFIG["data_dir"])
    output_dir = Path(CONFIG["output_dir"])
    prefix_length = CONFIG["prefix_length"]
    seed = CONFIG["seed"]
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("TRADITIONAL ML MODELS TRAINING")
    print("="*80)
    print(f"Data directory: {data_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Prefix length: {prefix_length}s")
    print(f"Seed: {seed}")
    
    # Load data
    pH_prefix, t_prefix, known_inputs, target_params, metadata = load_training_data(data_dir, prefix_length)
    
    # Prepare features
    print("\nPreparing features...")
    X = prepare_features(pH_prefix, t_prefix, known_inputs)
    y = target_params  # (n_samples, 2) - powder_activity_frac, k_d
    
    print(f"Feature shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Train/val/test split
    print("\nSplitting data...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=CONFIG["test_split"], random_state=seed, shuffle=True
    )
    val_size = CONFIG["val_split"] / (1 - CONFIG["test_split"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_size, random_state=seed, shuffle=True
    )
    
    print(f"Train: {len(X_train)} samples")
    print(f"Val: {len(X_val)} samples")
    print(f"Test: {len(X_test)} samples")
    
    # Feature scaling (for models that need it)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    with open(output_dir / f"scaler_prefix_{int(prefix_length)}s.pkl", "wb") as f:
        pickle.dump(scaler, f)
    
    # Train models
    models = {}
    model_configs = {
        "gpr": (train_gpr, CONFIG["gpr"], True, X_train_scaled),  # GPR needs scaling
        "xgboost": (train_xgboost, CONFIG["xgboost"], False, X_train),
        "lightgbm": (train_lightgbm, CONFIG["lightgbm"], False, X_train),
        "catboost": (train_catboost, CONFIG["catboost"], False, X_train),
        "random_forest": (train_random_forest, CONFIG["random_forest"], False, X_train),
        "extra_trees": (train_extra_trees, CONFIG["extra_trees"], False, X_train),
        "mlp": (train_mlp, CONFIG["mlp"], True, X_train_scaled),  # MLP needs scaling
    }
    
    train_flags = {
        "gpr": CONFIG["train_gpr"],
        "xgboost": CONFIG["train_xgboost"],
        "lightgbm": CONFIG["train_lightgbm"],
        "catboost": CONFIG["train_catboost"],
        "random_forest": CONFIG["train_random_forest"],
        "extra_trees": CONFIG["train_extra_trees"],
        "mlp": CONFIG["train_mlp"],
    }
    
    print("\n" + "="*80)
    print("TRAINING MODELS")
    print("="*80)
    
    # Count how many models will be trained
    models_to_train = [name for name in model_configs.keys() if train_flags[name]]
    n_models = len(models_to_train)
    
    if n_models == 0:
        print("No models enabled for training!")
        return
    
    # Create overall progress bar
    overall_pbar = tqdm(total=n_models, desc="Overall Progress", unit="model", 
                       position=0, leave=True, ncols=120, 
                       bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} models [{elapsed}<{remaining}, {rate_fmt}]')
    
    for model_idx, (model_name, (train_func, model_config, use_scaled, X_train_data)) in enumerate(model_configs.items()):
        if not train_flags[model_name]:
            continue
        
        try:
            # Update overall progress bar description
            overall_pbar.set_description(f"Training {model_name:15s}")
            
            # Create individual model progress bar (nested)
            # Will be updated by training function with actual total
            model_pbar = tqdm(total=100, desc=f"  {model_name:15s}", unit="", 
                            position=1, leave=False, ncols=120,
                            bar_format='{desc}: {bar}| {n_fmt}/{total_fmt} {unit} [{elapsed}<{remaining}]')
            
            start_time = time.time()
            model = train_func(X_train_data, y_train, model_config, pbar=model_pbar)
            train_time = time.time() - start_time
            
            model_pbar.close()
            
            # Evaluate on validation set
            X_eval = X_val_scaled if use_scaled else X_val
            y_pred = model.predict(X_eval)
            
            # Compute metrics
            mae = np.mean(np.abs(y_pred - y_val), axis=0)
            rmse = np.sqrt(np.mean((y_pred - y_val) ** 2, axis=0))
            
            # Write results (using write to avoid interfering with progress bars)
            overall_pbar.write(f"  ✓ {model_name:15s} - Val MAE: {mae}, RMSE: {rmse}, Time: {train_time:.2f}s")
            
            # Save model
            model_file = output_dir / f"{model_name}_prefix_{int(prefix_length)}s.pkl"
            with open(model_file, "wb") as f:
                pickle.dump(model, f)
            overall_pbar.write(f"    Saved: {model_file.name}")
            
            models[model_name] = {
                "model": model,
                "use_scaled": use_scaled,
                "train_time": train_time,
                "val_mae": mae.tolist(),
                "val_rmse": rmse.tolist(),
            }
            
            # Update overall progress
            overall_pbar.update(1)
            
        except Exception as e:
            overall_pbar.write(f"  ✗ ERROR training {model_name}: {e}")
            import traceback
            overall_pbar.write(traceback.format_exc())
            overall_pbar.update(1)  # Still count it as processed
    
    overall_pbar.close()
    
    # Save metadata
    training_metadata = {
        "prefix_length": prefix_length,
        "n_train": len(X_train),
        "n_val": len(X_val),
        "n_test": len(X_test),
        "n_features": X.shape[1],
        "n_targets": y.shape[1],
        "infer_params": metadata["infer_params"],
        "known_input_names": metadata.get("known_input_names", []),
        "trained_models": list(models.keys()),
        "model_metrics": {
            name: {
                "train_time": info["train_time"],
                "val_mae": info["val_mae"],
                "val_rmse": info["val_rmse"],
            }
            for name, info in models.items()
        },
        "config": CONFIG,
        "timestamp": datetime.now().isoformat(),
    }
    
    with open(output_dir / f"training_metadata_prefix_{int(prefix_length)}s.json", "w") as f:
        json.dump(training_metadata, f, indent=2)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"Trained {len(models)} models")
    print(f"All files saved to: {output_dir}")


if __name__ == "__main__":
    main()
