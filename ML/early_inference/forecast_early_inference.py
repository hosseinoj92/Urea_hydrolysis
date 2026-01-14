"""
Forecast function for early inference: uses ML estimator + mechanistic simulator.
"""

import torch
import numpy as np
from pathlib import Path
import json
from typing import Dict, Tuple, Optional

from early_inference_model import create_early_inference_model
from mechanistic_simulator import UreaseSimulator


def load_early_inference_model(model_path: Path, device: torch.device):
    """Load trained early inference model."""
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    metadata = checkpoint.get('metadata', {})
    normalization_stats = checkpoint.get('normalization_stats', {})
    prefix_length = checkpoint.get('prefix_length', 30.0)
    
    # Reconstruct model (unified: E0_g_per_L and k_d only)
    infer_params = metadata.get('infer_params', ['E0_g_per_L', 'k_d'])
    n_output_params = len(infer_params)
    n_known_inputs = len(metadata.get('known_input_names', []))
    
    # Get sequence length from data or config
    seq_length = config.get('prefix_n_points', 50)
    
    model = create_early_inference_model(
        seq_length=seq_length,
        n_known_inputs=n_known_inputs,
        n_output_params=n_output_params,
        tcn_channels=config.get('tcn_channels', [64, 128, 256]),
        tcn_kernel_size=config.get('tcn_kernel_size', 3),
        tcn_dropout=config.get('tcn_dropout', 0.2),
        mlp_hidden_dims=config.get('mlp_hidden_dims', [128, 64]),
        output_dropout=config.get('output_dropout', 0.1),
        use_uncertainty=config.get('use_uncertainty', True),
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, metadata, normalization_stats, prefix_length


def normalize_inputs(pH_seq: np.ndarray, t_seq: np.ndarray, known_inputs: np.ndarray, 
                    normalization_stats: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Normalize inputs using saved statistics.
    
    Parameters
    ----------
    pH_seq: (seq_len,) array
    t_seq: (seq_len,) array of time values
    known_inputs: (n_known,) array
    normalization_stats: dict with input normalization stats
    
    Returns
    -------
    pH_seq_norm: (1, seq_len) tensor
    t_seq_norm: (1, seq_len) tensor
    known_inputs_norm: (1, n_known) tensor
    """
    input_stats = normalization_stats.get('input', {})
    
    # Normalize pH
    pH_mean = input_stats.get('pH_mean', 0.0)
    pH_std = input_stats.get('pH_std', 1.0)
    pH_seq_norm = (pH_seq - pH_mean) / pH_std
    
    # Normalize time grid per-sequence (same as training - preserves dt relationships)
    # Note: Saved t_mean and t_std are dummy values (0.0, 1.0) since we use per-sequence norm
    if len(t_seq) > 1:
        t_mean_seq = np.mean(t_seq)
        t_std_seq = np.std(t_seq) + 1e-8
        t_seq_norm = (t_seq - t_mean_seq) / t_std_seq
    else:
        t_seq_norm = t_seq
    
    # Normalize known inputs
    known_mean = np.array(input_stats.get('known_mean', [0.0] * len(known_inputs)))
    known_std = np.array(input_stats.get('known_std', [1.0] * len(known_inputs)))
    known_inputs_norm = (known_inputs - known_mean) / (known_std + 1e-8)
    
    # Convert to tensors
    pH_seq_tensor = torch.FloatTensor(pH_seq_norm).unsqueeze(0)  # (1, seq_len)
    t_seq_tensor = torch.FloatTensor(t_seq_norm).unsqueeze(0)  # (1, seq_len)
    known_inputs_tensor = torch.FloatTensor(known_inputs_norm).unsqueeze(0)  # (1, n_known)
    
    return pH_seq_tensor, t_seq_tensor, known_inputs_tensor


def denormalize_outputs(params_norm: np.ndarray, normalization_stats: dict) -> np.ndarray:
    """
    Denormalize predicted parameters.
    
    Parameters
    ----------
    params_norm: (n_params,) array of normalized parameters
    normalization_stats: dict with output normalization stats
    
    Returns
    -------
    params: (n_params,) array of denormalized parameters
    """
    output_stats = normalization_stats.get('output', {})
    param_mean = np.array(output_stats.get('param_mean', [0.0] * len(params_norm)))
    param_std = np.array(output_stats.get('param_std', [1.0] * len(params_norm)))
    
    params = params_norm * param_std + param_mean
    return params


def forecast_ph(
    pH_prefix: np.ndarray,
    t_prefix: np.ndarray,
    known_inputs: Dict[str, float],
    model_path: Path,
    t_forecast: np.ndarray,
    device: torch.device = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Forecast pH trajectory using early inference model + mechanistic simulator.
    
    Parameters
    ----------
    pH_prefix: (n_points,) array of measured pH values
    t_prefix: (n_points,) array of time points for prefix
    known_inputs: dict with known inputs:
        - substrate_mM: float
        - grams_urease_powder: float
        - temperature_C: float
        - initial_pH: float
        - powder_activity_frac: float
        - volume_L: float
    model_path: path to trained early inference model
    t_forecast: (n_forecast,) array of time points to forecast
    device: torch device (if None, uses auto)
    
    Returns
    -------
    pH_forecast: (n_forecast,) array of forecasted pH values
    estimated_params: dict of estimated parameters
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model, metadata, normalization_stats, prefix_length = load_early_inference_model(model_path, device)
    infer_params = metadata.get('infer_params', ['E0_g_per_L', 'k_d'])
    # Unified: exactly 5 known inputs (no powder_activity_frac)
    known_input_names = metadata.get('known_input_names', [
        'substrate_mM', 'grams_urease_powder', 'temperature_C',
        'initial_pH', 'volume_L'
    ])
    
    # Prepare inputs
    # Extract known inputs in correct order
    known_inputs_array = np.array([
        known_inputs[name] for name in known_input_names
    ])
    
    # Normalize inputs (now includes time!)
    pH_seq_tensor, t_seq_tensor, known_inputs_tensor = normalize_inputs(
        pH_prefix, t_prefix, known_inputs_array, normalization_stats
    )
    
    # Predict parameters
    with torch.no_grad():
        pH_seq_tensor = pH_seq_tensor.to(device)
        t_seq_tensor = t_seq_tensor.to(device)  # Pass time to device!
        known_inputs_tensor = known_inputs_tensor.to(device)
        mean, logvar = model(pH_seq_tensor, t_seq_tensor, known_inputs_tensor)  # Include time!
        params_norm = mean.cpu().numpy().squeeze()
    
    # Denormalize parameters
    params = denormalize_outputs(params_norm, normalization_stats)
    
    # Create parameter dict (unified: E0_g_per_L and k_d only)
    estimated_params = {name: float(val) for name, val in zip(infer_params, params)}
    
    # Build simulator (dummy base loading, will be overridden by E_eff0)
    S0 = known_inputs['substrate_mM'] / 1000.0  # mM → M
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
    
    # Parameters for ODE solver (unified: use E_eff0 directly)
    sim_params = {
        'E_eff0': estimated_params.get('E0_g_per_L', 0.5),  # Direct enzyme loading [g/L]
        'k_d': estimated_params.get('k_d', 0.0),
        't_shift': 0.0,
        'tau_probe': 0.0,  # Not used (true pH space)
    }
    
    # Simulate forward (true pH space)
    pH_forecast = sim.simulate_forward(sim_params, t_forecast, return_totals=False, apply_probe_lag=False)
    
    return pH_forecast, estimated_params
