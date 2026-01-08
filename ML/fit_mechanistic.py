"""
Mechanistic parameter fitting using least squares optimization.
"""

import numpy as np
from scipy.optimize import least_squares
from typing import Dict, Tuple, Optional
from mechanistic_simulator import UreaseSimulator


def fit_mechanistic_parameters(
    pH_measured: np.ndarray,
    t_measured: np.ndarray,
    known_inputs: Dict[str, float],
    param_bounds: Dict[str, Tuple[float, float]] = None,
    initial_guess: Dict[str, float] = None,
) -> Dict[str, float]:
    """
    Fit mechanistic parameters using least squares optimization.
    
    Parameters
    ----------
    pH_measured: (n_points,) array of measured pH values
    t_measured: (n_points,) array of time points
    known_inputs: dict with known inputs (substrate_mM, grams_urease_powder, etc.)
    param_bounds: dict with parameter bounds, e.g., {'activity_scale': (0.1, 2.0), 'k_d': (0.0, 5e-3)}
    initial_guess: dict with initial parameter guesses
    
    Returns
    -------
    fitted_params: dict of fitted parameters
    """
    # Default bounds
    if param_bounds is None:
        param_bounds = {
            'activity_scale': (0.1, 2.0),
            'k_d': (0.0, 5e-3),
            'tau_probe': (0.0, 30.0),
            'pH_offset': (-0.1, 0.1),
        }
    
    # Default initial guess
    if initial_guess is None:
        initial_guess = {
            'activity_scale': 1.0,
            'k_d': 0.0,
            'tau_probe': 0.0,
            'pH_offset': 0.0,
        }
    
    # Parameters to fit
    param_names = list(param_bounds.keys())
    n_params = len(param_names)
    
    # Build simulator
    S0 = known_inputs['substrate_mM'] / 1000.0
    T_K = known_inputs['temperature_C'] + 273.15
    E_loading_base_g_per_L = (known_inputs['grams_urease_powder'] * 
                               known_inputs['powder_activity_frac'] / 
                               known_inputs['volume_L'])
    
    sim = UreaseSimulator(
        S0=S0,
        N0=0.0,
        C0=0.0,
        Pt_total_M=0.0,
        T_K=T_K,
        initial_pH=known_inputs['initial_pH'],
        E_loading_base_g_per_L=E_loading_base_g_per_L,
        use_T_dependent_pH_activity=True,
    )
    
    def residual(params_vec):
        """Residual function for optimization."""
        # Convert vector to dict
        params_dict = {name: params_vec[i] for i, name in enumerate(param_names)}
        
        # Simulate
        sim_params = {
            'a': params_dict.get('activity_scale', 1.0),
            'k_d': params_dict.get('k_d', 0.0),
            't_shift': 0.0,
            'tau_probe': params_dict.get('tau_probe', 0.0),
        }
        
        try:
            pH_sim = sim.simulate_forward(sim_params, t_measured, return_totals=False, apply_probe_lag=False)
            
            # Apply offset if fitting
            if 'pH_offset' in params_dict:
                pH_sim = pH_sim + params_dict['pH_offset']
            
            # Compute residual
            res = pH_sim - pH_measured
            return res
        except:
            # Return large residual if simulation fails
            return np.full_like(pH_measured, 1e6)
    
    # Initial guess vector
    x0 = np.array([initial_guess.get(name, 0.0) for name in param_names])
    
    # Bounds
    bounds = ([param_bounds[name][0] for name in param_names],
              [param_bounds[name][1] for name in param_names])
    
    # Optimize
    result = least_squares(
        residual,
        x0,
        bounds=bounds,
        method='trf',  # Trust Region Reflective
        max_nfev=1000,
        ftol=1e-6,
        xtol=1e-6,
    )
    
    # Convert result to dict
    fitted_params = {name: float(result.x[i]) for i, name in enumerate(param_names)}
    
    return fitted_params
