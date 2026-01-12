"""
Mechanistic parameter fitting using least squares optimization.
Unified parameterization: fits only E0_g_per_L and k_d in true pH space.
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
    
    Unified parameterization: fits only E0_g_per_L and k_d.
    
    Parameters
    ----------
    pH_measured: (n_points,) array of measured pH values
    t_measured: (n_points,) array of time points
    known_inputs: dict with 5 known inputs (substrate_mM, grams_urease_powder, 
                  temperature_C, initial_pH, volume_L) - NO powder_activity_frac
    param_bounds: dict with parameter bounds, e.g., {'E0_g_per_L': (5e-4, 2.5), 'k_d': (0.0, 5e-3)}
    initial_guess: dict with initial parameter guesses
    
    Returns
    -------
    fitted_params: dict with keys ['E0_g_per_L', 'k_d']
    """
    # Default bounds (unified: E0_g_per_L and k_d only)
    if param_bounds is None:
        param_bounds = {
            'E0_g_per_L': (5e-4, 1.25),  # Wide range covering slow to fast regimes [g/L]
            'k_d': (0.0, 5e-3),          # Deactivation rate [1/s]
        }
    
    # Default initial guess
    if initial_guess is None:
        initial_guess = {
            'E0_g_per_L': 0.5,  # Reasonable mid-range value
            'k_d': 0.001,       # Small deactivation
        }
    
    # Parameters to fit (only E0_g_per_L and k_d)
    param_names = list(param_bounds.keys())
    n_params = len(param_names)
    
    # Build simulator (dummy base loading, will be overridden by E_eff0)
    S0 = known_inputs['substrate_mM'] / 1000.0
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
    
    def residual(params_vec):
        """Residual function for optimization (true pH space, E_eff0 override)."""
        # Convert vector to dict
        params_dict = {name: params_vec[i] for i, name in enumerate(param_names)}
        
        # Simulate using E_eff0 directly (no activity_scale, no powder_activity_frac)
        sim_params = {
            'E_eff0': params_dict.get('E0_g_per_L', 0.5),  # Direct enzyme loading [g/L]
            'k_d': params_dict.get('k_d', 0.0),
            't_shift': 0.0,
            'tau_probe': 0.0,  # Not used (true pH space)
        }
        
        try:
            # Simulate true pH (no probe lag, no offset)
            pH_sim = sim.simulate_forward(sim_params, t_measured, return_totals=False, apply_probe_lag=False)
            
            # Compute residual (true pH space)
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
