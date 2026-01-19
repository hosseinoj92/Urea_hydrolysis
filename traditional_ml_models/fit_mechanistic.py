"""
Mechanistic parameter fitting using least squares optimization.
Parameterization: fits powder_activity_frac and k_d (computes E0_g_per_L from powder_activity_frac).

Part B: Extended to support nuisance parameters (measurement effects, gas exchange, mixing)
with regularization to prevent overfitting.
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
    # Part B: Nuisance parameter configuration
    fit_nuisance_params: bool = False,  # If True, fit nuisance parameters
    nuisance_param_bounds: Dict[str, Tuple[float, float]] = None,
    nuisance_initial_guess: Dict[str, float] = None,
    nuisance_regularization: Dict[str, float] = None,  # Regularization strength per parameter
    # Part B: Real-world effects configuration
    enable_measurement_effects: bool = False,
    enable_gas_exchange: bool = False,
    enable_mixing_ramp: bool = False,
    # Part B: Objective configuration
    use_integral_objective: bool = True,  # Use integral-based objective (insensitive to sampling density)
) -> Dict[str, float]:
    """
    Fit mechanistic parameters using least squares optimization.
    
    Parameterization: fits powder_activity_frac and k_d (computes E0_g_per_L from powder_activity_frac).
    
    Part B: Extended with:
    - Nuisance parameter fitting (measurement effects, gas exchange, mixing)
    - Regularization on nuisance parameters to prevent overfitting
    - Integral-based objective (insensitive to sampling density)
    
    Parameters
    ----------
    pH_measured: (n_points,) array of measured pH values
    t_measured: (n_points,) array of time points
    known_inputs: dict with 5 known inputs (substrate_mM, grams_urease_powder, 
                  temperature_C, initial_pH, volume_L)
    param_bounds: dict with parameter bounds, e.g., {'powder_activity_frac': (0.01, 1.0), 'k_d': (0.0, 5e-3)}
    initial_guess: dict with initial parameter guesses
    fit_nuisance_params: if True, fit nuisance parameters (measurement effects, etc.)
    nuisance_param_bounds: bounds for nuisance parameters
    nuisance_initial_guess: initial guesses for nuisance parameters
    nuisance_regularization: regularization strength (L2 penalty) per nuisance parameter
    enable_measurement_effects: enable measurement bias/drift/smoothing
    enable_gas_exchange: enable gas exchange in ODE
    enable_mixing_ramp: enable mixing/dispersion ramp
    use_integral_objective: if True, use integral-based objective (recommended)
    
    Returns
    -------
    fitted_params: dict with fitted parameters
    """
    # Default bounds (powder_activity_frac and k_d)
    if param_bounds is None:
        param_bounds = {
            'powder_activity_frac': (0.01, 1.0),  # Fraction of powder that is active enzyme [0-1]
            'k_d': (0.0, 5e-3),                   # Deactivation rate [1/s]
        }
    
    # Default initial guess (ensure it's within bounds)
    if initial_guess is None:
        # Get bounds to ensure initial guess is valid
        default_powder_bounds = param_bounds.get('powder_activity_frac', (0.01, 1.0))
        default_kd_bounds = param_bounds.get('k_d', (0.0, 5e-3))
        
        # Use midpoint of bounds for initial guess
        initial_guess = {
            'powder_activity_frac': 0.5 * (default_powder_bounds[0] + default_powder_bounds[1]),  # Midpoint
            'k_d': max(default_kd_bounds[0], 1e-5),  # Use lower bound (but at least 1e-5 if bounds start at 0)
        }
        # Ensure initial guess is within bounds
        initial_guess['powder_activity_frac'] = np.clip(initial_guess['powder_activity_frac'], 
                                                         default_powder_bounds[0], default_powder_bounds[1])
        initial_guess['k_d'] = np.clip(initial_guess['k_d'], 
                                       default_kd_bounds[0], default_kd_bounds[1])
    
    # Part B: Default nuisance parameter bounds and guesses
    if fit_nuisance_params:
        if nuisance_param_bounds is None:
            nuisance_param_bounds = {}
            if enable_measurement_effects:
                nuisance_param_bounds.update({
                    'pH_offset': (-0.1, 0.1),  # Small offset [pH units]
                    'pH_drift_rate': (-1e-5, 1e-5),  # Slow drift [pH units/s]
                    'tau_smoothing': (0.0, 10.0),  # Instrument smoothing [s]
                })
            if enable_gas_exchange:
                nuisance_param_bounds.update({
                    'gas_exchange_k': (0.0, 1e-4),  # Gas exchange rate [1/s]
                    'gas_exchange_C_eq': (0.0, 0.01),  # Equilibrium C [M]
                })
            if enable_mixing_ramp:
                nuisance_param_bounds.update({
                    'mixing_ramp_time_s': (0.0, 30.0),  # Mixing ramp time [s]
                })
        
        if nuisance_initial_guess is None:
            nuisance_initial_guess = {}
            if enable_measurement_effects:
                nuisance_initial_guess.update({
                    'pH_offset': 0.0,
                    'pH_drift_rate': 0.0,
                    'tau_smoothing': 0.0,
                })
            if enable_gas_exchange:
                nuisance_initial_guess.update({
                    'gas_exchange_k': 0.0,
                    'gas_exchange_C_eq': 0.0,
                })
            if enable_mixing_ramp:
                nuisance_initial_guess.update({
                    'mixing_ramp_time_s': 0.0,
                })
        
        # Part B: Default regularization (soft priors to keep nuisance factors small)
        if nuisance_regularization is None:
            nuisance_regularization = {}
            if enable_measurement_effects:
                nuisance_regularization.update({
                    'pH_offset': 1e2,  # Penalize offset (keep near 0)
                    'pH_drift_rate': 1e8,  # Strongly penalize drift (keep very small)
                    'tau_smoothing': 1e-1,  # Light penalty on smoothing
                })
            if enable_gas_exchange:
                nuisance_regularization.update({
                    'gas_exchange_k': 1e6,  # Penalize gas exchange (keep small unless needed)
                    'gas_exchange_C_eq': 1e2,  # Light penalty on equilibrium
                })
            if enable_mixing_ramp:
                nuisance_regularization.update({
                    'mixing_ramp_time_s': 1e-1,  # Light penalty on mixing ramp
                })
    
    # Parameters to fit: core parameters + optionally nuisance parameters
    param_names = list(param_bounds.keys())
    if fit_nuisance_params:
        param_names.extend(list(nuisance_param_bounds.keys()))
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
    
    # Part B: Sort time points for integral computation
    sort_idx = np.argsort(t_measured)
    t_sorted = t_measured[sort_idx]
    pH_measured_sorted = pH_measured[sort_idx]
    
    # Time weighting parameters (emphasize early times for better parameter estimation)
    # Similar to notebook: weight = 1 / (1 + (t/t_weight)^power)
    t_weight = 400.0  # seconds where weight ~1/2
    weight_power = 2.0  # larger → more emphasis on early data
    time_weights = 1.0 / (1.0 + (np.maximum(t_sorted, 0.0) / max(t_weight, 1e-12)) ** weight_power)
    
    def residual(params_vec):
        """
        Residual function for optimization.
        
        Part B: Uses integral-based objective to be insensitive to sampling density.
        Also includes regularization on nuisance parameters.
        """
        # Convert vector to dict
        params_dict = {name: params_vec[i] for i, name in enumerate(param_names)}
        
        # Compute E0_g_per_L from powder_activity_frac
        powder_activity_frac = params_dict.get('powder_activity_frac', 0.1)
        E0_g_per_L = powder_activity_frac * known_inputs['grams_urease_powder'] / known_inputs['volume_L']
        
        # Separate core and nuisance parameters
        sim_params = {
            'E_eff0': E0_g_per_L,  # Computed from powder_activity_frac
            'k_d': params_dict.get('k_d', 0.0),
            't_shift': 0.0,
            'tau_probe': 0.0,  # Not used (true pH space)
        }
        
        # Add nuisance parameters if fitting them
        if fit_nuisance_params:
            if enable_measurement_effects:
                sim_params['pH_offset'] = params_dict.get('pH_offset', 0.0)
                sim_params['pH_drift_rate'] = params_dict.get('pH_drift_rate', 0.0)
                sim_params['tau_smoothing'] = params_dict.get('tau_smoothing', 0.0)
            if enable_gas_exchange:
                sim_params['gas_exchange_k'] = params_dict.get('gas_exchange_k', 0.0)
                sim_params['gas_exchange_C_eq'] = params_dict.get('gas_exchange_C_eq', 0.0)
            if enable_mixing_ramp:
                sim_params['mixing_ramp_time_s'] = params_dict.get('mixing_ramp_time_s', 0.0)
        
        try:
            # Simulate with real-world effects if enabled
            pH_sim = sim.simulate_forward(
                sim_params, t_sorted, 
                return_totals=False, 
                apply_probe_lag=False,
                enable_measurement_effects=enable_measurement_effects,
                enable_gas_exchange=enable_gas_exchange,
                enable_mixing_ramp=enable_mixing_ramp,
            )
            
            # Compute residuals with time weighting (emphasize early times)
            # This is critical for parameter estimation - early dynamics are most informative
            error = pH_sim - pH_measured_sorted
            
            if use_integral_objective and len(t_sorted) > 1:
                # Weighted integral-based objective
                # Weight the squared error by time weights, then integrate
                weighted_error_sq = time_weights * (error ** 2)
                error_integral = np.trapz(weighted_error_sq, t_sorted)
                # Convert to pointwise residual (density-insensitive but time-weighted)
                res_per_point = np.sqrt(error_integral / len(t_sorted))
                res = np.full_like(pH_measured_sorted, res_per_point)
            else:
                # Weighted pointwise residuals (matches notebook approach)
                # Apply sqrt(weight) to residuals to emphasize early times
                res = np.sqrt(time_weights) * error
            
            # Part B: Add regularization terms for nuisance parameters
            # This prevents nuisance factors from freely compensating for kinetics parameters
            if fit_nuisance_params and nuisance_regularization:
                reg_terms = []
                for param_name, reg_strength in nuisance_regularization.items():
                    if param_name in params_dict and reg_strength > 0.0:
                        param_val = params_dict[param_name]
                        # L2 regularization: penalty = reg_strength * param_val^2
                        # Convert to residual scale by taking square root
                        reg_residual = np.sqrt(reg_strength) * abs(param_val)
                        reg_terms.append(reg_residual)
                
                # Append regularization residuals (one per regularized parameter)
                if reg_terms:
                    # Scale regularization to match data residual scale
                    # Use average magnitude of data residuals as reference
                    data_res_scale = np.mean(np.abs(res)) if len(res) > 0 else 1.0
                    reg_scale = data_res_scale / (np.mean(reg_terms) + 1e-12) if reg_terms else 1.0
                    reg_residuals = [r * reg_scale for r in reg_terms]
                    res = np.concatenate([res, reg_residuals])
            
            return res
        except Exception as e:
            # Return large residual if simulation fails
            return np.full(len(pH_measured_sorted) + (len(nuisance_regularization) if fit_nuisance_params and nuisance_regularization else 0), 1e6)
    
    # Initial guess vector (core + nuisance)
    x0 = []
    for name in param_names:
        if name in initial_guess:
            x0.append(initial_guess[name])
        elif fit_nuisance_params and name in nuisance_initial_guess:
            x0.append(nuisance_initial_guess[name])
        else:
            x0.append(0.0)
    x0 = np.array(x0)
    
    # Bounds (core + nuisance)
    bounds_lower = []
    bounds_upper = []
    for name in param_names:
        if name in param_bounds:
            bounds_lower.append(param_bounds[name][0])
            bounds_upper.append(param_bounds[name][1])
        elif fit_nuisance_params and name in nuisance_param_bounds:
            bounds_lower.append(nuisance_param_bounds[name][0])
            bounds_upper.append(nuisance_param_bounds[name][1])
        else:
            bounds_lower.append(-np.inf)
            bounds_upper.append(np.inf)
    bounds = (np.array(bounds_lower), np.array(bounds_upper))
    
    # Ensure initial guess is within bounds (clip to valid range)
    x0 = np.clip(x0, bounds[0], bounds[1])
    
    # Optimize with robust loss (similar to notebook)
    # Use 'soft_l1' robust loss to handle outliers better
    result = least_squares(
        residual,
        x0,
        bounds=bounds,
        method='trf',  # Trust Region Reflective
        max_nfev=1000,
        ftol=1e-8,  # Tighter tolerance (matches notebook)
        xtol=1e-8,  # Tighter tolerance (matches notebook)
        gtol=1e-8,  # Gradient tolerance (matches notebook)
        loss='soft_l1',  # Robust loss function (handles outliers)
        f_scale=0.03,  # Scale parameter for robust loss (pH units, matches notebook)
    )
    
    # Convert result to dict
    fitted_params = {name: float(result.x[i]) for i, name in enumerate(param_names)}
    
    return fitted_params
