"""
Mechanistic parameter fitting using least squares optimization.
Unified parameterization: fits only E0_g_per_L and k_d in true pH space.

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
    
    Unified parameterization: fits E0_g_per_L and k_d, optionally fits nuisance parameters.
    
    Part B: Extended with:
    - Nuisance parameter fitting (measurement effects, gas exchange, mixing)
    - Regularization on nuisance parameters to prevent overfitting
    - Integral-based objective (insensitive to sampling density)
    
    Parameters
    ----------
    pH_measured: (n_points,) array of measured pH values
    t_measured: (n_points,) array of time points
    known_inputs: dict with 5 known inputs (substrate_mM, grams_urease_powder, 
                  temperature_C, initial_pH, volume_L) - NO powder_activity_frac
    param_bounds: dict with parameter bounds, e.g., {'E0_g_per_L': (5e-4, 2.5), 'k_d': (0.0, 5e-3)}
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
    
    def residual(params_vec):
        """
        Residual function for optimization.
        
        Part B: Uses integral-based objective to be insensitive to sampling density.
        Also includes regularization on nuisance parameters.
        """
        # Convert vector to dict
        params_dict = {name: params_vec[i] for i, name in enumerate(param_names)}
        
        # Separate core and nuisance parameters
        sim_params = {
            'E_eff0': params_dict.get('E0_g_per_L', 0.5),  # Direct enzyme loading [g/L]
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
            
            # Part B: Integral-based objective (insensitive to sampling density)
            # Instead of sum of squared errors, use integral of squared error
            # This approximates continuous-time error and doesn't favor dense sampling
            if use_integral_objective and len(t_sorted) > 1:
                # Compute squared error
                error_sq = (pH_sim - pH_measured_sorted) ** 2
                # Integrate using trapezoidal rule
                # This gives error^2 * dt, which approximates continuous-time integral
                # Use numpy.trapz (scipy.integrate.trapz was deprecated/removed in newer SciPy)
                error_integral = np.trapz(error_sq, t_sorted)
                # Convert to pointwise residual by taking square root and normalizing
                # This preserves the least-squares structure while being density-insensitive
                # We normalize by sqrt(dt_avg) to make units consistent
                dt_avg = (t_sorted[-1] - t_sorted[0]) / (len(t_sorted) - 1) if len(t_sorted) > 1 else 1.0
                # Residual per point: sqrt(integral / n_points) approximates RMS
                res_per_point = np.sqrt(error_integral / len(t_sorted))
                # Return as vector of residuals (one per point, but density-insensitive)
                res = np.full_like(pH_measured_sorted, res_per_point)
            else:
                # Fallback: pointwise residual (original behavior)
                res = pH_sim - pH_measured_sorted
            
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
    bounds = (bounds_lower, bounds_upper)
    
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
