"""
Mechanistic simulator for Qin-Cabral buffer-free urease kinetics.
Refactored from simulation5_vs_exp.ipynb for use in DeepONet training.
"""

import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root_scalar
from typing import Dict, Tuple, Optional, Union
import json
import os

# Debug logging
LOG_PATH = r"c:\Users\vt4ho\Simulations\Urease_2\.cursor\debug.log"
def debug_log(location, message, data, hypothesis_id="A"):
    try:
        with open(LOG_PATH, 'a', encoding='utf-8') as f:
            log_entry = {
                "sessionId": "debug-session",
                "runId": "run1",
                "hypothesisId": hypothesis_id,
                "location": location,
                "message": message,
                "data": data,
                "timestamp": int(__import__('time').time() * 1000)
            }
            f.write(json.dumps(log_entry) + '\n')
    except:
        pass

# ╔══════════════════════════════════════════════════════════════╗
# ║                  CONSTANTS & PARAMETERS                      ║
# ╚══════════════════════════════════════════════════════════════╝

R = 8.314  # J/mol/K

# Kinetics (Qin & Cabral, per gram of enzyme)
k0_mol_per_s_per_g = 0.267   # mol NH3/(s·g)
Ea = 29.1e3                   # J/mol
KM, Ks = 2.56e-3, 6.18       # M

# ---------------------------------------------------------------------------
# pH-activity parameters
# ---------------------------------------------------------------------------
# Original Qin–Cabral constants (used if T-dependent pH-activity is off)
pKa_es1_const, pKa_es2_const = 9.07, 5.62
Kes1_const, Kes2_const = 10**(-pKa_es1_const), 10**(-pKa_es2_const)
alpha_e_const, beta_e_const = 0.373, 0.564

# T-dependent pH-activity fit from simulation5_batch.ipynb
# pKes1(T) = b1 + m1*T;  pKes2(T) = b2 + m2*T;  alpha(T) = b3 + m3*T;  beta(T) = b4 + m4*T
_m1, _b1 = 0.03428571428571438, 8.217142857142855
_m2, _b2 = -0.031714285714285605, 6.407142857142855
_m3, _b3 = 0.01131428571428572, 0.09085714285714282
_m4, _b4 = 0.013900000000000027, 0.2199999999999993


def get_pH_activity_params(T_K: float, use_T_dep: bool) -> Tuple[float, float, float, float]:
    """
    Return (Kes1, Kes2, alpha_e, beta_e) for given T [K].

    If use_T_dep is False, return the original Qin–Cabral constants.
    If True, use the linear-in-T fit from simulation5_batch.ipynb
    (valid ~25–40 °C, extrapolated to 20 °C).
    """
    if not use_T_dep:
        return Kes1_const, Kes2_const, alpha_e_const, beta_e_const
    T_C = T_K - 273.15
    pKes1 = _b1 + _m1 * T_C
    pKes2 = _b2 + _m2 * T_C
    alpha = _b3 + _m3 * T_C
    beta = _b4 + _m4 * T_C
    Kes1 = 10**(-pKes1)
    Kes2 = 10**(-pKes2)
    return Kes1, Kes2, float(alpha), float(beta)


# Product inhibition Kp(pH) (table-based, clamped)
_KP_POINTS = [
    (6.25, 0.1785),
    (6.50, 0.1194),
    (7.00, 0.0693),
    (7.50, 0.0386),
    (8.00, 0.0311),
    (8.50, 0.0327),
    (8.75, 0.0298),
    (9.00, 0.0310),
]


def Kp_of_pH(pH: float) -> float:
    """Product inhibition constant as function of pH."""
    pts = _KP_POINTS
    if pH <= pts[0][0]:
        return pts[0][1]
    if pH >= pts[-1][0]:
        return pts[-1][1]
    for (pa, Ka), (pb, Kb) in zip(pts, pts[1:]):
        if pa <= pH <= pb:
            w = (pH - pa) / (pb - pa)
            return Ka + w * (Kb - Ka)
    return pts[-1][1]


# Optional competitive phosphate (inactive by default)
Ki_phosphate = 0.010

# Acid–base constants
Kw = 1e-14
Ka_NH4 = 5.62e-10
Ka1 = 4.45e-7
Ka2 = 4.69e-11
Ka1p = 7.11e-3
Ka2p = 6.32e-8
Ka3p = 4.22e-13


class UreaseSimulator:
    """
    Mechanistic simulator for urease-urea kinetics with pH-dependent activity.
    
    Default conditions match 40°C batch experiments:
    - S0 = 0.020 M (urea)
    - T = 313.15 K (40°C)
    - Initial pH ~7.36 (unbuffered, set via B_STRONG)
    """
    
    def __init__(
        self,
        S0: float = 0.020,  # Initial urea [M]
        N0: float = 0.0,     # Initial total ammonia [M]
        C0: float = 0.0,     # Initial total inorganic carbon [M]
        Pt_total_M: float = 0.0,  # Phosphate [M]
        T_K: float = 313.15,  # Temperature [K]
        initial_pH: float = 7.36,  # Initial pH (sets B_STRONG)
        E_loading_base_g_per_L: float = 0.5,  # Base enzyme loading [g/L]
        use_T_dependent_pH_activity: bool = True,
    ):
        self.S0 = S0
        self.N0 = N0
        self.C0 = C0
        self.Pt_total_M = Pt_total_M
        self.T_K = T_K
        self.initial_pH = initial_pH
        self.E_loading_base_g_per_L = E_loading_base_g_per_L
        self.use_T_dependent_pH_activity = use_T_dependent_pH_activity
        
        # Initialize B_STRONG from initial pH
        self.B_STRONG = self._B_for_target_pH(initial_pH, N0, C0, Pt_total_M)
        self._spec_last_logH = {"val": initial_pH}
    
    def _B_for_target_pH(self, pH_target: float, Ntot: float, Ctot: float, Ptot: float) -> float:
        """Calculate strong-ion background (Na+ - Cl-) for target pH."""
        H = 10**(-pH_target)
        OH = Kw / H
        NH4 = Ntot * (H / (H + Ka_NH4))
        denom_c = H*H + Ka1*H + Ka1*Ka2
        HCO3 = Ctot * (Ka1 * H / denom_c)
        CO3 = Ctot * (Ka1 * Ka2 / denom_c)
        Dp = H**3 + Ka1p*H**2 + Ka1p*Ka2p*H + Ka1p*Ka2p*Ka3p
        H2PO4 = Ptot * (Ka1p * H**2 / Dp)
        HPO4 = Ptot * (Ka1p * Ka2p * H / Dp)
        PO4 = Ptot * (Ka1p * Ka2p * Ka3p / Dp)
        return (OH + HCO3 + 2*CO3 + H2PO4 + 2*HPO4 + 3*PO4) - (H + NH4)
    
    def _charge_balance(self, logH: float, Ntot: float, Ctot: float, Ptot: float) -> float:
        """Charge balance equation for pH calculation."""
        H = 10**(-logH)
        OH = Kw / H
        NH4 = Ntot * (H / (H + Ka_NH4))
        denom_c = H*H + Ka1*H + Ka1*Ka2
        HCO3 = Ctot * (Ka1 * H / denom_c)
        CO3 = Ctot * (Ka1 * Ka2 / denom_c)
        Dp = H**3 + Ka1p*H**2 + Ka1p*Ka2p*H + Ka1p*Ka2p*Ka3p
        H2PO4 = Ptot * (Ka1p * H**2 / Dp)
        HPO4 = Ptot * (Ka1p * Ka2p * H / Dp)
        PO4 = Ptot * (Ka1p * Ka2p * Ka3p / Dp)
        return (H + NH4) - (OH + HCO3 + 2*CO3 + H2PO4 + 2*HPO4 + 3*PO4) + self.B_STRONG
    
    def compute_speciation(self, Ntot: float, Ctot: float, Ptot: float) -> Dict[str, float]:
        """Compute pH and all species concentrations from totals."""
        Ntot = max(Ntot, 0.0)
        Ctot = max(Ctot, 0.0)
        Ptot = max(Ptot, 0.0)
        
        lo = max(1.0, self._spec_last_logH["val"] - 3.0)
        hi = min(13.0, self._spec_last_logH["val"] + 3.0)
        f_lo = self._charge_balance(lo, Ntot, Ctot, Ptot)
        f_hi = self._charge_balance(hi, Ntot, Ctot, Ptot)
        
        if f_lo * f_hi > 0:
            lo, hi = 1.0, 13.0
        
        sol = root_scalar(
            self._charge_balance,
            bracket=[lo, hi],
            method='brentq',
            args=(Ntot, Ctot, Ptot)
        )
        logH = sol.root
        self._spec_last_logH["val"] = logH
        
        H = 10**(-logH)
        OH = Kw / H
        NH4 = Ntot * (H / (H + Ka_NH4))
        NH3 = Ntot - NH4
        denom_c = H*H + Ka1*H + Ka1*Ka2
        CO2 = Ctot * (H*H / denom_c)
        HCO3 = Ctot * (Ka1 * H / denom_c)
        CO3 = Ctot * (Ka1 * Ka2 / denom_c)
        Dp = H**3 + Ka1p*H**2 + Ka1p*Ka2p*H + Ka1p*Ka2p*Ka3p
        H2PO4 = Ptot * (Ka1p * H**2 / Dp)
        HPO4 = Ptot * (Ka1p * Ka2p * H / Dp)
        PO4 = Ptot * (Ka1p * Ka2p * Ka3p / Dp)
        
        return {
            'pH': -math.log10(H),
            'H': H,
            'OH': OH,
            'NH3': NH3,
            'NH4': NH4,
            'CO2': CO2,
            'HCO3': HCO3,
            'CO3': CO3,
            'H2PO4': H2PO4,
            'HPO4': HPO4,
            'PO4': PO4
        }
    
    def rate_per_g(self, S: float, Ntot: float, pH: float, P_inhib: float = 0.0) -> float:
        """NH3 formation rate per gram of urease [mol/(g·s)]."""
        S = max(S, 0.0)
        H = 10**(-pH)

        # Arrhenius temperature dependence
        kT = k0_mol_per_s_per_g * math.exp(-Ea / (R * self.T_K))

        # pH-activity with optional T-dependent parameters (as in simulation5_batch)
        Kes1, Kes2, alpha_e, beta_e = get_pH_activity_params(
            self.T_K, self.use_T_dependent_pH_activity
        )
        pH_factor = 1.0 / (1.0 + (Kes1 / H) ** alpha_e + (H / Kes2) ** beta_e)
        alpha_comp = 1.0 + (P_inhib / Ki_phosphate if Ki_phosphate > 0.0 else 0.0)
        denom = max(KM * alpha_comp + S + (S*S)/Ks, 1e-12)
        v_sub = kT * pH_factor * (S / denom)
        return v_sub / (1.0 + max(Ntot, 0.0) / Kp_of_pH(pH))
    
    def make_rhs(
        self,
        E0_g_per_L: float,
        k_deact_per_s: float = 0.0,
        use_strip_NH3: bool = False,
        kLa_NH3_s: float = 0.0,
        # Part A: Real-world effects
        mixing_ramp_time_s: float = 0.0,  # Mixing/dispersion: smooth turn-on time [s]
        gas_exchange_k: float = 0.0,  # Gas exchange: first-order rate [1/s]
        gas_exchange_C_eq: float = 0.0,  # Gas exchange: equilibrium C [M]
    ):
        """
        Build RHS(t,y) with optional 1st-order deactivation, NH3 stripping, and real-world effects.
        
        Real-world effects (Part A):
        - mixing_ramp_time_s: Smooth turn-on of reaction rate during initial mixing [s]
        - gas_exchange_k: First-order rate for CO2 exchange with environment [1/s]
        - gas_exchange_C_eq: Equilibrium inorganic carbon concentration [M]
        """
        def rhs(t, y):
            S, Ntot, Ctot = y
            sp = self.compute_speciation(Ntot, Ctot, self.Pt_total_M)
            pH = sp['pH']
            P_inhib = 0.0  # Add H2PO4-/HPO4 if Pt_total_M > 0
            per_g = self.rate_per_g(S, Ntot, pH, P_inhib)
            # Active enzyme (g/L)
            E_active = E0_g_per_L * math.exp(-max(k_deact_per_s, 0.0) * max(t, 0.0))
            
            # Part A: Mixing/dispersion - smooth turn-on of reaction rate
            # Represents non-instantaneous mixing at experiment start
            # Uses smooth sigmoid-like function: 0.5 * (1 + tanh((t - t_ramp/2) / (t_ramp/4)))
            # This gives ~0 at t=0, ~1 at t=t_ramp, smooth transition
            if mixing_ramp_time_s > 0.0:
                if t < mixing_ramp_time_s:
                    # Smooth ramp: goes from 0 to 1 over mixing_ramp_time_s
                    # Using tanh for smooth S-curve
                    ramp_frac = 0.5 * (1.0 + math.tanh((t - mixing_ramp_time_s/2.0) / (mixing_ramp_time_s/6.0)))
                else:
                    ramp_frac = 1.0
            else:
                ramp_frac = 1.0
            
            r_NH3 = per_g * E_active * ramp_frac  # mol/L/s produced (with mixing ramp)
            
            # #region agent log
            if t == 0.0 or (t > 0 and t % 100.0 < 0.1):  # Log at start and periodically
                debug_log("mechanistic_simulator.py:245", "Enzyme activity", {
                    "t": float(t),
                    "k_deact_per_s": float(k_deact_per_s),
                    "E0": float(E0_g_per_L),
                    "E_active": float(E_active),
                    "E_active_frac": float(E_active / E0_g_per_L) if E0_g_per_L > 0 else 0.0,
                    "r_NH3": float(r_NH3),
                    "mixing_ramp_frac": float(ramp_frac)
                }, "F")
            # #endregion
            # Gas loss (mol/L/s)
            r_strp_NH3 = (kLa_NH3_s * sp['NH3']) if use_strip_NH3 else 0.0
            
            # Part A: Gas exchange - CO2 exchange with environment
            # Simple first-order mass transfer toward equilibrium
            # dC/dt includes: production from reaction + exchange term
            r_gas_exchange = 0.0
            if gas_exchange_k > 0.0:
                r_gas_exchange = gas_exchange_k * (gas_exchange_C_eq - Ctot)  # [M/s]
            
            dS_dt = -0.5 * r_NH3
            dN_dt = r_NH3 - r_strp_NH3
            dC_dt = 0.5 * r_NH3 + r_gas_exchange  # Add gas exchange term
            return [dS_dt, dN_dt, dC_dt]
        return rhs
    
    def simulate_forward(
        self,
        params: Dict[str, float],
        t_grid: np.ndarray,
        return_totals: bool = False,
        apply_probe_lag: bool = False,
        # Part A: Real-world effects configuration
        enable_measurement_effects: bool = False,  # Enable measurement layer effects
        enable_gas_exchange: bool = False,  # Enable gas exchange in ODE
        enable_mixing_ramp: bool = False,  # Enable mixing/dispersion ramp
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Forward simulation of pH(t) or totals (urea, total ammonia, total carbon).
        
        Parameters:
        -----------
        params : dict
            Must contain:
            - 'a' or 'activity_scale': activity multiplier (default 1.0)
            - 'E_eff0' or 'E_eff': effective enzyme loading [g/L] (overrides activity_scale)
            - 'k_d' or 'k_deact': deactivation rate [1/s] (default 0.0)
            - 't_shift': time shift [s] (default 0.0)
            - 'tau_probe': probe lag time [s] (default 0.0, only used if apply_probe_lag=True)
            - 'kLa_NH3': NH3 stripping rate [1/s] (default 0.0)
            - Part A - Real-world effects (only used if corresponding enable_* flag is True):
              - 'mixing_ramp_time_s': mixing ramp time [s] (default 0.0)
              - 'gas_exchange_k': gas exchange rate [1/s] (default 0.0)
              - 'gas_exchange_C_eq': equilibrium C [M] (default 0.0)
              - 'pH_offset': measurement offset [pH units] (default 0.0)
              - 'pH_drift_rate': drift rate [pH units/s] (default 0.0)
              - 'tau_smoothing': instrument smoothing time constant [s] (default 0.0)
        t_grid : np.ndarray
            Time points [s] at which to evaluate
        return_totals : bool
            If True, return (S, Ntot, Ctot) instead of pH
        apply_probe_lag : bool
            If True, apply first-order probe lag to pH output
        enable_measurement_effects : bool
            If True, apply measurement bias/drift and smoothing
        enable_gas_exchange : bool
            If True, include gas exchange in ODE
        enable_mixing_ramp : bool
            If True, include mixing/dispersion ramp in reaction rate
        
        Returns:
        --------
        If return_totals=False: pH(t) as np.ndarray
        If return_totals=True: (S, Ntot, Ctot) as tuple of np.ndarray
        """
        # Extract parameters with defaults
        activity_scale = params.get('a', params.get('activity_scale', 1.0))
        E_eff0 = params.get('E_eff0', params.get('E_eff', None))
        if E_eff0 is None:
            E0 = max(activity_scale, 0.0) * self.E_loading_base_g_per_L
        else:
            E0 = max(E_eff0, 0.0)
        
        k_deact = params.get('k_d', params.get('k_deact', 0.0))
        t_shift = params.get('t_shift', 0.0)
        tau_probe = params.get('tau_probe', 0.0)
        kLa_NH3 = params.get('kLa_NH3', 0.0)
        use_strip = (kLa_NH3 > 0.0)
        
        # Part A: Extract real-world effect parameters
        mixing_ramp_time_s = params.get('mixing_ramp_time_s', 0.0) if enable_mixing_ramp else 0.0
        gas_exchange_k = params.get('gas_exchange_k', 0.0) if enable_gas_exchange else 0.0
        gas_exchange_C_eq = params.get('gas_exchange_C_eq', 0.0) if enable_gas_exchange else 0.0
        pH_offset = params.get('pH_offset', 0.0) if enable_measurement_effects else 0.0
        pH_drift_rate = params.get('pH_drift_rate', 0.0) if enable_measurement_effects else 0.0
        tau_smoothing = params.get('tau_smoothing', 0.0) if enable_measurement_effects else 0.0
        
        # #region agent log
        debug_log("mechanistic_simulator.py:295", "k_deact extracted", {
            "k_deact": float(k_deact),
            "k_deact_used": float(max(k_deact, 0.0)),
            "E0": float(E0),
            "t_max": float(t_grid[-1]) if len(t_grid) > 0 else 0.0
        }, "E")
        # #endregion
        
        # Apply time shift
        t_model = np.clip(t_grid - t_shift, 0.0, None)
        
        # Ensure t_model is sorted and unique for solve_ivp
        t_model_sorted, unique_idx = np.unique(t_model, return_index=True)
        use_sorted = len(t_model_sorted) < len(t_model) or not np.all(np.diff(t_model) >= 0)
        
        if use_sorted:
            t_eval = t_model_sorted
        else:
            t_eval = t_model
        
        # Build RHS and integrate (with real-world effects if enabled)
        rhs = self.make_rhs(
            E0,
            k_deact_per_s=max(k_deact, 0.0),
            use_strip_NH3=use_strip,
            kLa_NH3_s=max(kLa_NH3, 0.0),
            mixing_ramp_time_s=mixing_ramp_time_s,
            gas_exchange_k=gas_exchange_k,
            gas_exchange_C_eq=gas_exchange_C_eq,
        )
        y0 = [self.S0, self.N0, self.C0]
        
        sol = solve_ivp(
            rhs,
            [t_eval[0], t_eval[-1]],
            y0,
            method='BDF',
            t_eval=t_eval,
            max_step=60.0,
            rtol=1e-6,
            atol=1e-12
        )
        
        if not sol.success:
            raise RuntimeError(f"Integration failed: {sol.message}")
        
        # Extract totals at evaluated times
        S_eval = sol.y[0, :]
        Ntot_eval = sol.y[1, :]
        Ctot_eval = sol.y[2, :]
        
        # Interpolate back to original t_model if needed
        if use_sorted:
            from scipy.interpolate import interp1d
            S_interp = interp1d(sol.t, S_eval, kind='linear', fill_value='extrapolate')
            Ntot_interp = interp1d(sol.t, Ntot_eval, kind='linear', fill_value='extrapolate')
            Ctot_interp = interp1d(sol.t, Ctot_eval, kind='linear', fill_value='extrapolate')
            S = S_interp(t_model)
            Ntot = Ntot_interp(t_model)
            Ctot = Ctot_interp(t_model)
        else:
            S = S_eval
            Ntot = Ntot_eval
            Ctot = Ctot_eval
        
        if return_totals:
            return S, Ntot, Ctot
        
        # Convert totals → pH (this is the "true" pH from chemistry)
        pH_true = np.empty(len(t_grid), dtype=float)
        for i in range(len(t_grid)):
            sp = self.compute_speciation(Ntot[i], Ctot[i], self.Pt_total_M)
            pH_true[i] = sp['pH']
        
        # Part A: Apply measurement layer effects in order:
        # 1. Probe lag (existing, if enabled)
        # 2. Measurement bias and drift (new)
        # 3. Instrument smoothing (new, beyond probe lag)
        
        # Step 1: Apply probe lag if requested (first-order response)
        if apply_probe_lag and tau_probe > 0.0:
            pH_after_lag = np.empty_like(pH_true, dtype=float)
            pH_after_lag[0] = pH_true[0]
            for i in range(1, len(t_grid)):
                dt = t_grid[i] - t_grid[i-1]  # Use original t_grid for dt
                a = math.exp(-dt / max(tau_probe, 1e-12))
                pH_after_lag[i] = a * pH_after_lag[i-1] + (1 - a) * pH_true[i]
        else:
            pH_after_lag = pH_true
        
        # Step 2: Apply measurement bias and drift (constant offset + linear drift)
        # This represents probe calibration offset and slow drift over time
        if enable_measurement_effects and (pH_offset != 0.0 or pH_drift_rate != 0.0):
            pH_after_bias = pH_after_lag + pH_offset + pH_drift_rate * t_grid
        else:
            pH_after_bias = pH_after_lag
        
        # Step 3: Apply instrument smoothing (low-pass filter beyond probe lag)
        # This represents additional smoothing from instrument electronics/firmware
        # Applied as exponential moving average with time constant tau_smoothing
        if enable_measurement_effects and tau_smoothing > 0.0:
            pH_meas = np.empty_like(pH_after_bias, dtype=float)
            pH_meas[0] = pH_after_bias[0]
            for i in range(1, len(t_grid)):
                dt = t_grid[i] - t_grid[i-1]
                a = math.exp(-dt / max(tau_smoothing, 1e-12))
                pH_meas[i] = a * pH_meas[i-1] + (1 - a) * pH_after_bias[i]
            return pH_meas
        
        # If no measurement effects, return pH after bias (or after lag, or true)
        return pH_after_bias


def simulate_forward(
    params: Dict[str, float],
    t_grid: np.ndarray,
    S0: float = 0.020,
    N0: float = 0.0,
    C0: float = 0.0,
    T_K: float = 313.15,
    initial_pH: float = 7.36,
    E_loading_base_g_per_L: float = 0.5,
    return_totals: bool = False,
    apply_probe_lag: bool = False,
    enable_measurement_effects: bool = False,
    enable_gas_exchange: bool = False,
    enable_mixing_ramp: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Convenience function for forward simulation.
    
    See UreaseSimulator.simulate_forward() for parameter details.
    """
    sim = UreaseSimulator(
        S0=S0,
        N0=N0,
        C0=C0,
        T_K=T_K,
        initial_pH=initial_pH,
        E_loading_base_g_per_L=E_loading_base_g_per_L
    )
    return sim.simulate_forward(
        params, t_grid, 
        return_totals=return_totals, 
        apply_probe_lag=apply_probe_lag,
        enable_measurement_effects=enable_measurement_effects,
        enable_gas_exchange=enable_gas_exchange,
        enable_mixing_ramp=enable_mixing_ramp,
    )
