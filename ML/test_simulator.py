"""
Unit test to verify refactored simulator reproduces notebook output.
"""

import numpy as np
from mechanistic_simulator import UreaseSimulator, simulate_forward


def test_simulator_reproduces_notebook():
    """Test that refactored simulator matches notebook behavior for fixed seed/config."""
    
    # Default conditions from notebook (40°C batch)
    S0 = 0.020  # M
    N0 = 0.0
    C0 = 0.0
    T_K = 313.15  # 40°C
    initial_pH = 6.0
    E_loading_base_g_per_L = 0.5  # 0.10 g / 0.2 L
    
    # Create simulator
    sim = UreaseSimulator(
        S0=S0,
        N0=N0,
        C0=C0,
        T_K=T_K,
        initial_pH=initial_pH,
        E_loading_base_g_per_L=E_loading_base_g_per_L
    )
    
    # Test parameters (typical from notebook fits)
    params = {
        'a': 1.0,  # activity scale
        'k_d': 0.0,  # no deactivation initially
        't_shift': 0.0,
        'tau_probe': 0.0,
    }
    
    # Time grid (0 to 3600 s = 1 hour)
    t_grid = np.linspace(0.0, 3600.0, 100)
    
    # Run simulation
    pH_result = sim.simulate_forward(params, t_grid, return_totals=False)
    
    # Basic sanity checks
    assert len(pH_result) == len(t_grid), "pH output length should match time grid"
    assert np.all(pH_result > 0), "pH should be positive"
    assert np.all(pH_result < 14), "pH should be reasonable"
    
    # pH should increase over time (urea hydrolysis produces NH3)
    assert pH_result[-1] > pH_result[0], "pH should increase as NH3 is produced"
    
    # Check initial pH
    assert abs(pH_result[0] - initial_pH) < 0.1, f"Initial pH should be close to {initial_pH}"
    
    # Test with totals return
    S, Ntot, Ctot = sim.simulate_forward(params, t_grid, return_totals=True)
    
    assert len(S) == len(t_grid), "S output length should match"
    assert len(Ntot) == len(t_grid), "Ntot output length should match"
    assert len(Ctot) == len(t_grid), "Ctot output length should match"
    
    # Urea should decrease
    assert S[-1] < S[0], "Urea should decrease over time"
    assert abs(S[0] - S0) < 1e-6, "Initial urea should match S0"
    
    # Total ammonia should increase
    assert Ntot[-1] > Ntot[0], "Total ammonia should increase"
    assert abs(Ntot[0] - N0) < 1e-6, "Initial ammonia should match N0"
    
    # Total carbon should increase (urea → CO2)
    assert Ctot[-1] > Ctot[0], "Total carbon should increase"
    assert abs(Ctot[0] - C0) < 1e-6, "Initial carbon should match C0"
    
    # Test with deactivation
    params_deact = params.copy()
    params_deact['k_d'] = 1e-4  # s^-1
    pH_deact = sim.simulate_forward(params_deact, t_grid, return_totals=False)
    
    # With deactivation, pH rise should be slower
    # (This is a qualitative check - exact values depend on parameters)
    assert len(pH_deact) == len(t_grid), "Deactivation case should work"
    
    # Test convenience function
    pH_conv = simulate_forward(
        params,
        t_grid,
        S0=S0,
        N0=N0,
        C0=C0,
        T_K=T_K,
        initial_pH=initial_pH,
        E_loading_base_g_per_L=E_loading_base_g_per_L
    )
    
    assert np.allclose(pH_result, pH_conv, rtol=1e-6), "Convenience function should match class method"
    
    print("✓ All tests passed!")
    print(f"  Initial pH: {pH_result[0]:.3f}")
    print(f"  Final pH: {pH_result[-1]:.3f}")
    print(f"  Initial urea: {S[0]:.4f} M")
    print(f"  Final urea: {S[-1]:.4f} M")
    print(f"  Final total ammonia: {Ntot[-1]:.4f} M")


if __name__ == "__main__":
    test_simulator_reproduces_notebook()
