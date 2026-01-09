"""
Verification tests for measurement model fixes.

Run this script after implementing the fixes to verify:
1. Measurement model is applied consistently
2. tau_probe is identifiable in mechanistic fitting
3. Adaptive sampling produces correct density distribution
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from mechanistic_simulator import UreaseSimulator
from forecast_early_inference import apply_measurement_model as forecast_apply_mm
from fit_mechanistic import apply_measurement_model as fit_apply_mm
from generate_early_inference_data import extract_prefix


def test_measurement_model_consistency():
    """Test that measurement model is applied identically in forecast and fit."""
    print("\n" + "="*60)
    print("TEST 1: Measurement Model Consistency")
    print("="*60)
    
    # Create dummy pH trajectory
    t = np.linspace(0, 100, 100)
    pH_true = 7.0 + 0.5 * np.log(1 + t/10)  # Logarithmic growth
    
    # Apply measurement model from both modules
    tau_probe = 15.0
    pH_offset = 0.05
    
    pH_forecast = forecast_apply_mm(pH_true, t, tau_probe, pH_offset)
    pH_fit = fit_apply_mm(pH_true, t, tau_probe, pH_offset)
    
    # Check they're identical
    difference = np.abs(pH_forecast - pH_fit).max()
    
    print(f"Maximum difference between forecast and fit measurement models: {difference:.10f}")
    
    if difference < 1e-10:
        print("✅ PASS: Measurement models are consistent")
    else:
        print("❌ FAIL: Measurement models differ!")
        return False
    
    # Check that lag and offset are actually applied
    lag_effect = np.abs(pH_forecast[10] - pH_true[10])
    offset_effect = np.abs(pH_forecast.mean() - pH_true.mean() - pH_offset)
    
    print(f"Lag effect at t=10s: {lag_effect:.4f} pH units (should be > 0.01)")
    print(f"Offset effect: {offset_effect:.4f} pH units (should be ≈ 0)")
    
    if lag_effect > 0.01 and offset_effect < 0.001:
        print("✅ PASS: Lag and offset are applied correctly")
    else:
        print("❌ FAIL: Lag or offset not applied correctly!")
        return False
    
    return True


def test_tau_probe_identifiability():
    """Test that tau_probe affects residual in mechanistic fitting."""
    print("\n" + "="*60)
    print("TEST 2: tau_probe Identifiability in Fitting")
    print("="*60)
    
    # Create test simulator
    sim = UreaseSimulator(
        S0=0.020, N0=0.0, C0=0.0, Pt_total_M=0.0, T_K=313.15,
        initial_pH=7.36, E_loading_base_g_per_L=0.5,
        use_T_dependent_pH_activity=True
    )
    
    # Generate test data with known tau_probe
    t_test = np.linspace(0, 30, 30)
    params_true = {'a': 1.0, 'k_d': 0.001, 't_shift': 0.0, 'tau_probe': 0.0}
    pH_true = sim.simulate_forward(params_true, t_test, return_totals=False, apply_probe_lag=False)
    
    # Apply measurement model with tau_probe = 10s
    pH_measured = fit_apply_mm(pH_true, t_test, tau_probe=10.0, pH_offset=0.0)
    
    # Compute residuals for different tau values
    from fit_mechanistic import apply_measurement_model
    
    tau_values = [0.0, 5.0, 10.0, 15.0, 20.0]
    residuals_rms = []
    
    for tau in tau_values:
        pH_sim = apply_measurement_model(pH_true, t_test, tau, 0.0)
        res = pH_sim - pH_measured
        rms = np.sqrt(np.mean(res**2))
        residuals_rms.append(rms)
    
    print(f"\nRMS residuals for different tau_probe values:")
    for tau, rms in zip(tau_values, residuals_rms):
        marker = " ← MINIMUM" if rms == min(residuals_rms) else ""
        print(f"  tau = {tau:4.1f}s: RMS = {rms:.6f}{marker}")
    
    # Find minimum
    min_idx = np.argmin(residuals_rms)
    best_tau = tau_values[min_idx]
    
    print(f"\nBest tau_probe: {best_tau:.1f}s (true value: 10.0s)")
    
    # Check gradient at true value
    grad = (residuals_rms[3] - residuals_rms[1]) / (tau_values[3] - tau_values[1])
    print(f"Gradient at tau=10s: {grad:.6f} (should be ≈ 0)")
    
    if abs(best_tau - 10.0) < 2.5 and abs(grad) < 0.001:
        print("✅ PASS: tau_probe is identifiable (minimum near true value)")
    else:
        print("❌ FAIL: tau_probe is not properly identifiable!")
        return False
    
    return True


def test_adaptive_sampling():
    """Test that adaptive sampling produces correct density distribution."""
    print("\n" + "="*60)
    print("TEST 3: Adaptive Sampling Density")
    print("="*60)
    
    # Create test trajectory
    t_full = np.linspace(0, 30, 3000)
    pH_full = 7.0 + 0.8 * np.log(1 + t_full/5)
    
    # Extract with adaptive sampling
    t_prefix, pH_prefix = extract_prefix(pH_full, t_full, prefix_length=30.0, n_points=50)
    
    print(f"Extracted {len(t_prefix)} points from prefix of length 30s")
    
    # Compute density in different time windows
    windows = [(0, 10), (10, 20), (20, 30)]
    densities = []
    
    print("\nSampling density by time window:")
    for t_start, t_end in windows:
        mask = (t_prefix >= t_start) & (t_prefix < t_end)
        n_points_in_window = np.sum(mask)
        density = n_points_in_window / (t_end - t_start)
        densities.append(density)
        print(f"  {t_start:2d}s - {t_end:2d}s: {n_points_in_window:2d} points ({density:.2f} points/s)")
    
    # Check that early density is higher
    early_density = densities[0]
    late_density = densities[2]
    ratio = early_density / late_density
    
    print(f"\nDensity ratio (early/late): {ratio:.2f}x")
    print(f"Expected: > 1.5x (more dense early)")
    
    if ratio > 1.5:
        print("✅ PASS: Adaptive sampling produces higher density early")
    else:
        print("❌ FAIL: Adaptive sampling not working correctly!")
        return False
    
    # Visualize
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot 1: Full trajectory with samples
    axes[0].plot(t_full, pH_full, 'b-', alpha=0.3, label='Full trajectory')
    axes[0].plot(t_prefix, pH_prefix, 'ro', markersize=4, label=f'Sampled points (n={len(t_prefix)})')
    axes[0].set_xlabel('Time [s]')
    axes[0].set_ylabel('pH')
    axes[0].set_title('Adaptive Sampling: Dense Early, Sparse Late')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Histogram of sample times
    axes[1].hist(t_prefix, bins=30, color='blue', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Time [s]')
    axes[1].set_ylabel('Number of samples')
    axes[1].set_title('Sample Distribution (should be skewed toward t=0)')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_dir = Path("evaluation_early_inference")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "test_adaptive_sampling.png", dpi=150, bbox_inches='tight')
    print(f"\n📊 Visualization saved to: {output_dir / 'test_adaptive_sampling.png'}")
    plt.close()
    
    return True


def test_forecast_vs_true_pH():
    """Test that forecasts differ from true pH (due to measurement model)."""
    print("\n" + "="*60)
    print("TEST 4: Forecast vs True pH Difference")
    print("="*60)
    
    # Simulate true pH
    sim = UreaseSimulator(
        S0=0.020, N0=0.0, C0=0.0, Pt_total_M=0.0, T_K=313.15,
        initial_pH=7.36, E_loading_base_g_per_L=0.5,
        use_T_dependent_pH_activity=True
    )
    
    t_forecast = np.linspace(0, 100, 100)
    params = {'a': 1.0, 'k_d': 0.001, 't_shift': 0.0, 'tau_probe': 0.0}
    pH_true = sim.simulate_forward(params, t_forecast, return_totals=False, apply_probe_lag=False)
    
    # Apply measurement model
    pH_forecast = forecast_apply_mm(pH_true, t_forecast, tau_probe=15.0, pH_offset=0.05)
    
    # Compute differences
    rmse = np.sqrt(np.mean((pH_forecast - pH_true)**2))
    max_diff = np.abs(pH_forecast - pH_true).max()
    mean_offset = (pH_forecast - pH_true).mean()
    
    print(f"RMSE between forecast and true pH: {rmse:.4f}")
    print(f"Max difference: {max_diff:.4f}")
    print(f"Mean offset: {mean_offset:.4f} (expected ≈ 0.05)")
    
    if rmse > 0.02 and 0.04 < mean_offset < 0.06:
        print("✅ PASS: Forecast differs from true pH (measurement model applied)")
    else:
        print("❌ FAIL: Forecast should differ from true pH!")
        return False
    
    return True


def run_all_tests():
    """Run all verification tests."""
    print("\n" + "="*70)
    print(" MEASUREMENT MODEL FIXES VERIFICATION")
    print("="*70)
    print("\nThis script verifies that the following fixes are working:")
    print("  1. Measurement model consistency (forecast vs fit)")
    print("  2. tau_probe identifiability in mechanistic fitting")
    print("  3. Adaptive sampling (dense early, sparse late)")
    print("  4. Forecasts predict sensor readings (not true pH)")
    
    tests = [
        ("Measurement Model Consistency", test_measurement_model_consistency),
        ("tau_probe Identifiability", test_tau_probe_identifiability),
        ("Adaptive Sampling", test_adaptive_sampling),
        ("Forecast vs True pH", test_forecast_vs_true_pH),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ ERROR in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*70)
    print(" TEST SUMMARY")
    print("="*70)
    
    n_passed = sum(1 for _, passed in results if passed)
    n_total = len(results)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {n_passed}/{n_total} tests passed")
    
    if n_passed == n_total:
        print("\n🎉 All tests passed! Fixes are working correctly.")
        return True
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
