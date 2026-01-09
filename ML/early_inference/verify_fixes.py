"""
Quick verification that fixes are in place (no heavy computations).
"""

import sys
from pathlib import Path

def check_file_contains(filepath, search_strings, description):
    """Check if file contains expected strings."""
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        missing = []
        for search_str in search_strings:
            if search_str not in content:
                missing.append(search_str)
        
        if missing:
            print(f"❌ FAIL: {description}")
            for m in missing:
                print(f"   Missing: {m}")
            return False
        else:
            print(f"✅ PASS: {description}")
            return True
    except Exception as e:
        print(f"❌ ERROR: {description} - {e}")
        return False


def main():
    print("\n" + "="*70)
    print(" QUICK VERIFICATION: Fixes Applied")
    print("="*70)
    
    checks = []
    
    # Check #1: forecast_early_inference.py has measurement model
    checks.append(check_file_contains(
        "forecast_early_inference.py",
        [
            "def apply_measurement_model",
            "pH_forecast = apply_measurement_model",
            "tau_probe=estimated_params.get('tau_probe', 0.0)",
            "pH_offset=estimated_params.get('pH_offset', 0.0)"
        ],
        "Forecast applies measurement model"
    ))
    
    # Check #2: fit_mechanistic.py has measurement model in residual
    checks.append(check_file_contains(
        "fit_mechanistic.py",
        [
            "def apply_measurement_model",
            "pH_sim = apply_measurement_model",
            "tau_probe=params_dict.get('tau_probe', 0.0)",
            "pH_offset=params_dict.get('pH_offset', 0.0)"
        ],
        "Mechanistic fit applies measurement model"
    ))
    
    # Check #3: evaluate_early_inference.py imports and uses measurement model
    checks.append(check_file_contains(
        "evaluate_early_inference.py",
        [
            "from forecast_early_inference import load_early_inference_model, normalize_inputs, denormalize_outputs, apply_measurement_model",
            "pH_ml = apply_measurement_model",
            "pH_fit = apply_measurement_model"
        ],
        "Evaluation uses measurement model consistently"
    ))
    
    # Check #4: generate_early_inference_data.py has adaptive sampling
    checks.append(check_file_contains(
        "generate_early_inference_data.py",
        [
            "alpha = 2.0",
            "u = np.linspace(0, 1, n_points) ** alpha",
            "Adaptive sampling"
        ],
        "Data generation uses adaptive sampling"
    ))
    
    # Check #5: Documentation files exist
    checks.append(Path("FIXES_SUMMARY.md").exists())
    if checks[-1]:
        print("✅ PASS: Documentation (FIXES_SUMMARY.md) exists")
    else:
        print("❌ FAIL: Documentation missing")
    
    checks.append(Path("QUICK_START_FIXES.md").exists())
    if checks[-1]:
        print("✅ PASS: Quick start guide exists")
    else:
        print("❌ FAIL: Quick start guide missing")
    
    # Summary
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    
    n_passed = sum(checks)
    n_total = len(checks)
    
    print(f"\nPassed: {n_passed}/{n_total} checks")
    
    if n_passed == n_total:
        print("\n✅ All verification checks passed!")
        print("\nNext steps:")
        print("  1. Regenerate data: python generate_early_inference_data.py")
        print("  2. Retrain model: python train_early_inference.py")
        print("  3. Re-evaluate: python evaluate_early_inference.py")
        print("\nSee QUICK_START_FIXES.md for detailed instructions.")
        return 0
    else:
        print("\n⚠️  Some checks failed. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
