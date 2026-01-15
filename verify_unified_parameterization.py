"""
Verification script for unified parameterization (E0_g_per_L + k_d).

Checks that all scripts are consistent with the unified parameterization:
- infer_params = ['E0_g_per_L', 'k_d']
- known_input_names = ['substrate_mM', 'grams_urease_powder', 'temperature_C', 'initial_pH', 'volume_L']
"""

import json
from pathlib import Path
import sys

def check_file(filepath, checks):
    """Check if file contains expected patterns."""
    print(f"\nChecking {filepath.name}...")
    
    if not filepath.exists():
        print(f"  ❌ File not found")
        return False
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    all_passed = True
    for check_name, pattern in checks.items():
        if pattern in content:
            print(f"  ✅ {check_name}")
        else:
            print(f"  ❌ {check_name} - pattern not found: {pattern[:50]}...")
            all_passed = False
    
    return all_passed


def main():
    base_dir = Path(__file__).parent
    
    print("="*60)
    print("UNIFIED PARAMETERIZATION VERIFICATION")
    print("="*60)
    
    all_passed = True
    
    # 1. Check generate_early_inference_data.py
    checks = {
        "infer_params contains E0_g_per_L": '"E0_g_per_L"',
        "infer_params contains k_d": '"k_d"',
        "E0_g_per_L range includes slow cases": '"E0_g_per_L": [5e-4',
        "known_input_names has 5 inputs": '"volume_L",\n        ],  # Unified: exactly 5',
        "Uses E_eff0 in simulator": "'E_eff0': E0_g_per_L",
    }
    all_passed &= check_file(base_dir / "generate_early_inference_data.py", checks)
    
    # 2. Check early_inference_model.py
    checks = {
        "n_known_inputs default is 5": "n_known_inputs: int = 5",
        "n_output_params default is 2": "n_output_params: int = 2",
    }
    all_passed &= check_file(base_dir / "early_inference_model.py", checks)
    
    # 3. Check forecast_early_inference.py
    checks = {
        "Default infer_params is E0_g_per_L, k_d": "['E0_g_per_L', 'k_d']",
        "Default known_input_names has 5 inputs": "'initial_pH', 'volume_L'\n    ])",
        "Uses E_eff0 in simulator": "'E_eff0': estimated_params.get('E0_g_per_L'",
    }
    all_passed &= check_file(base_dir / "forecast_early_inference.py", checks)
    
    # 4. Check fit_mechanistic.py
    checks = {
        "Default bounds for E0_g_per_L": "'E0_g_per_L': (5e-4, 2.5)",
        "Default bounds for k_d": "'k_d': (0.0, 5e-3)",
        "Uses E_eff0 in residual": "'E_eff0': params_dict.get('E0_g_per_L'",
        "No activity_scale": "activity_scale" not in content or "# Removed" in content,
    }
    all_passed &= check_file(base_dir / "fit_mechanistic.py", checks)
    
    # 5. Check evaluate_early_inference.py
    checks = {
        "Unified parameterization comment": "Unified parameterization: E0_g_per_L + k_d",
        "Uses E_eff0 for ML": "'E_eff0': ml_params.get('E0_g_per_L'",
        "Uses E_eff0 for Fit": "'E_eff0': fit_params.get('E0_g_per_L'",
        "Computes derived powder_activity_frac": "powder_activity_frac_derived",
    }
    all_passed &= check_file(base_dir / "evaluate_early_inference.py", checks)
    
    # 6. Check metadata.json if exists
    metadata_path = base_dir / "Generated_Data_EarlyInference_20000" / "metadata.json"
    if metadata_path.exists():
        print(f"\nChecking {metadata_path.name}...")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if metadata.get('infer_params') == ['E0_g_per_L', 'k_d']:
            print(f"  ✅ infer_params = {metadata['infer_params']}")
        else:
            print(f"  ❌ infer_params = {metadata.get('infer_params')} (expected ['E0_g_per_L', 'k_d'])")
            all_passed = False
        
        known_inputs = metadata.get('known_input_names', [])
        if len(known_inputs) == 5 and 'powder_activity_frac' not in known_inputs:
            print(f"  ✅ known_input_names has 5 inputs (no powder_activity_frac)")
        else:
            print(f"  ❌ known_input_names = {known_inputs}")
            all_passed = False
    else:
        print(f"\n⚠️  {metadata_path.name} not found (data not generated yet)")
    
    # Summary
    print("\n" + "="*60)
    if all_passed:
        print("✅ ALL CHECKS PASSED - Unified parameterization verified!")
    else:
        print("❌ SOME CHECKS FAILED - Review output above")
        sys.exit(1)
    print("="*60)


if __name__ == "__main__":
    main()
