"""
Example script demonstrating the full DeepONet workflow:
1. Generate training data
2. Train model
3. Evaluate model
4. Calibrate experiments (if experimental data available)

This is a convenience script that runs the full pipeline.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: {description} failed!")
        print(result.stderr)
        sys.exit(1)
    else:
        print(result.stdout)
        print(f"✓ {description} completed successfully")


def main():
    """Run the full workflow."""
    
    # Step 1: Generate training data
    run_command(
        [
            sys.executable, "generate_training_data.py",
            "--n_samples", "10000",
            "--t_max", "3600",
            "--n_times", "200",
            "--output_dir", "data",
            "--use_totals"
        ],
        "Step 1: Generate training data"
    )
    
    # Step 2: Train DeepONet
    run_command(
        [
            sys.executable, "train_deeponet.py",
            "--data_dir", "data",
            "--output_dir", "models",
            "--epochs", "100",
            "--batch_size", "256",
            "--use_physics_reg"
        ],
        "Step 2: Train DeepONet"
    )
    
    # Step 3: Evaluate model
    model_path = Path("models/best_model.pt")
    if model_path.exists():
        run_command(
            [
                sys.executable, "evaluate_deeponet.py",
                "--model_path", str(model_path),
                "--data_dir", "data",
                "--output_dir", "evaluation",
                "--n_test_samples", "100"
            ],
            "Step 3: Evaluate DeepONet"
        )
    else:
        print(f"Warning: Model not found at {model_path}, skipping evaluation")
    
    # Step 4: Calibrate experiments (optional - requires experimental data)
    print("\n" + "="*60)
    print("Step 4: Calibrate experiments (SKIPPED - requires experimental CSV files)")
    print("="*60)
    print("To calibrate experiments, run:")
    print("  python calibrate_experiments.py \\")
    print("    --model_path models/best_model.pt \\")
    print("    --exp_files exp1.csv exp2.csv exp3.csv \\")
    print("    --exp_labels replicate1 replicate2 replicate3 \\")
    print("    --fit_E_eff0 --fit_k_d")
    
    print("\n" + "="*60)
    print("WORKFLOW COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  - data/training_data.npz - Training dataset")
    print("  - models/best_model.pt - Trained model")
    print("  - evaluation/metrics.json - Evaluation metrics")
    print("\nNext steps:")
    print("  1. Review evaluation/metrics.json for model performance")
    print("  2. Use models/best_model.pt for fast forward predictions")
    print("  3. Run calibrate_experiments.py with your experimental data")


if __name__ == "__main__":
    main()
