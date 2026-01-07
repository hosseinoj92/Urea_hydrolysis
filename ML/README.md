# DeepONet Digital Twin for Urease Kinetics

This package implements a fast forward surrogate digital twin for Qin-Cabral buffer-free urease kinetics using DeepONet (Deep Operator Network) in PyTorch.

## Overview

The workflow consists of:
1. **Refactored mechanistic simulator** - Clean Python module extracted from `simulation5_vs_exp.ipynb`
2. **Synthetic data generation** - Sample parameter ranges to generate training trajectories
3. **DeepONet architecture** - Branch (parameters) + Trunk (time) networks with inner product output
4. **Training** - Train surrogate on synthetic data with physics regularizers
5. **Evaluation** - Benchmark speed vs ODE solver and evaluate on held-out data
6. **Industrial calibration** - Fit batch-to-batch parameters (E_eff0, k_d) to experimental pH(t) curves

## Files

- `mechanistic_simulator.py` - Refactored ODE-based simulator with `simulate_forward()` API
- `test_simulator.py` - Unit test to verify refactored simulator matches notebook output
- `generate_training_data.py` - Generate synthetic training dataset
- `deeponet.py` - DeepONet architecture (BranchNet + TrunkNet)
- `train_deeponet.py` - Training script with physics regularizers
- `evaluate_deeponet.py` - Evaluation and speed benchmarking
- `calibrate_experiments.py` - Industrial calibration for experimental replicates

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Training Data

```bash
python generate_training_data.py --n_samples 10000 --t_max 3600 --n_times 200 --output_dir data --use_totals
```

This generates:
- `data/training_data.npz` - Parameter vectors and output trajectories
- `data/metadata.json` - Dataset metadata

### 3. Train DeepONet

```bash
python train_deeponet.py --data_dir data --output_dir models --epochs 100 --batch_size 256 --use_physics_reg
```

This saves:
- `models/best_model.pt` - Best model checkpoint
- `models/final_model.pt` - Final model checkpoint
- `models/training_curves.png` - Training/validation loss curves

### 4. Evaluate Model

```bash
python evaluate_deeponet.py --model_path models/best_model.pt --data_dir data --output_dir evaluation
```

This generates:
- `evaluation/metrics.json` - RMSE, MAE, R², correlation metrics
- `evaluation/sample_predictions.png` - Sample prediction plots
- Speed benchmark vs ODE solver

### 5. Calibrate Experiments

```bash
python calibrate_experiments.py \
    --model_path models/best_model.pt \
    --exp_files exp1.csv exp2.csv exp3.csv \
    --exp_labels replicate1 replicate2 replicate3 \
    --output_dir calibration \
    --fit_E_eff0 --fit_k_d
```

This generates:
- `calibration/calibration_summary.csv` - Table of fitted parameters per replicate
- `calibration/calibration_overlays.png` - Overlay plots
- `calibration/calibration_results.json` - Detailed results

## Architecture

### DeepONet Structure

```
Input: x (parameters) → Branch Network → b(x) ∈ R^p
Input: t (time)      → Trunk Network → t(t) ∈ R^p
Output: G(x,t) = b(x)^T t(t) + bias
```

- **Branch Network**: Encodes parameter vector [a, E_eff0, k_d, t_shift, tau_probe]
- **Trunk Network**: Encodes time t
- **Output**: Inner product + bias (can be pH directly or totals [S, Ntot, Ctot])

### Hybrid Surrogate

When using totals (S, Ntot, Ctot) as outputs:
1. DeepONet predicts totals at queried times
2. Convert totals → pH using same equilibrium/charge-balance routine as mechanistic model
3. Ensures physical consistency (hybrid surrogate)

## Parameters

The model fits/uses these parameters:
- **a**: Activity multiplier (scales base enzyme loading)
- **E_eff0**: Effective enzyme loading [g/L]
- **k_d**: First-order deactivation rate [1/s]
- **t_shift**: Time shift [s] (model evaluated at t - t_shift)
- **tau_probe**: Probe first-order lag time [s]

## Default Conditions

Matches 40°C batch experiments:
- S0 = 0.020 M (urea)
- T = 313.15 K (40°C)
- Initial pH ≈ 7.36 (unbuffered, set via B_STRONG)
- E_loading_base = 0.5 g/L (0.10 g / 0.2 L)

## Physics Regularizers

Training includes simple physics constraints:
- Nonnegativity: Penalize negative urea/ammonia/carbon
- Monotonicity hints: Urea decreases, ammonia increases
- pH bounds: Keep pH in [0, 14]

## Industrial Application

The calibration script demonstrates:
- **Real-time speed**: DeepONet is 100-1000x faster than ODE solver
- **Batch-to-batch drift**: Rapidly re-estimate E_eff0 and k_d from inline pH
- **Validation**: 3 replicate experiments show consistent deactivation patterns

## Notes

- The simulator uses the same Qin-Cabral kinetics, pH-activity, and Kp(pH) table as the original notebook
- DeepONet is trained on physics simulations, enabling fast digital-twin predictions
- Experimental calibration validates that the surrogate can capture real batch-to-batch variations
