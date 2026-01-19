# Traditional ML Models for Early Inference

This folder contains scripts for training and evaluating traditional machine learning models (GPR, XGBoost, LightGBM, etc.) for early inference parameter estimation.

## Files

- `train_traditional_ml.py` - Train multiple traditional ML models
- `evaluate_traditional_ml.py` - Evaluate models and compare with mechanistic fitting
- `README.md` - This file

## Installation

Install required packages:

```bash
pip install scikit-learn xgboost lightgbm catboost pandas numpy matplotlib scipy tqdm
```

Note: CatBoost is optional. If not installed, set `train_catboost: False` in the config.

## Usage

### 1. Training Models

Edit `train_traditional_ml.py` to configure:
- Data directory
- Output directory
- Which models to train (boolean flags)
- Model hyperparameters

Then run:

```bash
python train_traditional_ml.py
```

**Model Selection:**
Set these flags in `CONFIG` to enable/disable models:
- `train_gpr`: Gaussian Process Regression
- `train_xgboost`: XGBoost
- `train_lightgbm`: LightGBM
- `train_catboost`: CatBoost (requires `pip install catboost`)
- `train_random_forest`: Random Forest
- `train_extra_trees`: Extra Trees
- `train_mlp`: Multi-layer Perceptron

**Features:**
The models use engineered features from the pH sequence:
- Statistical features (mean, std, min, max, initial, final, range, change)
- Time features (mean, std, max, final)
- Rate features (mean and max rate of pH change)
- Early vs late comparison (first 30% vs last 30% of sequence)
- Known inputs (substrate_mM, grams_urease_powder, temperature_C, initial_pH, volume_L)

### 2. Evaluating Models

Edit `evaluate_traditional_ml.py` to configure:
- Models directory (where trained models are saved)
- Data directory
- Output directory
- Number of test samples

Then run:

```bash
python evaluate_traditional_ml.py
```

This will:
- Load all trained models
- Generate test cases
- Evaluate each model's parameter estimation and trajectory forecasting
- Compare with mechanistic fitting
- Save aggregate metrics to CSV and JSON

## Output Files

### Training Outputs

- `{model_name}_prefix_{length}s.pkl` - Trained model (one per model)
- `scaler_prefix_{length}s.pkl` - Feature scaler (for GPR and MLP)
- `training_metadata_prefix_{length}s.json` - Training metadata and metrics

### Evaluation Outputs

- `aggregate_metrics.csv` - Per-sample metrics for all models
- `metrics.json` - Aggregate statistics

## Model Comparison

The evaluation compares:
- **Parameter Estimation**: MAE for `powder_activity_frac` and `k_d`
- **Trajectory Forecasting**: RMSE and R² for full pH trajectory
- **Baseline**: Mechanistic parameter fitting

## Notes

- GPR and MLP use scaled features (StandardScaler)
- Tree-based models (XGBoost, LightGBM, CatBoost, Random Forest, Extra Trees) use raw features
- GPR may be slow for large datasets (>10k samples) - automatically uses subset
- All models predict both `powder_activity_frac` and `k_d`
