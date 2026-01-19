# Training Improvements Summary

## Overview

This document summarizes the improvements made to address training plateaus and improve learning, especially for longer prefix lengths.

## Changes Implemented

### 1. Weighted Temporal Pooling (CRITICAL for Long Prefixes)

**File**: `early_inference_model.py`

**Problem**: `AdaptiveAvgPool1d(1)` averages over the entire sequence, diluting early-time information in longer prefixes.

**Solution**: Replaced with learnable attention-based weighted pooling that can emphasize early-time information.

**Implementation**:
- Added `WeightedTemporalPooling` class with learnable attention mechanism
- Uses Conv1d layers to compute attention weights per time step
- Allows model to focus on early dynamics even in long sequences
- Includes initialization bias toward early time points (exponential decay)

**Impact**: 
- Better handling of longer prefixes (60s, 300s)
- Model can learn to focus on informative early-time dynamics
- Prevents information dilution from simple averaging

### 2. Training Hyperparameter Improvements

**File**: `train_early_inference.py`

**Changes**:
- **Batch size**: Increased from 128 → 256 (better GPU utilization, more stable gradients)
- **Learning rate**: Increased from 5e-4 → 1e-3 (scaled with batch size)
- **Weight decay**: Added 1e-5 (L2 regularization to prevent overfitting)
- **Gradient clipping**: Increased max_norm from 1.0 → 5.0 (better gradient flow)

**Impact**:
- Faster training with larger batches
- More stable gradients
- Better convergence with higher initial LR

### 3. Learning Rate Scheduler Improvements

**File**: `train_early_inference.py`

**Changes**:
- **Scheduler factor**: Changed from 0.5 → 0.7 (less aggressive reduction)
- **Scheduler patience**: Increased from 5 → 10 epochs (more patience)
- **Minimum LR**: Added 1e-6 floor (prevents LR from going too low)
- **Warmup**: Added 5-epoch warmup period (stable training start)

**Implementation**:
- Uses `SequentialLR` to combine warmup and plateau schedulers
- Warmup: Linearly increases LR from 0 to target over 5 epochs
- Plateau: Reduces LR by factor 0.7 when validation loss plateaus

**Impact**:
- Prevents premature LR reduction
- More stable training start
- Better convergence in later epochs

### 4. Loss Function Improvements

**File**: `early_inference_model.py` - `gaussian_nll_loss()`

**Changes**:
- Added variance regularization penalty
- Prevents variance collapse/explosion
- Encourages reasonable uncertainty estimates (variance ~0.1)

**Impact**:
- More stable training
- Prevents model from "cheating" by inflating variance
- Better uncertainty quantification

### 5. Adaptive Prefix Points (Optional Feature)

**File**: `generate_early_inference_data.py`

**Changes**:
- Added `get_adaptive_prefix_n_points()` function
- Optionally scales `prefix_n_points` with `prefix_length` to maintain temporal resolution
- Maintains target dt ≈ 0.3s across different prefix lengths

**Configuration**:
```python
"use_adaptive_n_points": False,  # Enable to use adaptive scaling
"target_dt": 0.3,  # Target temporal resolution [s]
```

**Impact**:
- When enabled, longer prefixes get more points (maintains resolution)
- Prevents information loss from sparse sampling in long prefixes
- Currently disabled by default (backward compatible)

### 6. Documentation Updates

**Files**: `generate_early_inference_data.py`, `train_early_inference.py`

**Added**:
- Comments explaining n_times effect (doesn't affect training data)
- Documentation for new configuration options
- Notes about batch size and learning rate scaling

## Configuration Summary

### Recommended Settings

```python
# In train_early_inference.py CONFIG:
{
    "batch_size": 256,  # Increased from 128
    "lr": 1e-3,  # Increased from 5e-4
    "weight_decay": 1e-5,  # NEW
    "grad_clip_norm": 5.0,  # Increased from 1.0
    "scheduler_factor": 0.7,  # Less aggressive (was 0.5)
    "scheduler_patience": 10,  # More patience (was 5)
    "scheduler_min_lr": 1e-6,  # NEW
    "warmup_epochs": 5,  # NEW
}
```

### Optional: Adaptive Prefix Points

```python
# In generate_early_inference_data.py CONFIG:
{
    "use_adaptive_n_points": True,  # Enable for longer prefixes
    "target_dt": 0.3,  # Maintain 0.3s resolution
}
```

**Example**: 
- 30s prefix → 100 points (dt=0.3s)
- 60s prefix → 200 points (dt=0.3s)
- 300s prefix → 1000 points (dt=0.3s)

## Expected Improvements

1. **Better Learning for Long Prefixes**:
   - Weighted pooling prevents information dilution
   - Model can focus on early-time dynamics

2. **Faster Convergence**:
   - Higher initial LR + warmup = faster learning
   - Larger batches = more stable gradients

3. **More Stable Training**:
   - Less aggressive LR reduction = fewer plateaus
   - Variance regularization = stable loss

4. **Better GPU Utilization**:
   - Batch size 256 uses GPU more efficiently
   - Faster training overall

## Testing Recommendations

1. **Monitor Training**:
   - Check if loss decreases more smoothly
   - Verify LR warmup works (should see LR increase in first 5 epochs)
   - Check if longer prefixes (300s) train better

2. **Compare Results**:
   - Train with old config (batch=128, lr=5e-4) vs new config
   - Compare final validation loss
   - Check if longer prefixes perform better

3. **Tune if Needed**:
   - If still plateauing: increase `warmup_epochs` to 10
   - If overfitting: increase `weight_decay` to 1e-4
   - If underfitting: increase `lr` to 2e-3

## Notes

- All changes are backward compatible (defaults preserve old behavior where possible)
- Weighted pooling is always enabled (replaces AdaptiveAvgPool1d)
- Adaptive prefix points is optional (disabled by default)
- Batch size and LR changes require retraining from scratch

## Files Modified

1. `early_inference_model.py`:
   - Added `WeightedTemporalPooling` class
   - Updated `EarlyInferenceModel` to use weighted pooling
   - Improved `gaussian_nll_loss` with variance regularization

2. `train_early_inference.py`:
   - Updated CONFIG with new hyperparameters
   - Added warmup + plateau scheduler
   - Increased gradient clipping threshold
   - Added weight decay to optimizer

3. `generate_early_inference_data.py`:
   - Added `get_adaptive_prefix_n_points()` function
   - Added adaptive prefix points option (disabled by default)
   - Added documentation about n_times
