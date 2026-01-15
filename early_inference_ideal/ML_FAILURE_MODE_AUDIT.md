# Comprehensive ML Failure Mode Audit
## Early Inference Model for pH Trajectory Parameter Estimation

**Date:** 2024  
**Scope:** End-to-end audit of training, evaluation, and data pipeline  
**Focus:** Issues causing poor ML performance, especially 1s vs 10s sampling generalization

---

## Executive Summary: Top 10 Most Likely Causes

1. **Time normalization destroys temporal information** (CRITICAL)
   - Time normalized globally across all samples → loses absolute time scale
   - Model can't distinguish 0.3s vs 1s vs 10s spacing after normalization
   - **Location:** `train_early_inference.py:83-84, 96`

2. **Prefix extraction uses edge padding, corrupting time grid** (CRITICAL)
   - When `len(t_prefix_full) < n_points`, uses `edge` padding for time
   - Creates duplicate time values → breaks temporal relationships
   - **Location:** `generate_early_inference_data.py:493-494`

3. **Gaussian NLL loss can be negative, masking convergence issues** (HIGH)
   - Formula allows negative values when variance is large
   - Model can "cheat" by inflating variance to reduce loss
   - **Location:** `early_inference_model.py:339-340`

4. **R² computed on normalized values in some paths** (HIGH)
   - If denormalization fails or is skipped, R² on normalized scale is meaningless
   - Negative R² can occur if variance of normalized targets is tiny
   - **Location:** `evaluate_early_inference.py:46-67` (check if inputs are denormalized)

5. **No explicit dt feature - model must infer from normalized time** (MEDIUM)
   - Model only sees normalized time values, not sampling rate
   - Hard to learn that dt=0.3s vs dt=1s vs dt=10s matters
   - **Location:** Model architecture doesn't include dt as explicit feature

6. **Train/val split not seeded, causing non-reproducible leakage** (MEDIUM)
   - Random permutation without fixed seed → different splits each run
   - Could accidentally leak similar samples between train/val
   - **Location:** `train_early_inference.py:274`

7. **pH normalization destroys baseline information** (MEDIUM)
   - Global pH normalization removes absolute pH level
   - Initial pH differences (6.5 vs 7.5) become indistinguishable
   - **Location:** `train_early_inference.py:80-81, 95`

8. **Sequence padding with edge mode corrupts temporal structure** (MEDIUM)
   - Short sequences padded with edge values → creates flat time segments
   - **Location:** `generate_early_inference_data.py:493-494`

9. **No gradient clipping - potential exploding gradients** (LOW)
   - Large parameter ranges (E0: 0.05-1.25, k_d: 1e-5 to 5e-3) can cause instability
   - **Location:** Training loop has no gradient clipping

10. **Model capacity may be insufficient for complex temporal patterns** (LOW)
    - TCN with 3 layers may not capture long-range dependencies
    - **Location:** Architecture choice

---

## Full Checklist by Category

### A) Data Integrity & Leakage

#### A1. Train/Val Split Not Seeded
**Symptom:** Non-reproducible results, potential data leakage between runs  
**Location:** `train_early_inference.py:274`
```python
indices = np.random.permutation(n_samples)  # No seed!
```
**Test:**
```python
# Add after line 273:
np.random.seed(42)  # Or use CONFIG seed
indices = np.random.permutation(n_samples)
# Verify: run twice, check train_indices are identical
```
**Fix:**
```python
# Use fixed seed for reproducibility
rng = np.random.default_rng(CONFIG.get("seed", 42))
indices = rng.permutation(n_samples)
```

#### A2. Normalization Stats Computed on Train Only (CORRECT)
**Status:** ✓ Correctly implemented - train stats used for val  
**Location:** `train_early_inference.py:300-311`  
**Verification:** Val dataset uses `input_stats` and `output_stats` from train_dataset

#### A3. Parameter Ordering Mismatch Risk
**Symptom:** Wrong parameter assigned to wrong output head  
**Location:** `generate_early_inference_data.py:514-515`, `train_early_inference.py:256`  
**Test:**
```python
# In train_early_inference.py after line 256:
assert list(metadata['infer_params']) == ['E0_g_per_L', 'k_d'], \
    f"Param order mismatch! Expected ['E0_g_per_L', 'k_d'], got {metadata['infer_params']}"
```
**Fix:** Add explicit ordering checks throughout pipeline

#### A4. Prefix Length Mismatch Between Data and Config
**Symptom:** Model trained on 30s but data has 10s/60s only  
**Location:** `train_early_inference.py:248-250`  
**Test:**
```python
# Already has check, but add warning:
if prefix_key not in data:
    available = [k for k in data.keys() if k.startswith('prefix_')]
    raise ValueError(f"Prefix {prefix_length}s not found. Available: {available}")
```

#### A5. Time Grid Padding Corrupts Temporal Structure
**Symptom:** Sequences with <100 points get edge-padded, creating duplicate times  
**Location:** `generate_early_inference_data.py:490-497`
```python
if len(pH_seq) < max_len:
    pad_len = max_len - len(pH_seq)
    pH_seq = np.pad(pH_seq, (0, pad_len), mode='edge')  # ❌ BAD for time!
    t_seq = np.pad(t_seq, (0, pad_len), mode='edge')    # ❌ Creates duplicate times
```
**Test:**
```python
# Check for duplicate times after padding:
for i, t_seq in enumerate(t_prefix_padded):
    if len(np.unique(t_seq)) < len(t_seq):
        print(f"WARNING: Sample {i} has duplicate time values!")
```
**Fix:**
```python
# Use linear interpolation instead of edge padding for time:
if len(t_seq) < max_len:
    # Interpolate to uniform grid from 0 to prefix_length
    t_new = np.linspace(0, prefix_length, max_len)
    pH_seq = np.interp(t_new, t_seq, pH_seq)
    t_seq = t_new
```

---

### B) Feature Construction

#### B1. Time Normalization Destroys Absolute Scale (CRITICAL)
**Symptom:** Model can't distinguish 0.3s vs 1s vs 10s spacing after normalization  
**Location:** `train_early_inference.py:83-84, 96`
```python
self.t_mean = np.mean(t_prefix)  # Global mean across ALL samples
self.t_std = np.std(t_prefix)    # Global std across ALL samples
self.t_prefix = (t_prefix - self.t_mean) / self.t_std  # ❌ Loses absolute scale!
```
**Test:**
```python
# After normalization, check if different dt's are distinguishable:
dt_1s = np.diff(t_prefix_norm[0])  # Should be ~constant for uniform grid
dt_10s = np.diff(t_prefix_norm[1])  # Should be ~10x larger
# If normalized, these will be similar → model can't tell difference!
print(f"dt_1s mean: {np.mean(dt_1s):.6f}, dt_10s mean: {np.mean(dt_10s):.6f}")
```
**Fix Option 1: Don't normalize time, or normalize per-sequence**
```python
# Option 1: Don't normalize time (recommended)
if normalize_inputs:
    # ... pH normalization ...
    # DON'T normalize time - keep absolute values
    self.t_prefix = t_prefix  # Keep original time values
    self.t_mean = 0.0
    self.t_std = 1.0
```

**Fix Option 2: Normalize time per-sequence (preserves relative spacing)**
```python
# Normalize each sequence independently to preserve dt relationships
t_prefix_norm = []
for i in range(t_prefix.shape[0]):
    t_seq = t_prefix[i]
    t_mean_i = np.mean(t_seq)
    t_std_i = np.std(t_seq) + 1e-8
    t_prefix_norm.append((t_seq - t_mean_i) / t_std_i)
self.t_prefix = np.array(t_prefix_norm)
```

**Fix Option 3: Add explicit dt as feature**
```python
# In model forward pass, compute dt explicitly:
dt = t_seq[:, 1:] - t_seq[:, :-1]  # (batch, seq_len-1)
dt_padded = torch.cat([dt[:, 0:1], dt], dim=1)  # Pad first element
# Use dt as third channel: (pH, t, dt)
```

#### B2. pH Normalization Destroys Baseline Information
**Symptom:** Initial pH differences (6.5 vs 7.5) become indistinguishable  
**Location:** `train_early_inference.py:80-81, 95`
```python
self.pH_mean = np.mean(pH_prefix)  # Global mean
self.pH_std = np.std(pH_prefix)
self.pH_prefix = (pH_prefix - self.pH_mean) / self.pH_std
```
**Test:**
```python
# Check if initial pH is preserved after normalization:
initial_pH_before = pH_prefix[:, 0]
initial_pH_after = (pH_prefix[:, 0] - self.pH_mean) / self.pH_std
# If std is large, initial_pH differences get compressed
print(f"Initial pH range before: {initial_pH_before.min():.2f} - {initial_pH_before.max():.2f}")
print(f"Initial pH range after: {initial_pH_after.min():.2f} - {initial_pH_after.max():.2f}")
```
**Fix:** Include initial_pH in known_inputs (already done) - this is actually OK since initial_pH is separate feature

#### B3. Missing Explicit dt Feature
**Symptom:** Model must infer sampling rate from normalized time values  
**Location:** Model architecture - no explicit dt channel  
**Test:**
```python
# In model forward, add debug:
dt = t_seq[:, 1:] - t_seq[:, :-1]
print(f"dt range: {dt.min():.6f} - {dt.max():.6f}")  # Should see 0.3s, 1s, 10s
# If all similar after normalization → problem!
```
**Fix:** Add dt as explicit third channel (see B1 Fix Option 3)

#### B4. Known Inputs Order Consistency
**Status:** ✓ Correctly ordered in `generate_early_inference_data.py:553-559`  
**Verification:** Order matches across all files

---

### C) Normalization / Denormalization

#### C1. Output Denormalization Applied Correctly
**Status:** ✓ Correctly implemented in `forecast_early_inference.py:92-110`  
**Location:** `evaluate_early_inference.py:155` uses `denormalize_outputs`

#### C2. R² Computed on Correct Scale
**Symptom:** Negative R² if computed on normalized values or wrong baseline  
**Location:** `evaluate_early_inference.py:46-67`
```python
def compute_metrics(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-12))
```
**Test:**
```python
# In evaluate_single_case, before compute_metrics:
assert not np.any(np.isnan(params)), "Params contain NaN before denormalization"
params_denorm = denormalize_outputs(params_norm, normalization_stats)
# Verify params_denorm are in physical units (E0: 0.05-1.25, k_d: 1e-5 to 5e-3)
assert 0.01 < params_denorm[0] < 2.0, f"E0 out of range: {params_denorm[0]}"
```
**Fix:** Already correct - params are denormalized before metrics

#### C3. Normalization Stats Saved/Loaded Correctly
**Status:** ✓ Stats saved in checkpoint (`train_early_inference.py:394-404`)  
**Status:** ✓ Stats loaded in `forecast_early_inference.py:20`

#### C4. Epsilon in Normalization Prevents Division by Zero
**Status:** ✓ Correct - `1e-8` added in multiple places  
**Location:** `train_early_inference.py:81, 84, 86, 113`

---

### D) Loss Function Correctness

#### D1. Gaussian NLL Loss Can Be Negative
**Symptom:** Loss becomes negative when variance is large, masking poor fits  
**Location:** `early_inference_model.py:324-340`
```python
def gaussian_nll_loss(mean: torch.Tensor, logvar: torch.Tensor, 
                     target: torch.Tensor) -> torch.Tensor:
    precision = torch.exp(-logvar)
    return 0.5 * (logvar + precision * (target - mean) ** 2).mean()
```
**Issue:** When `logvar` is large negative (small variance), loss is negative  
**Test:**
```python
# In training loop, add check:
if use_uncertainty:
    mean, logvar = model(pH_seq, t_seq, known_inputs)
    loss = gaussian_nll_loss(mean, logvar, target_params)
    if loss.item() < 0:
        print(f"WARNING: Negative loss {loss.item():.6f}, logvar range: [{logvar.min():.4f}, {logvar.max():.4f}]")
```
**Fix Option 1: Clamp logvar to prevent extreme values**
```python
def gaussian_nll_loss(mean: torch.Tensor, logvar: torch.Tensor, 
                     target: torch.Tensor, logvar_clamp: float = 10.0) -> torch.Tensor:
    logvar = torch.clamp(logvar, min=-logvar_clamp, max=logvar_clamp)
    precision = torch.exp(-logvar)
    return 0.5 * (logvar + precision * (target - mean) ** 2).mean()
```

**Fix Option 2: Add regularization to prevent variance collapse/explosion**
```python
def gaussian_nll_loss(mean: torch.Tensor, logvar: torch.Tensor, 
                     target: torch.Tensor, var_reg: float = 0.01) -> torch.Tensor:
    precision = torch.exp(-logvar)
    nll = 0.5 * (logvar + precision * (target - mean) ** 2).mean()
    # Regularize logvar to prevent extreme values
    var_reg_loss = var_reg * (logvar ** 2).mean()
    return nll + var_reg_loss
```

#### D2. Loss Reduction (Mean vs Sum)
**Status:** ✓ Correct - uses `.mean()` reduction  
**Location:** `early_inference_model.py:340`

#### D3. Loss Scale with Uncertainty Head
**Symptom:** Loss magnitude changes when switching uncertainty on/off  
**Location:** Training uses different loss functions  
**Test:**
```python
# Compare loss scales:
mse_loss = criterion(mean, target_params)  # MSE
nll_loss = gaussian_nll_loss(mean, logvar, target_params)  # NLL
print(f"MSE: {mse_loss.item():.6f}, NLL: {nll_loss.item():.6f}")
# If scales are very different, may need to adjust learning rate
```

---

### E) Model Architecture / Capacity

#### E1. TCN Input Shape Correct
**Status:** ✓ Correct - (batch, 2, seq_len) for pH + time  
**Location:** `early_inference_model.py:209-211`

#### E2. TCN Receptive Field Sufficient
**Symptom:** Model can't see full prefix if receptive field too small  
**Location:** `early_inference_model.py:72-84`
```python
dilation_size = 2 ** i  # i=0: 1, i=1: 2, i=2: 4
padding = (kernel_size - 1) * dilation_size  # kernel_size=3
# Receptive field = 1 + sum(2 * (kernel_size-1) * 2^i) for i in [0,1,2]
# = 1 + 2*2*1 + 2*2*2 + 2*2*4 = 1 + 4 + 8 + 16 = 29
```
**Test:**
```python
# For seq_length=100, receptive field=29 is sufficient
# But check if model actually uses full sequence:
receptive_field = 1 + sum(2 * (3-1) * (2**i) for i in range(3))
print(f"Receptive field: {receptive_field}, Sequence length: {seq_length}")
assert receptive_field >= seq_length, "Receptive field too small!"
```
**Fix:** Increase TCN depth or kernel size if needed

#### E3. Known Inputs Actually Used
**Status:** ✓ Correct - concatenated with TCN output  
**Location:** `early_inference_model.py:217, 220`

#### E4. Dropout Too High
**Symptom:** Model underfits, can't learn complex patterns  
**Location:** `train_early_inference.py:36, 38` - dropout=0.2, 0.1  
**Test:**
```python
# Try reducing dropout:
"tcn_dropout": 0.1,  # Reduce from 0.2
"output_dropout": 0.05,  # Reduce from 0.1
```

#### E5. No Batch Normalization
**Symptom:** Training instability, slow convergence  
**Location:** `early_inference_model.py:14-55` - TemporalBlock has no BatchNorm  
**Fix:**
```python
# Add BatchNorm1d to TemporalBlock:
self.bn1 = nn.BatchNorm1d(n_outputs)
self.bn2 = nn.BatchNorm1d(n_outputs)
# Use in forward:
out = self.bn1(self.conv1(x))
```

---

### F) Optimization & Training Dynamics

#### F1. Learning Rate Too High/Low
**Symptom:** Loss doesn't decrease or oscillates wildly  
**Location:** `train_early_inference.py:29` - lr=1e-3  
**Test:**
```python
# Monitor gradient norms:
total_norm = 0
for p in model.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
total_norm = total_norm ** (1. / 2)
print(f"Gradient norm: {total_norm:.6f}")
# If > 10, LR too high; if < 0.01, LR too low
```

#### F2. No Gradient Clipping
**Symptom:** Exploding gradients, NaN losses  
**Location:** Training loop has no clipping  
**Fix:**
```python
# In train_epoch, after loss.backward():
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

#### F3. Early Stopping Patience Too Short
**Symptom:** Training stops before convergence  
**Location:** `train_early_inference.py:31` - patience=10  
**Test:** Check if val loss still decreasing when early stopping triggers

#### F4. Scheduler Behavior
**Status:** ✓ Using ReduceLROnPlateau correctly  
**Location:** `train_early_inference.py:340, 374`

---

### G) Evaluation Pitfalls

#### G1. Negative R² Causes
**Symptom:** R² < 0 indicates predictions worse than mean baseline  
**Location:** `evaluate_early_inference.py:46-67`  
**Possible Causes:**
1. Computing R² on normalized values (should be on denormalized)
2. Wrong baseline (using wrong mean)
3. Tiny target variance (ss_tot ≈ 0)
4. Parameter mismatch (predicting wrong param)

**Test:**
```python
# In compute_metrics, add diagnostics:
ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
if ss_tot < 1e-12:
    print(f"WARNING: Target variance too small: {ss_tot:.2e}")
    print(f"  y_true range: [{y_true.min():.6f}, {y_true.max():.6f}]")
    print(f"  y_true mean: {np.mean(y_true):.6f}")
# Check if predictions are in correct units:
print(f"y_pred range: [{y_pred.min():.6f}, {y_pred.max():.6f}]")
```

#### G2. ML vs Fit Comparison Apples-to-Apples
**Status:** ✓ Both use same prefix and same simulator  
**Location:** `evaluate_early_inference.py:134-135, 172-176`

#### G3. Metrics Computed on Reference Grid
**Status:** ✓ All metrics computed on uniform reference grid  
**Location:** `evaluate_early_inference.py:206-230`

---

### H) Domain Shift: Synthetic → Experimental

#### H1. Interpolation Artifacts Not in Training
**Symptom:** Model fails on interpolated 1s data if trained on uniform 0.3s  
**Location:** `generate_early_inference_data.py:209-245` - extract_prefix interpolates  
**Test:**
```python
# Check if training data includes interpolation:
# Training uses uniform prefix extraction (extract_prefix)
# If experimental data is interpolated differently, mismatch occurs
```

#### H2. Noise Model Mismatch
**Status:** Training uses noise_std=0.01, should match experimental  
**Location:** `generate_early_inference_data.py:66`

#### H3. Simulator Consistency
**Status:** ✓ Same simulator used for training and fitting  
**Location:** Both use `UreaseSimulator`

---

### I) Reproducibility & Debugging Aids

#### I1. Add Shape Assertions
**Fix:**
```python
# In train_early_inference.py, after loading data:
assert pH_prefix.shape[0] == t_prefix.shape[0] == known_inputs.shape[0] == target_params.shape[0]
assert pH_prefix.shape[1] == t_prefix.shape[1]
assert known_inputs.shape[1] == 5
assert target_params.shape[1] == 2
```

#### I2. Add Range Checks
**Fix:**
```python
# Check parameter ranges:
assert np.all(target_params[:, 0] >= 0.01), "E0_g_per_L too small"
assert np.all(target_params[:, 0] <= 2.0), "E0_g_per_L too large"
assert np.all(target_params[:, 1] >= 0), "k_d negative"
assert np.all(target_params[:, 1] <= 0.01), "k_d too large"
```

#### I3. Plot Sample Batches
**Fix:**
```python
# In train_epoch, add periodic visualization:
if epoch % 10 == 0:
    # Plot first batch
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    # Plot pH sequences, time sequences, predictions vs targets
    plt.savefig(f"debug_batch_epoch_{epoch}.png")
```

#### I4. Overfit Test
**Fix:**
```python
# Create tiny dataset (10 samples) and verify model can overfit:
tiny_dataset = EarlyInferenceDataset(train_pH[:10], train_t[:10], ...)
# Train for 100 epochs, should achieve near-zero loss
```

---

## Triage Plan: Fast Experiments (<1 hour)

### Experiment 1: Fix Time Normalization (15 min)
**Action:** Remove time normalization or normalize per-sequence  
**Expected:** Model should learn to distinguish different sampling rates  
**Test:** Train on 30s prefix, evaluate on 1s and 10s interpolated data

### Experiment 2: Fix Time Padding (10 min)
**Action:** Replace edge padding with linear interpolation for time  
**Expected:** No duplicate time values, proper temporal structure  
**Test:** Check `np.unique(t_prefix)` has no duplicates

### Experiment 3: Add Gradient Clipping (5 min)
**Action:** Add `clip_grad_norm_(model.parameters(), 1.0)`  
**Expected:** More stable training, fewer NaN losses  
**Test:** Monitor gradient norms during training

### Experiment 4: Clamp logvar in NLL Loss (5 min)
**Action:** Clamp logvar to [-10, 10] in gaussian_nll_loss  
**Expected:** Loss stays positive, prevents variance collapse  
**Test:** Check loss values stay > 0

### Experiment 5: Overfit Test (20 min)
**Action:** Train on 100 samples for 50 epochs  
**Expected:** Loss should go to near-zero if model can learn  
**Test:** If loss doesn't decrease, architecture or data issue

### Experiment 6: Check Denormalization (5 min)
**Action:** Add asserts to verify params are in physical units  
**Expected:** E0 in [0.05, 1.25], k_d in [1e-5, 5e-3]  
**Test:** Print param ranges after denormalization

### Experiment 7: Visualize Time Features (10 min)
**Action:** Plot normalized time sequences, check if dt is preserved  
**Expected:** Different dt's should be distinguishable  
**Test:** Plot `np.diff(t_prefix_norm)` for different samples

---

## Priority Fixes (Apply in Order)

1. **CRITICAL:** Fix time normalization (B1) - use per-sequence or don't normalize
2. **CRITICAL:** Fix time padding (A5) - use interpolation not edge padding
3. **HIGH:** Clamp logvar in NLL loss (D1)
4. **HIGH:** Add gradient clipping (F2)
5. **MEDIUM:** Add explicit dt feature (B3)
6. **MEDIUM:** Seed train/val split (A1)
7. **LOW:** Add batch normalization (E5)
8. **LOW:** Reduce dropout if underfitting (E4)

---

## Code Snippets for Quick Fixes

### Fix 1: Time Normalization (Per-Sequence)
```python
# In EarlyInferenceDataset.__init__, replace lines 83-84, 96:
if normalize_inputs:
    # ... pH normalization ...
    # Normalize time per-sequence to preserve dt relationships
    t_prefix_norm = []
    for i in range(t_prefix.shape[0]):
        t_seq = t_prefix[i]
        if len(t_seq) > 1:
            t_mean_i = np.mean(t_seq)
            t_std_i = np.std(t_seq) + 1e-8
            t_prefix_norm.append((t_seq - t_mean_i) / t_std_i)
        else:
            t_prefix_norm.append(t_seq)
    self.t_prefix = np.array(t_prefix_norm)
    self.t_mean = 0.0  # Not used for per-sequence norm
    self.t_std = 1.0
```

### Fix 2: Time Padding with Interpolation
```python
# In generate_early_inference_data.py, replace lines 490-497:
if len(pH_seq) < max_len:
    # Interpolate to uniform grid
    t_old = t_seq
    t_new = np.linspace(0, prefix_length, max_len)
    pH_seq = np.interp(t_new, t_old, pH_seq)
    t_seq = t_new
```

### Fix 3: Clamp logvar
```python
# In early_inference_model.py, modify gaussian_nll_loss:
def gaussian_nll_loss(mean: torch.Tensor, logvar: torch.Tensor, 
                     target: torch.Tensor) -> torch.Tensor:
    logvar = torch.clamp(logvar, min=-10.0, max=10.0)  # Add this line
    precision = torch.exp(-logvar)
    return 0.5 * (logvar + precision * (target - mean) ** 2).mean()
```

### Fix 4: Gradient Clipping
```python
# In train_epoch, after loss.backward() and before optimizer.step():
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Add this
optimizer.step()
```

---

**End of Audit**
