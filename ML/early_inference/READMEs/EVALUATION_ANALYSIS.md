# Evaluation Results Analysis

## Summary of Issues Found

After analyzing the evaluation results in `evaluation_early_inference_30s/`, I've identified several critical issues explaining why ML is performing worse than mechanistic fitting.

---

## Key Findings

### 1. **ML Predictions Are Unbounded** ❌

**Problem:**
- ML model predicts **negative E0_g_per_L values** in 19/100 cases
- Negative values are physically impossible (enzyme loading can't be negative)
- When E0_g_per_L < 0, the simulator produces flat pH trajectories (no reaction occurs)
- This causes catastrophic forecast failures (negative R² values)

**Evidence:**
- Sample 39: ML predicts E0 = -0.013378 (true = 0.196961) → R² = -226.4
- Sample 22: ML predicts E0 = -0.044976 (true = 0.100300) → R² = -153.2
- Sample 14: ML predicts E0 = -0.001499 (true = 0.203183) → R² = -138.3

**Why Fit Doesn't Have This Issue:**
- Fit uses bounds: `'E0_g_per_L': (5e-4, 1.25)` which prevents negative values
- ML has no bounds checking - model can output any value

### 2. **Prefix Data Comparison**

**ML Uses:**
- Uniform interpolated prefix (100 points, uniform spacing)
- Same format as training data
- Extracted via `extract_prefix()` which interpolates to uniform grid

**Fit Uses:**
- Original grid prefix (all points where t <= 30s)
- For uniform grid with 2000 points over 2000s: ~30 points in 30s prefix
- Extracted via simple masking: `t_full <= prefix_length`

**Fairness:**
- ML actually has **MORE data points** (100 vs ~30)
- But ML points are **interpolated** (less information than original)
- Fit has **fewer but original points** (more information per point)

**Conclusion:** This difference might contribute, but it's not the main issue.

### 3. **Performance Metrics**

**Parameter Estimation (MAE):**
- ML E0_g_per_L MAE: **0.359 ± 0.425** (very poor, high variance)
- Fit E0_g_per_L MAE: **0.0087 ± 0.023** (excellent)
- ML k_d MAE: **0.0027 ± 0.0029** (acceptable)
- Fit k_d MAE: **0.0011 ± 0.0008** (better)

**Trajectory Forecasting (RMSE, R²):**
- ML RMSE: **0.224 ± 0.484** (very poor, high variance)
- Fit RMSE: **0.021 ± 0.017** (excellent, low variance)
- ML R²: **-10.6** (catastrophic - worse than mean baseline!)
- Fit R²: **0.968** (excellent)

**Additional Stats:**
- 77/100 cases: ML RMSE > Fit RMSE
- 19/100 cases: ML R² < 0 (catastrophic failures)
- 19/100 cases: ML predicts negative E0_g_per_L

---

## Root Causes

### Primary Issue: **Unbounded ML Predictions**

The ML model is not constrained to output physically valid parameters:
- No bounds checking on outputs
- No regularization to prevent negative values
- Normalization might allow negative values in output space

**Why Training Doesn't Prevent This:**
- Training data has E0_g_per_L in range [5e-4, 1.25] (all positive)
- But normalization maps this range to different values
- Model can learn to output values outside the training range after denormalization
- No explicit constraint ensures predictions stay in valid range

### Secondary Issues:

1. **Time Grid Difference:**
   - ML: Interpolated uniform (100 points)
   - Fit: Original grid (~30 points for uniform grid)
   - Could contribute but not the main issue

2. **Evaluation Methodology:**
   - Both methods use same prefix length (30s)
   - Both forecast full trajectory (0-2000s)
   - Metrics computed on same reference grid
   - Methodology is fair

---

## Recommendations

### Immediate Fixes:

1. **Add Bounds Checking to ML Predictions:**
   ```python
   # Clamp ML predictions to valid range
   ml_params['E0_g_per_L'] = np.clip(ml_params['E0_g_per_L'], 5e-4, 1.25)
   ml_params['k_d'] = np.clip(ml_params['k_d'], 0.0, 5e-3)
   ```

2. **Improve Model Training:**
   - Add output constraints/loss penalties for out-of-bounds predictions
   - Consider using bounded activation functions (e.g., sigmoid for E0)
   - Add regularization to prevent extreme predictions

3. **Consider Using Same Prefix for Both:**
   - For fairer comparison, use same prefix data format
   - Or document that fit uses original grid intentionally (more realistic)

### Long-Term Improvements:

1. **Better Normalization:**
   - Ensure output normalization handles edge cases
   - Consider using bounded transforms (e.g., logit for bounded ranges)

2. **Model Architecture:**
   - Add output layers that enforce bounds (e.g., sigmoid + scaling)
   - Consider uncertainty quantification to identify unreliable predictions

3. **Training Data:**
   - Ensure full coverage of parameter space
   - Consider data augmentation near boundaries

---

## Questions to Investigate

1. **Why is ML predicting negative values?**
   - Is normalization working correctly?
   - Are there edge cases in training data?
   - Is the model overfitting or underfitting?

2. **Is the prefix data difference significant?**
   - Test: Use same prefix format for both ML and Fit
   - Compare performance with identical input data

3. **Can we fix this without retraining?**
   - Yes: Add bounds clamping (quick fix)
   - Better: Retrain with bounds constraints

---

## Conclusion

**Main Issue:** ML model predicts unbounded (negative) E0_g_per_L values, causing catastrophic forecast failures. The mechanistic fit is constrained by bounds, preventing this issue.

**Secondary Issue:** ML uses interpolated uniform prefix (100 points) while Fit uses original grid prefix (~30 points). This difference is minor compared to the unbounded predictions issue.

**Next Steps:**
1. Add bounds clamping to ML predictions (immediate fix)
2. Retrain model with bounds constraints (long-term fix)
3. Consider making prefix data format consistent for fairer comparison
