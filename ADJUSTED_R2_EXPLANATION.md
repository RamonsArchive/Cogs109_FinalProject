# Adjusted R² Explanation for Polynomial Models

## The Problem: Two Different Adjusted R² Values

In the report `poly_deg_1_20251123_223129_report.txt`, there are **two different Adjusted R² values**:

1. **Line 25 (TEST SET RESULTS)**: Adjusted R² = **-0.3047** (using 78 features) ✅ **CORRECT**
2. **Line 35 (ADDITIONAL METRICS)**: Adjusted R² = **0.1050** ❌ **WRONG** (using 13 predictors)

## Why This Happened

### Root Cause

The `generate_text_report` function in `graph.py` calculates Adjusted R² using:

```python
p = len(predictors)  # This is 13 (original predictors)
adj_r2 = calculate_adjusted_r2(r2, n, p)
```

But for polynomial models:

- **Original predictors**: 13 (surface_type, Avg_Temp, day, week, etc.)
- **Actual features after transformation**: 78 (after polynomial + one-hot encoding)

### The Bug

1. `generate_text_report` calculates Adjusted R² **twice**:
   - Once in "TEST SET RESULTS" section (line 1123-1126)
   - Once in "ADDITIONAL METRICS" section (line 1196-1205)
2. Both calculations use `p = len(predictors) = 13` instead of the actual number of features (78)

3. The code in `polytest.py` was only replacing the **first occurrence** (TEST SET RESULTS), leaving the second one (ADDITIONAL METRICS) with the wrong value.

## Why Adjusted R² is Negative (-0.3047)

Adjusted R² can be **negative** when the model performs worse than simply predicting the mean!

### Formula

```
Adjusted R² = 1 - (1 - R²) × (n - 1) / (n - p - 1)
```

Where:

- **R²** = 0.1579 (15.79% variance explained)
- **n** = number of test samples (~200-300)
- **p** = 78 (actual features after transformation)

### Why It's Negative

1. **Too many features relative to samples**: With 78 features and ~200-300 samples, the model is **overfitting**
2. **Penalty for complexity**: Adjusted R² penalizes models with many features
3. **When p approaches n**: The denominator `(n - p - 1)` becomes small, making the penalty large
4. **Result**: The penalty exceeds the R² benefit, making Adjusted R² negative

### What Negative Adjusted R² Means

- The model with 78 features performs **worse** than a model that just predicts the mean
- The complexity penalty outweighs the explanatory power
- This indicates **severe overfitting**

## Why the Second Value (0.1050) is Wrong

The value 0.1050 is calculated using **p = 13** (original predictors) instead of **p = 78** (actual features):

```python
# WRONG calculation (what generate_text_report does):
adj_r2_wrong = calculate_adjusted_r2(0.1579, n, 13)  # Uses 13 predictors
# Result: 0.1050 (positive, but misleading!)

# CORRECT calculation (what we should do):
adj_r2_correct = calculate_adjusted_r2(0.1579, n, 78)  # Uses 78 features
# Result: -0.3047 (negative, but accurate!)
```

The 0.1050 value is **misleading** because it doesn't account for the actual model complexity (78 features).

## The Fix

The code in `polytest.py` now:

1. Calculates Adjusted R² correctly using `n_features = 78`
2. Replaces **BOTH** occurrences in the report:
   - TEST SET RESULTS section
   - ADDITIONAL METRICS section
3. Also updates the MODEL INTERPRETATION section

### Code Fix

```python
# Calculate adjusted R² with correct number of features
adj_r2 = calculate_adjusted_r2(test_r2, n_samples, n_features)  # Uses 78 features

# Replace ALL occurrences in the report
for i, line in enumerate(report_lines):
    if "Adjusted R²:" in line and "accounts for" not in line:
        report_lines[i] = f"  • Adjusted R²:       {adj_r2:.4f} (using {n_features} features after polynomial transformation)"
```

## Interpretation

### For Polynomial Degree 1:

- **R² = 0.1579**: Model explains 15.79% of variance
- **Adjusted R² = -0.3047**: After accounting for 78 features, the model is worse than predicting the mean
- **Conclusion**: The model is **overfitting** - too many features for the sample size

### Comparison with Backward Selected Log:

- **R² = 0.2720**: Model explains 27.2% of variance
- **Adjusted R² = 0.2654**: After accounting for 2 features, still positive
- **Conclusion**: Much better generalization despite slightly higher RMSE

## Key Takeaway

**The negative Adjusted R² (-0.3047) is CORRECT** - it accurately reflects that the polynomial model with 78 features is overfitting. The positive value (0.1050) was **wrong** because it used the wrong number of features (13 instead of 78).

After the fix, both values in the report will show **-0.3047**, which is the accurate assessment of the model's performance after accounting for its true complexity.
