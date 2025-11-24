# Polynomial Model Analysis and Bug Report

## Summary of Findings

### 1. **Polynomial Degree 1 Should Be Equivalent to Regular OLS**

You are **correct** - polynomial degree 1 with `PolynomialFeatures(degree=1, include_bias=False)` should be mathematically equivalent to regular OLS (non-logged) on the same predictors.

**Why?**

- `PolynomialFeatures(degree=1, include_bias=False)` on numeric predictors just returns the original numeric predictors unchanged (no polynomial terms created)
- So polynomial degree 1 = regular OLS on original scale

### 2. **Current Results Comparison**

| Model                          | Predictors                 | CV RMSE | Test RMSE | Notes                           |
| ------------------------------ | -------------------------- | ------- | --------- | ------------------------------- |
| **Polynomial Degree 1**        | Model 3 (13 predictors)    | 1.9651  | 1.9411    | Non-logged, original scale      |
| **Backward Selected Log**      | 2 predictors (week, yds_l) | 1.9757  | 1.9813    | Log-transformed, converted back |
| **Kitchen Sink All (Poisson)** | Model 6 (19 predictors)    | 1.9986  | 2.0314    | Poisson regression              |

**Observation**: Polynomial degree 1 has the **lowest RMSE** among classical linear models, which is correct!

### 3. **Critical Bugs Found in `polytest.py`**

#### Bug #1: **Data Leakage - One-Hot Encoding Before Split**

```python
# CURRENT (WRONG):
X_cat = pd.get_dummies(df[cat_cols], drop_first=True)  # Applied to FULL dataset
X = np.hstack([X_poly, X_cat.values])
X_train, X_test, y_train, y_test = train_test_split(X, y, ...)  # Split AFTER encoding
```

**Problem**: `pd.get_dummies` is applied to the entire dataset before splitting. This causes:

- Information leakage (test set categories seen during training)
- Inconsistent encoding between train/test
- Overly optimistic performance estimates

**Fix**: Use `OneHotEncoder` in a pipeline (like `main.py` does) so encoding happens AFTER splitting.

#### Bug #2: **Data Leakage - PolynomialFeatures Before Split**

```python
# CURRENT (WRONG):
X_poly = poly.fit_transform(X_num)  # Applied to FULL dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, ...)  # Split AFTER transformation
```

**Problem**: `PolynomialFeatures` is fit on the entire dataset before splitting. This causes:

- Information leakage (test set statistics used during training)
- Overly optimistic performance estimates

**Fix**: Apply `PolynomialFeatures` inside a pipeline so it's fit only on training data.

#### Bug #3: **Missing StandardScaler**

```python
# CURRENT (WRONG):
# No standardization of numeric predictors
X_num = df[num_cols]
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_poly = poly.fit_transform(X_num)  # Raw numeric values
```

**Problem**: `main.py` uses `StandardScaler` for numeric predictors, but `polytest.py` does not. This causes:

- Different coefficient scales
- Different model performance
- Inconsistent comparison with `main.py` models

**Fix**: Add `StandardScaler` to the pipeline (like `main.py` does).

### 4. **Why Polynomial Degree 1 ≠ Non-Logged OLS from main.py**

Even after fixing the bugs, polynomial degree 1 might still differ from a non-logged OLS in `main.py` because:

1. **Different Predictor Sets**:

   - Polynomial: Uses Model 3 (13 predictors)
   - `main.py` linear models: Use log-transformed target (different approach)
   - No direct non-logged OLS in `main.py` to compare with

2. **Different Preprocessing** (before bug fixes):

   - Polynomial: No standardization, `pd.get_dummies` before split
   - `main.py`: StandardScaler, OneHotEncoder in pipeline

3. **Random State Differences**:
   - Polynomial: `random_state=42` for train_test_split
   - `main.py`: `RANDOM_STATE=42` for train_test_split
   - Should be the same, but verify

### 5. **Expected Behavior After Fixes**

After fixing the bugs:

- **Polynomial Degree 1** should be equivalent to a non-logged OLS with:
  - Same predictors (Model 3: 13 predictors)
  - Same preprocessing (StandardScaler + OneHotEncoder in pipeline)
  - Same random state (42)
  - Same train/test split (80/20)

The RMSE should be **very similar** (within numerical precision differences).

### 6. **Why Polynomial Degree 1 Has Lower RMSE Than Logged OLS**

This is actually **expected** and **correct**:

1. **More Predictors**: Polynomial uses 13 predictors (Model 3), while backward_selected_log uses only 2 predictors
2. **No Transformation Loss**: Polynomial works directly on original scale, while logged OLS has transformation error when converting back
3. **Model Complexity**: More predictors can capture more variance (though risk of overfitting)

However, the **adjusted R²** tells a different story:

- Polynomial Degree 1: Adjusted R² = -0.3143 (using 79 features) → **Overfitting!**
- Backward Selected Log: Adjusted R² = 0.2654 (using 2 predictors) → **Better generalization**

### 7. **Recommendations**

1. **Fix the bugs** in `polytest.py` (data leakage + missing StandardScaler)
2. **Compare apples to apples**:
   - Create a non-logged OLS model in `main.py` with Model 3 predictors
   - Compare it to polynomial degree 1 (after fixes)
3. **Use Adjusted R²** for model selection, not just RMSE
4. **Consider**: The backward_selected_log model (2 predictors, R²=27.2%) might be better for generalization despite slightly higher RMSE

### 8. **Conclusion**

- ✅ Polynomial degree 1 **should** be equivalent to non-logged OLS (mathematically)
- ❌ Current implementation has **data leakage bugs** that make it not equivalent
- ✅ Polynomial degree 1 has **lowest RMSE** among classical linear models (correct)
- ⚠️ But **adjusted R²** suggests it's overfitting compared to backward_selected_log
- 🔧 **Fix bugs** to ensure fair comparison
