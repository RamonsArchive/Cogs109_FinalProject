# Backward Selection Feature Engineering Guide

## Overview

This document explains the **backward selection** implementation for finding the optimal predictor set for NFL injury prediction models.

---

## What is Backward Selection?

**Backward Elimination** is a feature selection technique that:

1. Starts with ALL predictors in the model
2. Removes the LEAST significant predictor (highest p-value)
3. Re-fits the model without that predictor
4. Repeats until all remaining predictors are significant (p < α)

### Why Use It?

- **Reduces overfitting**: Eliminates noise predictors that don't truly predict injuries
- **Improves interpretability**: Simpler models are easier to explain
- **Better generalization**: Models with fewer, stronger predictors often perform better on new data
- **Statistical rigor**: Only keeps predictors with proven significance

---

## Your Implementation

### Key Design Decisions

#### 1. **Run Backward Selection ONCE**

- You perform backward selection on **Logistic Regression** (binary classification)
- The selected features are then used for **ALL THREE model types**:
  - Poisson Regression (count data)
  - Logistic Regression (binary classification)
  - Linear Regression (log-transformed target)

**Why?** Because you're predicting the SAME phenomenon (injuries) with different model assumptions. The underlying relationships between predictors and injuries should be consistent.

#### 2. **Significance Level: α = 0.1**

- More lenient than traditional α = 0.05
- Good for exploratory analysis
- Reduces risk of removing marginally significant predictors

**Interpretation**: Keep predictors where we're at least 90% confident they have a real relationship with injuries.

#### 3. **Starting Point: Model 6 (Kitchen Sink)**

- Starts with all 18 predictors:
  ```
  Surface: surface_type, surface, stadium, dome
  Weather: Avg_Temp, Avg_Wind_MPH, Avg_Humidity_Percent, Avg_Percipitation_Prob_Percent
  Temporal: day, week, season
  Game Intensity: num_plays, yds_w, yds_l, tov_w, tov_l
  Rest/Travel: HOME_day_since_last_game, AWAY_day_since_last_game, distance_miles
  ```

---

## Data Preprocessing (Already Handled!)

### ✅ Your data is ALREADY normalized!

In `model.py`, line 35:

```python
("num", StandardScaler(), num_cols),  # Standardizes automatically
```

**What `StandardScaler()` does:**

- Transforms each numeric feature to have mean = 0, standard deviation = 1
- Formula: `z = (x - μ) / σ`
- **Why it matters**: Ensures all features are on the same scale, making coefficients comparable

### ✅ Categorical Variables are One-Hot Encoded

In `model.py`, line 36:

```python
("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
```

**What this does:**

- Converts categorical variables (e.g., "surface_type") into binary dummy variables
- Example: `surface_type = "Turf"` → `surface_type_Turf = 1, surface_type_Grass = 0`

---

## How Backward Selection Works in Your Code

### Step-by-Step Process:

1. **Preprocessing**

   ```python
   # Scale numeric features, one-hot encode categorical
   preprocessor = ColumnTransformer([
       ("num", StandardScaler(), numeric_cols),
       ("cat", OneHotEncoder(drop="first"), categorical_cols),
   ])
   X_processed = preprocessor.fit_transform(X)
   ```

2. **Fit Initial Model with ALL Features**

   ```python
   model = sm.Logit(y, X_with_const).fit()
   p_values = model.pvalues
   ```

3. **Find Least Significant Feature**

   ```python
   max_p_value = p_values.max()
   max_p_feature = p_values.idxmax()
   ```

4. **Eliminate if p > 0.1**

   ```python
   if max_p_value > 0.1:
       X_with_const = X_with_const.drop(columns=[max_p_feature])
   ```

5. **Repeat Until All Features Significant**

6. **Map Back to Original Feature Names**
   - One-hot encoded features (e.g., `surface_type_Turf`) → `surface_type`
   - Numeric features stay the same

---

## Output: Model 7

After backward selection, your code creates **Model 7**:

- Name: `"backward_selected"` (and `_log`, `_binary` variants)
- Contains only statistically significant predictors
- Tested across all three model types for fair comparison

### Comparison Metrics:

The code also compares Model 6 (all features) vs Model 7 (selected features) using:

- **AIC (Akaike Information Criterion)**: Lower is better, balances fit and complexity
- **BIC (Bayesian Information Criterion)**: Lower is better, penalizes complexity more than AIC
- **Log-Likelihood**: Higher is better, measures model fit

**Interpretation:**

- If Model 7 has **lower AIC/BIC**: Feature selection improved the model! ✅
- If Model 7 has **higher AIC/BIC**: All features were useful, no overfitting issue ⚠️

---

## Should You Normalize Your Data Further?

### ❌ NO - Your data is already properly normalized!

**What you DON'T need:**

- ✗ Manual standardization (already done by `StandardScaler`)
- ✗ Min-Max scaling (StandardScaler is better for regression)
- ✗ Log-transform numeric features (you already have a log-transformed target variant)

**What's ALREADY handled:**

- ✅ Numeric features: Standardized (mean=0, std=1)
- ✅ Categorical features: One-hot encoded
- ✅ Target variable: Three variants tested (count, log, binary)

---

## Running the Code

### What Happens When You Run `main.py`:

1. **Data Loading & Cleaning**

   ```
   ✅ Loaded data
   ✅ Cleaned data
   ✅ Created log-transformed target
   ✅ Created binary target (num_injuries > 4)
   ```

2. **Exploratory Data Analysis**

   ```
   📊 Correlation analysis
   📊 Distribution plots
   ```

3. **Backward Selection** ⬅️ NEW!

   ```
   🔍 Starting with 18 predictors
   🔍 Eliminating features with p > 0.1
   ✅ Final selection: X features
   📊 Comparing Model 6 vs Model 7 (AIC/BIC)
   ```

4. **Model Training**

   ```
   Models 1-6: Your original predictor sets
   Model 7: Backward-selected features ⬅️ NEW!
   ```

   - Trained on Poisson, Logistic, and Linear models
   - 10-fold cross-validation
   - Test set evaluation

5. **Best Model Selection**
   ```
   ✅ Best Poisson model
   ✅ Best Logistic model
   ✅ Best Linear model
   ```

---

## Expected Outcomes

### Scenario 1: Backward Selection Eliminates Features

```
🔍 Eliminated: dome, tov_l, Avg_Percipitation_Prob_Percent
✅ Selected: 15 features
📊 Model 7 has LOWER AIC/BIC → Improved model!
```

**Interpretation**: Some features were noise, Model 7 should generalize better

### Scenario 2: Backward Selection Keeps Most/All Features

```
✅ Selected: 17-18 features (only 0-1 eliminated)
📊 Model 7 has SIMILAR AIC/BIC → No major improvement
```

**Interpretation**: Your Model 6 was already well-designed, no overfitting issues

### Scenario 3: All Features Eliminated (Rare)

```
⚠️  All features eliminated! Keeping all original features.
```

**Interpretation**: Model convergence issue or extreme multicollinearity

---

## FAQ

### Q: Why use Logistic Regression for backward selection?

**A:** Because binary classification (high injury risk yes/no) is your MAIN safety prediction goal. The features that predict "high risk" should also predict injury counts.

### Q: Can I use different α for different model types?

**A:** Yes, but NOT recommended. Using the same feature set ensures fair comparison across models.

### Q: What if backward selection eliminates important features like "surface_type"?

**A:** If it's eliminated with p > 0.1, it means the feature has NO statistically significant relationship with injuries in your dataset. Trust the statistics!

### Q: Should I use forward selection or stepwise selection instead?

**A:**

- **Forward**: Starts with nothing, adds features one-by-one
- **Backward**: Starts with everything, removes features (what we use)
- **Stepwise**: Combination of both

Backward is preferred when you have a reasonable number of predictors (<20) and want to avoid missing interactions.

### Q: What if Model 7 performs WORSE than Model 6?

**A:** This can happen if:

1. Your test set is small (randomness)
2. Eliminated features had small but real effects
3. Non-linear relationships exist

If Model 7 is worse, stick with Model 6!

---

## Summary

✅ **Your current Model 6 predictor set is excellent** - you're using all valuable features
✅ **Data normalization is already handled** by `StandardScaler` and `OneHotEncoder`
✅ **Backward selection runs ONCE** on logistic regression, then applied to all models
✅ **Model 7 will be created** with statistically significant features (α = 0.1)
✅ **AIC/BIC comparison** will tell you if feature selection helped

**Bottom Line**: You're doing feature selection the RIGHT way! 🎯
