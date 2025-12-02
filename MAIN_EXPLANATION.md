# Explanation of main.py Structure

## Overview

The `main.py` file trains and compares multiple models across three different approaches:

1. **Poisson Regression** (for count data)
2. **Logistic Regression** (for binary classification)
3. **Linear Regression** (for log-transformed data)

## Data Flow and Structure

### STEP 1: Data Splitting (Lines 144-156)

```
Full Dataset (100%)
└─> ONE SPLIT: Train (80%) | Test (20%)  [FIXED - NO DATA LEAKAGE]
    │
    ├─> train_df: Used for ALL model training and hyperparameter tuning
    └─> test_df: ONLY used for final unbiased evaluation (never touched during training)
```

**Key Points:**

- Split happens ONCE at the beginning
- All models use the same train/test split
- Test set is NEVER used for:
  - Feature selection
  - Hyperparameter tuning
  - Model selection
  - Cross-validation

### STEP 2: Data Transformations (Lines 158-178)

- **log_train_df / log_test_df**: Log-transformed versions (for linear regression)
- **binary_train_df / binary_test_df**: Binary target (1 if num_injuries > 4, else 0) for logistic regression
- All transformations use the SAME train/test split

### STEP 3: Backward Feature Selection (Lines 186-229)

**What Happens:**

1. Backward selection runs ONCE on **logistic regression** using Model 6 (kitchen sink) predictors
2. Uses **training data only** (binary_train_df) - NO DATA LEAKAGE
3. Finds significant features (p-value < 0.1)
4. Selected features become **Model 7** (backward_selected)

**Comparison:**

- **Model 6**: All 19 predictors from kitchen sink
- **Model 7**: Selected predictors from backward elimination

**IMPORTANT:** The comparison is now done for ALL THREE model types:

- Logistic Regression: Model 6 vs Model 7
- Poisson Regression: Model 6 vs Model 7
- Linear Regression: Model 6 vs Model 7

### STEP 4: Model Training (Lines 246-316)

**For EACH model type (Poisson, Logistic, Linear):**

- Trains Models 1-6 (predefined predictor sets)
- Trains Model 7 (backward selected features)
- All models use the SAME train/test split
- All models evaluated with 10-fold CV on training data

**Model Sets:**

1. **baseline**: Basic features (surface, day, week, season, stadium, surface, dome)
2. **Avg_Temp**: Baseline + temperature
3. **game_intensity**: Avg_Temp + game metrics (plays, yards, turnovers)
4. **rest_travel**: Avg_Temp + rest/travel factors
5. **weather**: Avg_Temp + weather conditions
6. **kitchen_sink_all**: ALL features (19 predictors)
7. **backward_selected**: Features selected by backward elimination

### STEP 5: Best Model Selection (Lines 318-353)

**For each model type:**

- Selects best model based on CV performance:
  - **Poisson**: Lowest Poisson Deviance
  - **Logistic**: Highest ROC-AUC (stored as negative for argmin)
  - **Linear**: Lowest MSE
- Retrains best model on FULL training set
- Evaluates on held-out test set (unbiased estimate)

## Key Questions Answered

### Q1: Does `compare_models_with_without_selection` compare for all models?

**A:** YES - Now it does! The code now calls this function for:

- Logistic Regression (Model 6 vs Model 7)
- Poisson Regression (Model 6 vs Model 7)
- Linear Regression (Model 6 vs Model 7)

### Q2: Where is the comparison for Poisson and Linear?

**A:** Lines 218-242 in main.py - Added comparison calls for all three model types.

### Q3: Is there a naive baseline?

**A:** YES - Now added! For logistic regression:

- **Naive Baseline**: Always predicts the majority class
- Calculated for: Accuracy, F1 Score, ROC-AUC
- Shows improvement percentage over baseline
- Included in reports

### Q4: Is there data leakage?

**A:** NO - The code is structured correctly:

- Data split ONCE at the beginning (line 152)
- All feature selection uses training data only
- All model training uses training data only
- Test set only used for final evaluation

## Model Comparison Logic

**What gets compared:**

- **Model 6 (kitchen_sink_all)**: All 19 predictors
- **Model 7 (backward_selected)**: Features selected by backward elimination

**Comparison happens for:**

1. ✅ Logistic Regression (binary classification)
2. ✅ Poisson Regression (count data)
3. ✅ Linear Regression (log-transformed data)

**How comparison works:**

- Uses 5-fold cross-validation on training data
- Compares CV scores between Model 6 and Model 7
- Shows whether feature selection improved performance

## Naive Baseline Explanation

**For Logistic Regression:**

- **Strategy**: Always predict the most common class (majority class)
- **Example**: If 60% of games have num_injuries ≤ 4, always predict 0
- **Purpose**: Shows minimum performance threshold
- **Interpretation**:
  - If model accuracy < naive baseline → Model is WORSE than random guessing
  - If model accuracy > naive baseline → Model learned something useful
  - ROC-AUC > 0.5 → Model is better than random

## Summary

**Data Flow:**

1. Split data once (80/20)
2. Transform data (log, binary) from split
3. Feature selection on training data only
4. Compare Model 6 vs Model 7 for all three model types
5. Train all models (1-7) for each model type
6. Select best model for each type
7. Final evaluation on held-out test set

**No Data Leakage:**

- ✅ Single split at beginning
- ✅ Feature selection uses training data only
- ✅ All comparisons use training data only
- ✅ Test set only for final evaluation

**Comparisons:**

- ✅ Model 6 vs Model 7 for Logistic Regression
- ✅ Model 6 vs Model 7 for Poisson Regression
- ✅ Model 6 vs Model 7 for Linear Regression

**Baselines:**

- ✅ Naive baseline for Logistic Regression (majority class predictor)
