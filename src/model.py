import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_poisson_deviance,
)
from sklearn.linear_model import PoissonRegressor

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def make_pipeline(num_cols, cat_cols):
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    clf = PoissonRegressor(max_iter=5000)
    return Pipeline([("preprocessor", pre), ("model", clf)])


def train_model(df: pd.DataFrame, predictors: list, results: dict, model_name: str):
    """Train a single model with 10-fold CV"""
    outcomes = ["num_injuries"]
    target = outcomes[0]

    num_cols_all = [c for c in predictors if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols_all = [c for c in predictors if not pd.api.types.is_numeric_dtype(df[c])]

    print(f"\n{'='*70}")
    print(f"Training {model_name}")
    print(f"{'='*70}")
    print("Data shape:", df.shape)
    print("Number of numeric columns:", len(num_cols_all))
    print("Number of categorical columns:", len(cat_cols_all))
    print("Predictors:", predictors)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
    X_train = train_df[predictors]
    y_train = train_df[target].astype(int)
    X_test = test_df[predictors]
    y_test = test_df[target].astype(int)

    print("Train size:", train_df.shape, "| Test size:", test_df.shape)

    # Cross-validation
    pipeline = make_pipeline(num_cols_all, cat_cols_all)
    cv = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    scoring = {
        "neg_poisson_dev": "neg_mean_poisson_deviance",
        "neg_mse": "neg_mean_squared_error",
        "neg_mae": "neg_mean_absolute_error",
    }

    cv_out = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)
    mean_poisson_dev = -np.mean(cv_out["test_neg_poisson_dev"])
    mean_mse = -np.mean(cv_out["test_neg_mse"])
    mean_mae = -np.mean(cv_out["test_neg_mae"])

    print(f"[CV] Mean Poisson deviance: {mean_poisson_dev:.4f}")
    print(f"[CV] RMSE: {np.sqrt(mean_mse):.4f}   (MSE: {mean_mse:.4f})")
    print(f"[CV] MAE : {mean_mae:.4f}")

    # Test set
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    test_mse = mean_squared_error(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_poisson_dev = mean_poisson_deviance(y_test, np.clip(y_pred, 1e-9, None))

    print(f"[Test] Poisson deviance: {test_poisson_dev:.4f}")
    print(f"[Test] RMSE: {np.sqrt(test_mse):.4f}   (MSE: {test_mse:.4f})")
    print(f"[Test] MAE : {test_mae:.4f}")

    # Store results
    results[model_name] = {
        "10foldCV": {
            "val_poisson_dev": mean_poisson_dev,
            "val_mse": mean_mse,
            "val_mae": mean_mae,
            "test_poisson_dev": test_poisson_dev,
            "test_mse": test_mse,
            "test_mae": test_mae,
        },
        "y_test": y_test,
        "y_pred": y_pred,
        "df": df,
        "predictors": predictors,
    }

    return results[model_name]


def use_best_model(results: dict, df: pd.DataFrame, err_k10: list, predictors: list):
    """
    Select best model based on CV error and retrain on full 80/20 split (NO validation)
    """
    best_model_idx = np.argmin(err_k10)
    best_predictors = predictors[best_model_idx]

    # Get the original CV results from the best model
    best_model_name = list(results.keys())[best_model_idx]
    original_cv_results = results[best_model_name]["10foldCV"]

    print(f"\n{'='*70}")
    print(f"🎯 BEST MODEL SELECTED: Model {best_model_idx + 1}")
    print(f"{'='*70}")
    print(f"Best CV Poisson Deviance: {original_cv_results['val_poisson_dev']:.4f}")
    print(f"Best Predictors ({len(best_predictors)}):")
    for i, pred in enumerate(best_predictors, 1):
        print(f"  {i}. {pred}")
    print(f"\n{'='*70}")
    print("Retraining on FULL 80/20 split (no validation)...")
    print(f"{'='*70}\n")

    # Retrain on full 80/20 split
    outcomes = ["num_injuries"]
    target = outcomes[0]

    num_cols = [c for c in best_predictors if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in best_predictors if not pd.api.types.is_numeric_dtype(df[c])]

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
    X_train = train_df[best_predictors]
    y_train = train_df[target].astype(int)
    X_test = test_df[best_predictors]
    y_test = test_df[target].astype(int)

    print("Final Train size:", train_df.shape, "| Test size:", test_df.shape)

    # Train final model (no CV, just straight training)
    pipeline = make_pipeline(num_cols, cat_cols)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Calculate final test metrics
    test_mse = mean_squared_error(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_poisson_dev = mean_poisson_deviance(y_test, np.clip(y_pred, 1e-9, None))

    print(f"\n[FINAL TEST] Poisson deviance: {test_poisson_dev:.4f}")
    print(f"[FINAL TEST] RMSE: {np.sqrt(test_mse):.4f}   (MSE: {test_mse:.4f})")
    print(f"[FINAL TEST] MAE : {test_mae:.4f}")

    # Package results using ORIGINAL CV metrics for comparison
    best_model_results = {
        "10foldCV": {
            "val_poisson_dev": original_cv_results[
                "val_poisson_dev"
            ],  # Original CV error
            "val_mse": original_cv_results["val_mse"],  # Original CV MSE
            "val_mae": original_cv_results["val_mae"],  # Original CV MAE
            "test_poisson_dev": test_poisson_dev,  # New test error
            "test_mse": test_mse,  # New test MSE
            "test_mae": test_mae,  # New test MAE
        },
        "y_test": y_test,
        "y_pred": y_pred,
        "df": df,
        "predictors": best_predictors,
        "pipeline": pipeline,
        "best_model_idx": best_model_idx,
    }

    return best_model_results
