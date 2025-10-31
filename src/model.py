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
from sklearn.linear_model import PoissonRegressor, LinearRegression, Ridge

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def make_pipeline(num_cols, cat_cols, model_type="poisson"):
    """Create pipeline with choice of model"""
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),  # Standardizes automatically
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    if model_type == "poisson":
        clf = PoissonRegressor(max_iter=5000)
    elif model_type == "linear":
        clf = LinearRegression()  # OLS
    elif model_type == "ridge":
        clf = Ridge(alpha=1.0)  # Regularized linear regression
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return Pipeline([("preprocessor", pre), ("model", clf)])


def train_model(
    df: pd.DataFrame,
    predictors: list,
    results: dict,
    model_name: str,
    model_type="poisson",
    is_log_target=False,
):
    """Train a single model with 10-fold CV

    Args:
        model_type: 'poisson', 'linear', or 'ridge'
        is_log_target: True if target is log-transformed
    """
    outcomes = ["num_injuries"]
    target = outcomes[0]

    num_cols_all = [c for c in predictors if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols_all = [c for c in predictors if not pd.api.types.is_numeric_dtype(df[c])]

    print(f"\n{'='*70}")
    print(
        f"Training {model_name} ({model_type.upper()} on {'LOG-TRANSFORMED' if is_log_target else 'COUNT'} data)"
    )
    print(f"{'='*70}")
    print("Data shape:", df.shape)
    print("Number of numeric columns:", len(num_cols_all))
    print("Number of categorical columns:", len(cat_cols_all))

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
    X_train = train_df[predictors]
    y_train = train_df[target]  # Keep as float if log-transformed
    X_test = test_df[predictors]
    y_test = test_df[target]

    if not is_log_target:
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

    print("Train size:", train_df.shape, "| Test size:", test_df.shape)

    # Cross-validation
    pipeline = make_pipeline(num_cols_all, cat_cols_all, model_type=model_type)
    cv = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)

    if model_type == "poisson":
        scoring = {
            "neg_poisson_dev": "neg_mean_poisson_deviance",
            "neg_mse": "neg_mean_squared_error",
            "neg_mae": "neg_mean_absolute_error",
        }
    else:  # linear/ridge
        scoring = {
            "neg_mse": "neg_mean_squared_error",
            "neg_mae": "neg_mean_absolute_error",
            "r2": "r2",
        }

    cv_out = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)

    mean_mse = -np.mean(cv_out["test_neg_mse"])
    mean_mae = -np.mean(cv_out["test_neg_mae"])

    if model_type == "poisson":
        mean_poisson_dev = -np.mean(cv_out["test_neg_poisson_dev"])
        print(f"[CV] Mean Poisson deviance: {mean_poisson_dev:.4f}")
        primary_metric = mean_poisson_dev
    else:
        mean_r2 = np.mean(cv_out["test_r2"])
        print(f"[CV] R²: {mean_r2:.4f}")
        primary_metric = mean_mse  # Use MSE for linear model selection

    print(f"[CV] RMSE: {np.sqrt(mean_mse):.4f}   (MSE: {mean_mse:.4f})")
    print(f"[CV] MAE : {mean_mae:.4f}")

    # Test set
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    test_mse = mean_squared_error(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)

    if model_type == "poisson":
        test_poisson_dev = mean_poisson_deviance(y_test, np.clip(y_pred, 1e-9, None))
        print(f"[Test] Poisson deviance: {test_poisson_dev:.4f}")
    else:
        from sklearn.metrics import r2_score

        test_r2 = r2_score(y_test, y_pred)
        print(f"[Test] R²: {test_r2:.4f}")

    print(f"[Test] RMSE: {np.sqrt(test_mse):.4f}   (MSE: {test_mse:.4f})")
    print(f"[Test] MAE : {test_mae:.4f}")

    # If log-transformed, show performance in original scale
    if is_log_target:
        y_test_original = np.expm1(y_test)  # Inverse of log1p
        y_pred_original = np.expm1(y_pred)

        # Clip predictions to non-negative
        y_pred_original = np.maximum(y_pred_original, 0)

        original_rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
        original_mae = mean_absolute_error(y_test_original, y_pred_original)

        print(f"\n[Test - Original Scale]")
        print(f"RMSE: {original_rmse:.4f}")
        print(f"MAE : {original_mae:.4f}")

    # Store results
    results[model_name] = {
        "10foldCV": {
            "val_primary_metric": primary_metric,  # MSE for linear, Poisson dev for Poisson
            "val_mse": mean_mse,
            "val_mae": mean_mae,
            "test_mse": test_mse,
            "test_mae": test_mae,
        },
        "y_test": y_test,
        "y_pred": y_pred,
        "df": df,
        "predictors": predictors,
        "model_type": model_type,
        "is_log_target": is_log_target,
    }

    return results[model_name]


def use_best_model(
    results: dict,
    df: pd.DataFrame,
    err_k10: list,
    predictors: list,
    model_type="poisson",
    is_log_target=False,
):
    """Select best model based on CV error and retrain on full 80/20 split"""
    best_model_idx = np.argmin(err_k10)
    best_predictors = predictors[best_model_idx]

    best_model_name = list(results.keys())[best_model_idx]
    original_cv_results = results[best_model_name]["10foldCV"]

    print(f"\n{'='*70}")
    print(f"🎯 BEST MODEL SELECTED: Model {best_model_idx + 1} ({model_type.upper()})")
    print(f"{'='*70}")
    print(f"Best CV Primary Metric: {original_cv_results['val_primary_metric']:.4f}")
    print(f"Best Predictors ({len(best_predictors)}):")
    for i, pred in enumerate(best_predictors, 1):
        print(f"  {i}. {pred}")

    outcomes = ["num_injuries"]
    target = outcomes[0]

    num_cols = [c for c in best_predictors if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in best_predictors if not pd.api.types.is_numeric_dtype(df[c])]

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
    X_train = train_df[best_predictors]
    y_train = train_df[target]
    X_test = test_df[best_predictors]
    y_test = test_df[target]

    if not is_log_target:
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

    pipeline = make_pipeline(num_cols, cat_cols, model_type=model_type)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    test_mse = mean_squared_error(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)

    print(f"\n[FINAL TEST] RMSE: {np.sqrt(test_mse):.4f}   (MSE: {test_mse:.4f})")
    print(f"[FINAL TEST] MAE : {test_mae:.4f}")

    best_model_results = {
        "10foldCV": original_cv_results,
        "y_test": y_test,
        "y_pred": y_pred,
        "df": df,
        "predictors": best_predictors,
        "pipeline": pipeline,
        "best_model_idx": best_model_idx,
        "model_type": model_type,
        "is_log_target": is_log_target,
    }

    return best_model_results
