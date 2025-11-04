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
from sklearn.linear_model import (
    PoissonRegressor,
    LinearRegression,
    Ridge,
    LogisticRegression,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
)

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
    elif model_type == "logistic":
        clf = LogisticRegression(
            max_iter=5000,
            random_state=RANDOM_STATE,
            class_weight="balanced",  # Automatically weights classes inversely to frequency
        )
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
    is_binary=False,
):
    """Train a single model with 10-fold CV

    Args:
        model_type: 'poisson', 'linear', 'ridge', or 'logistic'
        is_log_target: True if target is log-transformed
        is_binary: True if target is binary (for logistic regression)
    """
    outcomes = ["num_injuries"]
    target = outcomes[0]

    num_cols_all = [c for c in predictors if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols_all = [c for c in predictors if not pd.api.types.is_numeric_dtype(df[c])]

    print(f"\n{'='*70}")
    data_type = (
        "BINARY" if is_binary else ("LOG-TRANSFORMED" if is_log_target else "COUNT")
    )
    print(f"Training {model_name} ({model_type.upper()} on {data_type} data)")
    print(f"{'='*70}")
    print("Data shape:", df.shape)
    print("Number of numeric columns:", len(num_cols_all))
    print("Number of categorical columns:", len(cat_cols_all))

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
    X_train = train_df[predictors]
    y_train = train_df[target]  # Keep as float if log-transformed
    X_test = test_df[predictors]
    y_test = test_df[target]

    if is_binary:
        # Already binary, ensure int type
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
        print(
            f"Binary class distribution - Train: {np.bincount(y_train)} | Test: {np.bincount(y_test)}"
        )
    elif not is_log_target:
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
    elif model_type == "logistic":
        scoring = {
            "accuracy": "accuracy",
            "precision": "precision",
            "recall": "recall",
            "f1": "f1",
            "roc_auc": "roc_auc",
            "neg_log_loss": "neg_log_loss",
        }
    else:  # linear/ridge
        scoring = {
            "neg_mse": "neg_mean_squared_error",
            "neg_mae": "neg_mean_absolute_error",
            "r2": "r2",
        }

    cv_out = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)

    if model_type == "logistic":
        # Classification metrics
        mean_accuracy = np.mean(cv_out["test_accuracy"])
        mean_precision = np.mean(cv_out["test_precision"])
        mean_recall = np.mean(cv_out["test_recall"])
        mean_f1 = np.mean(cv_out["test_f1"])
        mean_roc_auc = np.mean(cv_out["test_roc_auc"])
        mean_log_loss = -np.mean(cv_out["test_neg_log_loss"])

        print(f"[CV] Accuracy:  {mean_accuracy:.4f}")
        print(f"[CV] Precision: {mean_precision:.4f}")
        print(f"[CV] Recall:    {mean_recall:.4f}")
        print(f"[CV] F1 Score:  {mean_f1:.4f}")
        print(f"[CV] ROC-AUC:   {mean_roc_auc:.4f}")
        print(f"[CV] Log Loss:  {mean_log_loss:.4f}")
        primary_metric = -mean_roc_auc  # Negative so argmin works (we want max AUC)
    else:
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

    # Get probability predictions for logistic regression (for ROC curve)
    y_pred_proba = None
    if model_type == "logistic":
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]  # Probability of class 1
        # Use lower threshold to catch more high-risk games (better for safety)
        classification_threshold = 0.4  # Adjust this value (try 0.2, 0.3, 0.4, 0.5)
        y_pred = (y_pred_proba >= classification_threshold).astype(int)
        print(f"   Using classification threshold: {classification_threshold}")
    else:
        y_pred = pipeline.predict(X_test)

    if model_type == "logistic":
        # Classification metrics on test set
        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred)
        test_recall = recall_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred)
        test_roc_auc = roc_auc_score(y_test, y_pred_proba)
        test_log_loss_val = log_loss(y_test, y_pred_proba)

        print(f"\n[Test] Accuracy:  {test_accuracy:.4f}")
        print(f"[Test] Precision: {test_precision:.4f}")
        print(f"[Test] Recall:    {test_recall:.4f}")
        print(f"[Test] F1 Score:  {test_f1:.4f}")
        print(f"[Test] ROC-AUC:   {test_roc_auc:.4f}")
        print(f"[Test] Log Loss:  {test_log_loss_val:.4f}")
    else:
        test_mse = mean_squared_error(y_test, y_pred)
        test_mae = mean_absolute_error(y_test, y_pred)

        if model_type == "poisson":
            test_poisson_dev = mean_poisson_deviance(
                y_test, np.clip(y_pred, 1e-9, None)
            )
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
    if model_type == "logistic":
        results[model_name] = {
            "10foldCV": {
                "val_primary_metric": primary_metric,  # Negative ROC-AUC for argmin
                "val_accuracy": mean_accuracy,
                "val_precision": mean_precision,
                "val_recall": mean_recall,
                "val_f1": mean_f1,
                "val_roc_auc": mean_roc_auc,
                "val_log_loss": mean_log_loss,
                "test_accuracy": test_accuracy,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_f1": test_f1,
                "test_roc_auc": test_roc_auc,
                "test_log_loss": test_log_loss_val,
            },
            "y_test": y_test,
            "y_pred": y_pred,
            "y_pred_proba": y_pred_proba,
            "df": df,
            "predictors": predictors,
            "model_type": model_type,
            "is_log_target": is_log_target,
            "is_binary": is_binary,
        }
    else:
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
            "is_binary": is_binary,
        }

    return results[model_name]


def use_best_model(
    results: dict,
    df: pd.DataFrame,
    err_k10: list,
    predictors: list,
    model_type="poisson",
    is_log_target=False,
    is_binary=False,
):
    """Select best model based on CV error and retrain on full 80/20 split"""
    best_model_idx = np.argmin(err_k10)
    best_predictors = predictors[best_model_idx]

    best_model_name = list(results.keys())[best_model_idx]
    original_cv_results = results[best_model_name]["10foldCV"]

    print(f"\n{'='*70}")
    print(f"🎯 BEST MODEL SELECTED: Model {best_model_idx + 1} ({model_type.upper()})")
    print(f"{'='*70}")
    if model_type == "logistic":
        print(f"Best CV ROC-AUC: {original_cv_results['val_roc_auc']:.4f}")
    else:
        print(
            f"Best CV Primary Metric: {original_cv_results['val_primary_metric']:.4f}"
        )
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

    if is_binary or not is_log_target:
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

    pipeline = make_pipeline(num_cols, cat_cols, model_type=model_type)
    pipeline.fit(X_train, y_train)

    # Get probability predictions for logistic regression
    y_pred_proba = None
    if model_type == "logistic":
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
        # Use same lower threshold for consistency
        classification_threshold = 0.4
        y_pred = (y_pred_proba >= classification_threshold).astype(int)
        print(f"   Using classification threshold: {classification_threshold}")
    else:
        y_pred = pipeline.predict(X_test)

    if model_type == "logistic":
        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred)
        test_recall = recall_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred)
        test_roc_auc = roc_auc_score(y_test, y_pred_proba)
        test_log_loss_val = log_loss(y_test, y_pred_proba)

        print(f"\n[FINAL TEST] Accuracy:  {test_accuracy:.4f}")
        print(f"[FINAL TEST] Precision: {test_precision:.4f}")
        print(f"[FINAL TEST] Recall:    {test_recall:.4f}")
        print(f"[FINAL TEST] F1 Score:  {test_f1:.4f}")
        print(f"[FINAL TEST] ROC-AUC:   {test_roc_auc:.4f}")
        print(f"[FINAL TEST] Log Loss:  {test_log_loss_val:.4f}")
    else:
        test_mse = mean_squared_error(y_test, y_pred)
        test_mae = mean_absolute_error(y_test, y_pred)

        print(f"\n[FINAL TEST] RMSE: {np.sqrt(test_mse):.4f}   (MSE: {test_mse:.4f})")
        print(f"[FINAL TEST] MAE : {test_mae:.4f}")

    best_model_results = {
        "10foldCV": original_cv_results,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_pred_proba": y_pred_proba,
        "df": df,
        "predictors": best_predictors,
        "pipeline": pipeline,
        "best_model_idx": best_model_idx,
        "model_type": model_type,
        "is_log_target": is_log_target,
        "is_binary": is_binary,
    }

    return best_model_results
