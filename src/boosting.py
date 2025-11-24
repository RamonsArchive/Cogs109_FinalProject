"""
Gradient Boosting Models for NFL Injury Prediction
Uses GradientBoostingRegressor on count data
StandardScaler and OneHotEncoder handle data standardization automatically
"""

import os
import numpy as np
import pandas as pd
from load import load_data
from clean import clean_data, log_transform_data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from graph_boosting import graph_boosting_model

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def make_boosting_pipeline(num_cols, cat_cols, n_estimators=200, learning_rate=0.1, max_depth=3):
    """Create pipeline with Gradient Boosting Regressor
    
    Args:
        num_cols: List of numeric column names
        cat_cols: List of categorical column names
        n_estimators: Number of boosting stages
        learning_rate: Learning rate (shrinkage)
        max_depth: Maximum depth of trees
    
    Returns:
        Pipeline with preprocessing and GradientBoostingRegressor
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
        subsample=0.8,  # Use 80% of samples for each tree (helps prevent overfitting)
        max_features='sqrt',  # Use sqrt(n_features) for each tree
    )
    
    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def train_boosting_model(
    df: pd.DataFrame,
    predictors: list,
    results: dict,
    model_name: str,
    is_log_target=False,
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
):
    """
    Train Gradient Boosting model with cross-validation
    
    Args:
        df: DataFrame with data
        predictors: List of predictor column names
        results: Dictionary to store results
        model_name: Name for this model
        is_log_target: If True, use log1p transformation on target (better for counts)
        n_estimators: Number of boosting stages (B trees)
        learning_rate: Shrinkage parameter (learning rate)
        max_depth: Maximum depth of trees (number of splits)
    """
    target_column = "num_injuries"
    
    num_cols = [c for c in predictors if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in predictors if not pd.api.types.is_numeric_dtype(df[c])]

    print(f"\n{'='*70}")
    data_type = "Log-Transformed" if is_log_target else "Count"
    print(f"Training {model_name} (Gradient Boosting on {data_type} Data)")
    print(f"{'='*70}")
    print(f"Predictors: {len(predictors)} ({len(num_cols)} numeric, {len(cat_cols)} categorical)")
    print(f"Hyperparameters:")
    print(f"  • n_estimators (B trees): {n_estimators}")
    print(f"  • learning_rate (shrinkage): {learning_rate}")
    print(f"  • max_depth (splits): {max_depth}")

    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
    X_train = train_df[predictors]
    y_train = train_df[target_column]
    X_test = test_df[predictors]
    y_test = test_df[target_column]

    # Log-transform target if requested
    if is_log_target:
        y_train = np.log1p(y_train)
        y_test_original = y_test.copy()
        y_test = np.log1p(y_test)
    else:
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
        y_test_original = y_test.copy()

    print(f"Train size: {train_df.shape[0]} | Test size: {test_df.shape[0]}")

    # Create pipeline
    pipeline = make_boosting_pipeline(num_cols, cat_cols, n_estimators, learning_rate, max_depth)

    # 10-fold cross-validation
    cv = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "neg_mse": "neg_mean_squared_error",
        "neg_mae": "neg_mean_absolute_error",
        "r2": "r2",
    }

    cv_out = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)

    # Extract CV metrics
    mean_mse = -np.mean(cv_out["test_neg_mse"])
    mean_mae = -np.mean(cv_out["test_neg_mae"])
    mean_r2 = np.mean(cv_out["test_r2"])
    primary_metric = mean_mse

    print(f"\n[CV] R²: {mean_r2:.4f}")
    print(f"[CV] RMSE: {np.sqrt(mean_mse):.4f}   (MSE: {mean_mse:.4f})")
    print(f"[CV] MAE : {mean_mae:.4f}")

    # Fit on full training set and predict on test set
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # If log-transformed, convert predictions back to original scale
    if is_log_target:
        y_pred_original = np.maximum(np.expm1(y_pred), 0)  # expm1 inverse of log1p
        y_test_for_metrics = y_test_original
        y_pred_for_metrics = y_pred_original
        
        # Also calculate metrics on log scale
        test_mse_log = mean_squared_error(y_test, y_pred)
        test_mae_log = mean_absolute_error(y_test, y_pred)
        test_r2_log = r2_score(y_test, y_pred)
        
        print(f"[Test - Log Scale] R²: {test_r2_log:.4f}")
        print(f"[Test - Log Scale] RMSE: {np.sqrt(test_mse_log):.4f}")
    else:
        y_test_for_metrics = y_test
        y_pred_for_metrics = y_pred

    # Test metrics (on appropriate scale)
    test_mse = mean_squared_error(y_test_for_metrics, y_pred_for_metrics)
    test_mae = mean_absolute_error(y_test_for_metrics, y_pred_for_metrics)
    test_r2 = r2_score(y_test_for_metrics, y_pred_for_metrics)

    print(f"[Test] R²: {test_r2:.4f}")
    print(f"[Test] RMSE: {np.sqrt(test_mse):.4f}   (MSE: {test_mse:.4f})")
    print(f"[Test] MAE : {test_mae:.4f}")

    # Get feature importances
    model = pipeline.named_steps["model"]
    feature_importances = model.feature_importances_
    
    # Get feature names after preprocessing
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = []
    feature_names.extend(num_cols)
    
    if cat_cols:
        cat_encoder = preprocessor.named_transformers_["cat"]
        for i, col in enumerate(cat_cols):
            categories = cat_encoder.categories_[i]
            feature_names.extend([f"{col}_{cat}" for cat in categories])

    # Store results
    results[model_name] = {
        "10foldCV": {
            "val_primary_metric": primary_metric,
            "val_mse": mean_mse,
            "val_mae": mean_mae,
            "val_r2": mean_r2,
            "test_mse": test_mse,
            "test_mae": test_mae,
            "test_r2": test_r2,
        },
        "y_test": y_test_for_metrics,
        "y_pred": y_pred_for_metrics,
        "y_test_original": y_test_original if is_log_target else y_test,
        "is_log_target": is_log_target,
        "df": df,
        "predictors": predictors,
        "model_type": "boosting",
        "pipeline": pipeline,
        "feature_importances": feature_importances,
        "feature_names": feature_names,
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "max_depth": max_depth,
    }

    return results[model_name]


def main():
    """
    Main function for Gradient Boosting models
    
    Uses Model 6 (kitchen sink) predictors - no feature selection needed
    since boosting automatically handles feature importance.
    """
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(
        curr_dir,
        "../datasets/df_master_schedule_injury_surface_2019_23_weather_distance_days_since_last_game.csv",
    )

    # Model 6: Kitchen sink (all potentially relevant features)
    # No feature selection needed - boosting handles it automatically
    PREDICTORS = [
        "surface_type",
        "Avg_Temp",
        "Avg_Wind_MPH",
        "Avg_Humidity_Percent",
        "Avg_Percipitation_Prob_Percent",
        "day",
        "week",
        "season",
        "stadium",
        "surface",
        "dome",
        "num_plays",
        "yds_w",
        "yds_l",
        "tov_w",
        "tov_l",
        "HOME_day_since_last_game",
        "AWAY_day_since_last_game",
        "distance_miles",
    ]

    print("\n" + "=" * 70)
    print("GRADIENT BOOSTING MODELS")
    print("=" * 70)
    print(f"Using {len(PREDICTORS)} predictors from Model 6 (kitchen sink)")
    print("Boosting automatically handles feature importance and selection")
    print("=" * 70)

    # Load and prepare data
    df = load_data(data_dir)
    df = clean_data(df)
    log_df = log_transform_data(df, "num_injuries")

    # ========================================================================
    # GRADIENT BOOSTING ON COUNT DATA
    # ========================================================================
    print("\n" + "=" * 70)
    print("TRAINING GRADIENT BOOSTING (Count Data)")
    print("=" * 70)
    
    # HYPERPARAMETERS FOR COUNT DATA MODEL - Adjust these values as needed
    COUNT_N_ESTIMATORS = 1200    # Number of B trees
    COUNT_LEARNING_RATE = 0.01   # Shrinkage parameter
    COUNT_MAX_DEPTH = 3          # Maximum depth (number of splits)
    
    count_results = {}
    
    count_model = train_boosting_model(
        df=df,
        predictors=PREDICTORS,
        results=count_results,
        model_name="boosting_count",
        is_log_target=False,
        n_estimators=COUNT_N_ESTIMATORS,
        learning_rate=COUNT_LEARNING_RATE,
        max_depth=COUNT_MAX_DEPTH,
    )
    
    graph_boosting_model(count_model, "boosting_count", is_best=False)

    # ========================================================================
    # GRADIENT BOOSTING ON LOG-TRANSFORMED DATA
    # Better for count data - reduces impact of outliers
    # ========================================================================
    print("\n" + "=" * 70)
    print("TRAINING GRADIENT BOOSTING (Log-Transformed Target)")
    print("=" * 70)
    
    # HYPERPARAMETERS FOR LOG-TRANSFORMED MODEL - Adjust these values as needed
    LOG_N_ESTIMATORS = 300      # Number of B trees
    LOG_LEARNING_RATE = 0.001   # Shrinkage parameter
    LOG_MAX_DEPTH = 1            # Maximum depth (number of splits)
    
    log_results = {}
    
    log_model = train_boosting_model(
        df=df,  # Use original df, we'll log-transform target inside the function
        predictors=PREDICTORS,
        results=log_results,
        model_name="boosting_log",
        is_log_target=True,
        n_estimators=LOG_N_ESTIMATORS,
        learning_rate=LOG_LEARNING_RATE,
        max_depth=LOG_MAX_DEPTH,
    )
    
    graph_boosting_model(log_model, "boosting_log", is_best=False)

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("✅ GRADIENT BOOSTING TRAINING COMPLETE!")
    print("=" * 70)
    
    print(f"\n📊 COUNT DATA MODEL:")
    print(f"   CV R²: {count_model['10foldCV']['val_r2']:.4f}")
    print(f"   Test R²: {count_model['10foldCV']['test_r2']:.4f}")
    print(f"   CV RMSE: {np.sqrt(count_model['10foldCV']['val_mse']):.4f}")
    print(f"   Test RMSE: {np.sqrt(count_model['10foldCV']['test_mse']):.4f}")
    
    print(f"\n📊 LOG-TRANSFORMED MODEL:")
    print(f"   CV R²: {log_model['10foldCV']['val_r2']:.4f}")
    print(f"   Test R²: {log_model['10foldCV']['test_r2']:.4f}")
    print(f"   CV RMSE: {np.sqrt(log_model['10foldCV']['val_mse']):.4f}")
    print(f"   Test RMSE: {np.sqrt(log_model['10foldCV']['test_mse']):.4f}")
    
    print(f"\n📁 All plots saved to: plots/boosting/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
