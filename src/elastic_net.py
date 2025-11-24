"""
Elastic Net Regression for Poisson and Linear Models
Uses Elastic Net regularization (L1 + L2) for automatic feature selection
"""

import os
import numpy as np
import pandas as pd
from load import load_data
from clean import clean_data, log_transform_data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_validate, GridSearchCV, cross_val_predict
from sklearn.linear_model import PoissonRegressor, ElasticNet
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_poisson_deviance,
    r2_score,
)
from elastic_graphs import graph_elastic_net_model
from pathlib import Path

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# origanl was alpha = 1.0 and l1_ratio = 0.5
def make_elastic_net_pipeline(num_cols, cat_cols, model_type="poisson", alpha=1, l1_ratio=0.5):
    """Create pipeline with Elastic Net regularization
    
    Note: PoissonRegressor only supports L2 regularization (Ridge) via alpha.
    For true Elastic Net, use linear model type.
    """
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    if model_type == "poisson":
        # PoissonRegressor only supports L2 regularization (Ridge) via alpha
        # sklearn's PoissonRegressor doesn't support l1_ratio
        model = PoissonRegressor(alpha=alpha, max_iter=5000)
    elif model_type == "linear":
        # ElasticNet for linear regression supports full Elastic Net (L1 + L2)
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=5000, random_state=RANDOM_STATE)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return Pipeline([("preprocessor", pre), ("model", model)])


def train_elastic_net_model(
    df: pd.DataFrame,
    predictors: list,
    results: dict,
    model_name: str,
    model_type="poisson",
    is_log_target=False,
    alpha=1,
    l1_ratio=0.5,
    tune_hyperparameters=False,
):
    """
    Train Elastic Net model with optional hyperparameter tuning
    
    Args:
        df: DataFrame with data
        predictors: List of predictor column names
        results: Dictionary to store results
        model_name: Name for this model
        model_type: 'poisson' or 'linear'
        is_log_target: True if target is log-transformed
        alpha: Regularization strength (used if tune_hyperparameters=False)
        l1_ratio: L1/L2 mixing (0=Ridge, 1=Lasso, 0.5=Elastic Net)
        tune_hyperparameters: If True, use GridSearchCV to find best alpha and l1_ratio
                              NOTE: When True, the alpha and l1_ratio parameters are IGNORED
    """
    target_column = "num_injuries"
    
    num_cols = [c for c in predictors if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in predictors if not pd.api.types.is_numeric_dtype(df[c])]

    print(f"\n{'='*70}")
    data_type = "LOG-TRANSFORMED" if is_log_target else "COUNT"
    print(f"Training {model_name} (Elastic Net {model_type.upper()} on {data_type} data)")
    print(f"{'='*70}")
    print(f"Predictors: {len(predictors)} ({len(num_cols)} numeric, {len(cat_cols)} categorical)")

    # Split data
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
    X_train = train_df[predictors]
    y_train = train_df[target_column]
    X_test = test_df[predictors]
    y_test = test_df[target_column]

    if not is_log_target:
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

    print(f"Train size: {train_df.shape[0]} | Test size: {test_df.shape[0]}")

    # Hyperparameter tuning
    if tune_hyperparameters:
        if model_type == "poisson":
            print("\n🔍 Tuning hyperparameters (alpha only - Poisson uses L2 regularization)...")
        else:
            print("\n🔍 Tuning hyperparameters (alpha, l1_ratio)...")
        
        # Create base pipeline without hyperparameters
        base_pipeline = Pipeline([
            ("preprocessor", ColumnTransformer(
                transformers=[
                    ("num", StandardScaler(), num_cols),
                    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
                ]
            )),
            ("model", None)  # Will be set in param_grid
            ])
        
        # Define parameter grid
        if model_type == "poisson":
            # PoissonRegressor only supports alpha (L2 regularization)
            param_grid = {
                "model": [PoissonRegressor(max_iter=5000)],
                "model__alpha": [0.01, 0.1, 1.0, 10.0],
            }
            scoring = "neg_mean_poisson_deviance"
        else:  # linear
            # ElasticNet supports both alpha and l1_ratio (true Elastic Net)
            param_grid = {
                "model": [ElasticNet(max_iter=5000, random_state=RANDOM_STATE)],
                "model__alpha": [0.01, 0.1, 1.0, 10.0],
                "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
            }
            scoring = "neg_mean_squared_error"
        
        # Grid search with 5-fold CV
        grid_search = GridSearchCV(
            base_pipeline,
            param_grid,
            cv=5,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        best_alpha = grid_search.best_params_["model__alpha"]
        if model_type == "poisson":
            # Poisson doesn't support l1_ratio, set to None
            best_l1_ratio = None
            print(f"✅ Best hyperparameters: alpha={best_alpha} (L2 regularization only)")
        else:
            best_l1_ratio = grid_search.best_params_["model__l1_ratio"]
            print(f"✅ Best hyperparameters: alpha={best_alpha}, l1_ratio={best_l1_ratio}")
        print(f"   Best CV score: {grid_search.best_score_:.4f}")
        
        # Use best model (already fitted on full training set)
        pipeline = grid_search.best_estimator_
    else:
        if model_type == "poisson":
            print(f"\nUsing fixed hyperparameters: alpha={alpha} (L2 regularization only)")
            best_l1_ratio = None
        else:
            print(f"\nUsing fixed hyperparameters: alpha={alpha}, l1_ratio={l1_ratio}")
            best_l1_ratio = l1_ratio
        pipeline = make_elastic_net_pipeline(num_cols, cat_cols, model_type, alpha, l1_ratio)
        pipeline.fit(X_train, y_train)
        best_alpha = alpha

    # 10-fold cross-validation (will refit on each fold)
    cv = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    
    if model_type == "poisson":
        scoring = {
            "neg_poisson_dev": "neg_mean_poisson_deviance",
            "neg_mse": "neg_mean_squared_error",
            "neg_mae": "neg_mean_absolute_error",
        }
    else:  # linear
        scoring = {
            "neg_mse": "neg_mean_squared_error",
            "neg_mae": "neg_mean_absolute_error",
            "r2": "r2",
        }

    cv_out = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)
    
    # For log-transformed models, also calculate CV metrics on original scale
    cv_mse_original = None
    cv_mae_original = None
    cv_rmse_original = None
    if is_log_target and model_type != "poisson":
        # Get CV predictions on log scale
        y_train_original = np.expm1(y_train)  # Original scale targets
        y_cv_pred_log = cross_val_predict(pipeline, X_train, y_train, cv=cv)
        # Transform predictions back to original scale
        y_cv_pred_original = np.maximum(np.expm1(y_cv_pred_log), 0)
        # Calculate metrics on original scale
        cv_mse_original = mean_squared_error(y_train_original, y_cv_pred_original)
        cv_mae_original = mean_absolute_error(y_train_original, y_cv_pred_original)
        cv_rmse_original = np.sqrt(cv_mse_original)
        print(f"[CV - Original Scale] RMSE: {cv_rmse_original:.4f}   (MSE: {cv_mse_original:.4f})")
        print(f"[CV - Original Scale] MAE : {cv_mae_original:.4f}")

    # Extract CV metrics
    mean_mse = -np.mean(cv_out["test_neg_mse"])
    mean_mae = -np.mean(cv_out["test_neg_mae"])

    if model_type == "poisson":
        mean_poisson_dev = -np.mean(cv_out["test_neg_poisson_dev"])
        print(f"\n[CV] Mean Poisson deviance: {mean_poisson_dev:.4f}")
        primary_metric = mean_poisson_dev
    else:
        mean_r2 = np.mean(cv_out["test_r2"])
        print(f"[CV] R²: {mean_r2:.4f}")
        primary_metric = mean_mse

    print(f"[CV] RMSE: {np.sqrt(mean_mse):.4f}   (MSE: {mean_mse:.4f})")
    print(f"[CV] MAE : {mean_mae:.4f}")

    # Test set predictions
    y_pred = pipeline.predict(X_test)

    # Test metrics
    test_mse = mean_squared_error(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)

    if model_type == "poisson":
        test_poisson_dev = mean_poisson_deviance(y_test, np.clip(y_pred, 1e-9, None))
        print(f"[Test] Poisson deviance: {test_poisson_dev:.4f}")
    else:
        test_r2 = r2_score(y_test, y_pred)
        print(f"[Test] R²: {test_r2:.4f}")

    print(f"[Test] RMSE: {np.sqrt(test_mse):.4f}   (MSE: {test_mse:.4f})")
    print(f"[Test] MAE : {test_mae:.4f}")

    # If log-transformed, show performance in original scale
    if is_log_target:
        y_test_original = np.expm1(y_test)
        y_pred_original = np.maximum(np.expm1(y_pred), 0)
        
        original_rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
        original_mae = mean_absolute_error(y_test_original, y_pred_original)
        original_r2 = r2_score(y_test_original, y_pred_original)
        
        print(f"\n[Test - Original Scale]")
        print(f"RMSE: {original_rmse:.4f}")
        print(f"MAE : {original_mae:.4f}")
        print(f"R²  : {original_r2:.4f}")

    # Get feature coefficients (for feature importance)
    model = pipeline.named_steps["model"]
    feature_names = []
    
    # Get feature names after preprocessing
    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names.extend(num_cols)
    
    if cat_cols:
        cat_encoder = preprocessor.named_transformers_["cat"]
        for i, col in enumerate(cat_cols):
            categories = cat_encoder.categories_[i]
            feature_names.extend([f"{col}_{cat}" for cat in categories])
    
    # Get coefficients
    if hasattr(model, "coef_"):
        coefficients = model.coef_
    else:
        coefficients = None

    # Store results
    cv_dict = {
        "val_primary_metric": primary_metric,
        "val_mse": mean_mse,
        "val_mae": mean_mae,
        "test_mse": test_mse,
        "test_mae": test_mae,
    }
    # Add original scale CV metrics for log-transformed models
    if is_log_target and cv_mse_original is not None:
        cv_dict["val_mse_original"] = cv_mse_original
        cv_dict["val_mae_original"] = cv_mae_original
        cv_dict["val_rmse_original"] = cv_rmse_original
    
    results[model_name] = {
        "10foldCV": cv_dict,
        "y_test": y_test,
        "y_pred": y_pred,
        "df": df,
        "predictors": predictors,
        "model_type": model_type,
        "is_log_target": is_log_target,
        "pipeline": pipeline,
        "alpha": best_alpha,
        "l1_ratio": best_l1_ratio if model_type != "poisson" else None,
        "coefficients": coefficients,
        "feature_names": feature_names,
    }
    
    if model_type == "poisson":
        results[model_name]["10foldCV"]["val_poisson_dev"] = mean_poisson_dev
        results[model_name]["10foldCV"]["test_poisson_dev"] = test_poisson_dev
    else:
        results[model_name]["10foldCV"]["val_r2"] = mean_r2
        results[model_name]["10foldCV"]["test_r2"] = test_r2

    return results[model_name]


def main():
    """
    Main function for Elastic Net models
    
    IMPORTANT: To use your own alpha and l1_ratio values:
    - Set tune_hyperparameters=False in the train_elastic_net_model() calls below
    - Then modify the alpha and l1_ratio parameters
    - If tune_hyperparameters=True, GridSearchCV will find the best values and ignore your specified ones
    
    NOTE: Custom hyperparameters are NOT guaranteed to be worse!
    - GridSearchCV only searches within a fixed grid (alpha: [0.01, 0.1, 1.0, 10.0], l1_ratio: [0.1, 0.3, 0.5, 0.7, 0.9])
    - Your custom values might be better if:
      * You have domain knowledge about appropriate regularization
      * You want more/less regularization than the grid provides
      * You want to test specific values (e.g., pure Lasso l1_ratio=1.0, pure Ridge l1_ratio=0.0)
    - GridSearchCV optimizes CV score, but higher CV score doesn't always mean better generalization
    - Sometimes more regularization (higher alpha) can prevent overfitting better
    """
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(
        curr_dir,
        "../datasets/df_master_schedule_injury_surface_2019_23_weather_distance_days_since_last_game.csv",
    )

    # Use Model 6 (kitchen sink) - Elastic Net will handle feature selection automatically
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
    print("ELASTIC NET REGRESSION MODELS")
    print("=" * 70)
    print(f"Using {len(PREDICTORS)} predictors from Model 6 (kitchen sink)")
    print("Elastic Net will automatically select important features via L1 regularization")
    print("=" * 70)

    # Load and prepare data
    df = load_data(data_dir)
    df = clean_data(df)
    target_column = "num_injuries"
    log_df = log_transform_data(df, target_column)

    # Perform correlation analysis (optional)
    from exploratory import correlation_analysis
    corr_original, corr_log = correlation_analysis(df, log_df, target_column)

    # ========================================================================
    # ELASTIC NET POISSON MODELS
    # ========================================================================
    print("\n" + "=" * 70)
    print("TRAINING ELASTIC NET POISSON REGRESSION (Count Data)")
    print("=" * 70)
    poisson_results = {}
    
    poisson_model = train_elastic_net_model(
        df=df,
        predictors=PREDICTORS,
        results=poisson_results,
        model_name="elastic_net_poisson",
        model_type="poisson",
        is_log_target=False,
        alpha=0.01,  # Change this to use different alpha values
        tune_hyperparameters=True,  # Set to False to use specified alpha/l1_ratio
    )
    
    graph_elastic_net_model(poisson_model, "elastic_net_poisson", is_best=False)

    # ========================================================================
    # ELASTIC NET LINEAR MODELS (Log-Transformed)
    # ========================================================================
    print("\n" + "=" * 70)
    print("TRAINING ELASTIC NET LINEAR REGRESSION (Log-Transformed Target)")
    print("=" * 70)
    linear_results = {}
    

    #  Adjusted R² = 21.7% l1_ratio = 0.8
    linear_model = train_elastic_net_model(
        df=log_df,
        predictors=PREDICTORS,
        results=linear_results,
        model_name="elastic_net_linear",
        model_type="linear",
        is_log_target=True,
        alpha=0.001,  # Change this to use different alpha values
        l1_ratio=0.8,  # Change this to use different l1_ratio values (0=Ridge, 1=Lasso, 0.5=Elastic Net)
        tune_hyperparameters=False,  # Set to False to use specified alpha/l1_ratio
    )
    
    graph_elastic_net_model(linear_model, "elastic_net_linear", is_best=False)

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("✅ ELASTIC NET TRAINING COMPLETE!")
    print("=" * 70)
    
    print(f"\n📊 POISSON MODEL (L2 Regularization):")
    print(f"   CV Poisson Deviance: {poisson_model['10foldCV']['val_poisson_dev']:.4f}")
    print(f"   Test Poisson Deviance: {poisson_model['10foldCV']['test_poisson_dev']:.4f}")
    print(f"   Best alpha: {poisson_model['alpha']} (Note: Poisson uses L2 only, not full Elastic Net)")
    
    print(f"\n📊 LINEAR MODEL (Log-Transformed, Full Elastic Net):")
    print(f"   CV R²: {linear_model['10foldCV']['val_r2']:.4f}")
    print(f"   Test R²: {linear_model['10foldCV']['test_r2']:.4f}")
    print(f"   Best alpha: {linear_model['alpha']}, l1_ratio: {linear_model['l1_ratio']}")
    
    print(f"\n📁 All plots saved to: plots/elastic/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
