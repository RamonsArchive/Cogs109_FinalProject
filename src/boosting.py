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
from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_predict, GridSearchCV
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


def train_final_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    predictors: list,
    results: dict,
    model_name: str,
    is_log_target=False,
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
):
    """
    Train final Gradient Boosting model on full training set and evaluate on held-out test set.
    
    IMPORTANT: This function receives pre-split train_df and test_df.
    - Trains on FULL train_df (80% of data)
    - Evaluates on held-out test_df (20% of data) - UNBIASED estimate
    - Also performs 10-fold CV on train_df for model assessment
    
    Args:
        train_df: Training DataFrame (80% of data) - used for training and CV
        test_df: Test DataFrame (20% of data) - ONLY used for final evaluation
        predictors: List of predictor column names
        results: Dictionary to store results
        model_name: Name for this model
        is_log_target: If True, use log1p transformation on target (better for counts)
        n_estimators: Number of boosting stages (B trees)
        learning_rate: Shrinkage parameter (learning rate)
        max_depth: Maximum depth of trees (number of splits)
    """
    target_column = "num_injuries"
    
    num_cols = [c for c in predictors if pd.api.types.is_numeric_dtype(train_df[c])]
    cat_cols = [c for c in predictors if not pd.api.types.is_numeric_dtype(train_df[c])]

    print(f"\n{'='*70}")
    data_type = "Log-Transformed" if is_log_target else "Count"
    print(f"Training {model_name} (Gradient Boosting on {data_type} Data)")
    print(f"{'='*70}")
    print(f"Predictors: {len(predictors)} ({len(num_cols)} numeric, {len(cat_cols)} categorical)")
    print(f"Hyperparameters:")
    print(f"  • n_estimators (B trees): {n_estimators}")
    print(f"  • learning_rate (shrinkage): {learning_rate}")
    print(f"  • max_depth (splits): {max_depth}")
    print(f"\n📊 Data Split:")
    print(f"  • Training set: {len(train_df)} samples (80%)")
    print(f"  • Test set: {len(test_df)} samples (20%) - UNBIASED evaluation")

    # Prepare training data
    X_train = train_df[predictors]
    y_train = train_df[target_column]
    
    # Prepare test data
    X_test = test_df[predictors]
    y_test = test_df[target_column]

    # Store original scale targets before any transformation
    y_train_original = y_train.copy()
    y_test_original = y_test.copy()
    
    # Log-transform target if requested
    if is_log_target:
        y_train = np.log1p(y_train)
        y_test = np.log1p(y_test)
    else:
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

    # Create pipeline
    pipeline = make_boosting_pipeline(num_cols, cat_cols, n_estimators, learning_rate, max_depth)

    # 10-fold cross-validation on training set (for model assessment)
    print(f"\n📈 Performing 10-fold CV on training set...")
    cv = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
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
    if is_log_target:
        # Get CV predictions on log scale
        y_cv_pred_log = cross_val_predict(pipeline, X_train, y_train, cv=cv)
        # Transform predictions back to original scale
        y_cv_pred_original = np.maximum(np.expm1(y_cv_pred_log), 0)
        # Calculate metrics on original scale (using original targets)
        cv_mse_original = mean_squared_error(y_train_original, y_cv_pred_original)
        cv_mae_original = mean_absolute_error(y_train_original, y_cv_pred_original)
        cv_rmse_original = np.sqrt(cv_mse_original)
        print(f"[CV - Original Scale] RMSE: {cv_rmse_original:.4f}   (MSE: {cv_mse_original:.4f})")
        print(f"[CV - Original Scale] MAE : {cv_mae_original:.4f}")

    # Extract CV metrics
    mean_mse = -np.mean(cv_out["test_neg_mse"])
    mean_mae = -np.mean(cv_out["test_neg_mae"])
    mean_r2 = np.mean(cv_out["test_r2"])
    primary_metric = mean_mse

    print(f"[CV] R²: {mean_r2:.4f}")
    print(f"[CV] RMSE: {np.sqrt(mean_mse):.4f}   (MSE: {mean_mse:.4f})")
    print(f"[CV] MAE : {mean_mae:.4f}")

    # Fit on FULL training set and predict on held-out test set
    print(f"\n🎯 Training final model on FULL training set ({len(train_df)} samples)...")
    pipeline.fit(X_train, y_train)
    print(f"✅ Evaluating on held-out test set ({len(test_df)} samples) - UNBIASED estimate")
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

    # Test metrics (on appropriate scale) - UNBIASED estimate
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
    cv_dict = {
        "val_primary_metric": primary_metric,
        "val_mse": mean_mse,
        "val_mae": mean_mae,
        "val_r2": mean_r2,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "test_r2": test_r2,
    }
    # Add original scale CV metrics for log-transformed models
    if is_log_target and cv_mse_original is not None:
        cv_dict["val_mse_original"] = cv_mse_original
        cv_dict["val_mae_original"] = cv_mae_original
        cv_dict["val_rmse_original"] = cv_rmse_original
    
    # Combine train_df and test_df for full dataset (for plotting)
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    
    results[model_name] = {
        "10foldCV": cv_dict,
        "y_test": y_test_for_metrics,
        "y_pred": y_pred_for_metrics,
        "y_test_original": y_test_original if is_log_target else y_test,
        "is_log_target": is_log_target,
        "df": full_df,  # Full dataset for plotting
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


def tune_boosting_hyperparameters(
    train_df: pd.DataFrame,
    predictors: list,
    is_log_target=False,
    n_estimators_grid=[300, 500, 800, 1200],
    learning_rate_grid=[0.01, 0.03, 0.05, 0.1],
    max_depth_grid=[1, 2, 3, 4],
):
    """
    Tune hyperparameters using GridSearchCV on training data only.
    
    IMPORTANT: This function receives train_df which is already split (80% of data).
    Grid search performs 10-fold CV WITHIN this training set only.
    The test set (20%) is never touched during hyperparameter tuning.
    
    Args:
        train_df: Training DataFrame (80% of data) - NO splitting inside this function
        predictors: List of predictor column names
        is_log_target: If True, use log1p transformation on target
        n_estimators_grid: List of n_estimators values to try
        learning_rate_grid: List of learning_rate values to try
        max_depth_grid: List of max_depth values to try
    
    Returns:
        Dictionary with best hyperparameters and grid search results
    """
    target_column = "num_injuries"
    
    num_cols = [c for c in predictors if pd.api.types.is_numeric_dtype(train_df[c])]
    cat_cols = [c for c in predictors if not pd.api.types.is_numeric_dtype(train_df[c])]
    
    data_type = "Log-Transformed" if is_log_target else "Count"
    print(f"\n{'='*70}")
    print(f"🔍 GRID SEARCH: Tuning hyperparameters for {data_type} model")
    print(f"{'='*70}")
    print(f"Training data size: {len(train_df)} (80% of full dataset)")
    print(f"Grid size: {len(n_estimators_grid)} × {len(learning_rate_grid)} × {len(max_depth_grid)} = {len(n_estimators_grid) * len(learning_rate_grid) * len(max_depth_grid)} combinations")
    print(f"Using 10-fold CV for each combination (on training data only)")
    print(f"⚠️  Test set (20%) is NOT used for hyperparameter tuning!")
    
    X_train = train_df[predictors]
    y_train = train_df[target_column]
    
    # Log-transform target if requested
    if is_log_target:
        y_train = np.log1p(y_train)
    else:
        y_train = y_train.astype(int)
    
    print(f"Training samples: {len(X_train)}")
    
    # Create base pipeline (without hyperparameters)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    
    base_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", GradientBoostingRegressor(
            random_state=RANDOM_STATE,
            subsample=0.8,
            max_features='sqrt',
        ))
    ])
    
    # Define parameter grid
    param_grid = {
        "model__n_estimators": n_estimators_grid,
        "model__learning_rate": learning_rate_grid,
        "model__max_depth": max_depth_grid,
    }
    
    # Grid search with 10-fold CV (faster than 10-fold for grid search)
    # Use negative MSE as scoring (lower is better, so we minimize)
    grid_search = GridSearchCV(
        base_pipeline,
        param_grid,
        cv=10,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    
    print(f"\n⏳ Starting grid search (this may take a while)...")
    grid_search.fit(X_train, y_train)
    
    # Get best parameters
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_  # Convert back to positive MSE
    
    print(f"\n✅ Grid search complete!")
    print(f"Best CV MSE: {best_score:.4f}")
    print(f"Best CV RMSE: {np.sqrt(best_score):.4f}")
    print(f"Best hyperparameters:")
    print(f"  • n_estimators: {best_params['model__n_estimators']}")
    print(f"  • learning_rate: {best_params['model__learning_rate']}")
    print(f"  • max_depth: {best_params['model__max_depth']}")
    
    # Get top 5 combinations for display
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df['mean_test_rmse'] = np.sqrt(-results_df['mean_test_score'])
    top_5 = results_df.nsmallest(5, 'mean_test_rmse')[
        ['param_model__n_estimators', 'param_model__learning_rate', 
         'param_model__max_depth', 'mean_test_rmse', 'std_test_score']
    ]
    
    print(f"\n📊 Top 5 hyperparameter combinations:")
    print(top_5.to_string(index=False))
    
    # Get top 3 combinations for training
    top_3 = results_df.nsmallest(3, 'mean_test_rmse')
    top_3_combinations = []
    for idx, row in top_3.iterrows():
        top_3_combinations.append({
            "n_estimators": row['param_model__n_estimators'],
            "learning_rate": row['param_model__learning_rate'],
            "max_depth": row['param_model__max_depth'],
            "cv_rmse": row['mean_test_rmse'],
            "cv_mse": -row['mean_test_score'],
            "rank": len(top_3_combinations) + 1
        })
    
    print(f"\n🏆 Top 3 combinations selected for training:")
    for i, combo in enumerate(top_3_combinations, 1):
        print(f"  {i}. n_estimators={combo['n_estimators']}, "
              f"learning_rate={combo['learning_rate']}, "
              f"max_depth={combo['max_depth']}, "
              f"CV RMSE={combo['cv_rmse']:.4f}")
    
    return {
        "best_n_estimators": best_params['model__n_estimators'],
        "best_learning_rate": best_params['model__learning_rate'],
        "best_max_depth": best_params['model__max_depth'],
        "best_cv_mse": best_score,
        "best_cv_rmse": np.sqrt(best_score),
        "grid_search": grid_search,
        "results_df": results_df,
        "top_3_combinations": top_3_combinations,  # Add top 3 for training
    }


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

    # ========================================================================
    # STEP 1: ONE SPLIT - Create held-out test set (NEVER touch until end)
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: DATA SPLITTING")
    print("=" * 70)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
    print(f"📊 Outer split: Train={len(train_df)} ({len(train_df)/len(df)*100:.1f}%), Test={len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    print("⚠️  Test set will NOT be used for hyperparameter tuning or model selection!")
    print("⚠️  Test set will ONLY be used for final unbiased evaluation at the end")

    # ========================================================================
    # STEP 2: GRID SEARCH WITH CV ON TRAINING DATA ONLY
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: HYPERPARAMETER TUNING WITH GRID SEARCH")
    print("=" * 70)
    print("Grid search performs 10-fold CV WITHIN the training set (80%)")
    print("The test set (20%) is never touched during this step")
    
    # Define grid search parameters
    N_ESTIMATORS_GRID = [300, 500, 800, 1200]
    LEARNING_RATE_GRID = [0.01, 0.03, 0.05, 0.1]
    MAX_DEPTH_GRID = [1, 2, 3, 4]
    
    # Tune hyperparameters for COUNT model (on training data only)
    print("\n" + "=" * 70)
    print("COUNT MODEL: Hyperparameter Tuning")
    print("=" * 70)
    count_tuning = tune_boosting_hyperparameters(
        train_df=train_df,  # ← Only the 80% training data
        predictors=PREDICTORS,
        is_log_target=False,
        n_estimators_grid=N_ESTIMATORS_GRID,
        learning_rate_grid=LEARNING_RATE_GRID,
        max_depth_grid=MAX_DEPTH_GRID,
    )
    
    # ========================================================================
    # STEP 3: TRAIN FINAL MODELS WITH BEST HYPERPARAMETERS
    # Retrain on FULL training set, evaluate on held-out test set
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: TRAINING FINAL MODELS WITH BEST HYPERPARAMETERS")
    print("=" * 70)
    print("Using the BEST overall combination from grid search")
    print("Retraining on FULL training set (80%), evaluating on held-out test set (20%)")
    
    # ========================================================================
    # GRADIENT BOOSTING ON COUNT DATA (Top 3 Models)
    # ========================================================================
    print("\n" + "=" * 70)
    print("TRAINING TOP 3 COUNT MODELS")
    print("=" * 70)
    print("Training the top 3 hyperparameter combinations from grid search")
    print("Each model will be trained and evaluated separately")
    
    count_results = {}
    count_models = []
    
    # Train all top 3 models
    for i, combo in enumerate(count_tuning['top_3_combinations'], 1):
        print(f"\n{'='*70}")
        print(f"COUNT MODEL #{i} (Rank {combo['rank']} from grid search)")
        print(f"{'='*70}")
        print(f"Hyperparameters:")
        print(f"  • n_estimators: {combo['n_estimators']}")
        print(f"  • learning_rate: {combo['learning_rate']}")
        print(f"  • max_depth: {combo['max_depth']}")
        print(f"  • Grid Search CV RMSE: {combo['cv_rmse']:.4f}")
        
        model_name = f"boosting_count_rank{combo['rank']}"
        
        count_model = train_final_model(
            train_df=train_df,  # ← Full 80% for training
            test_df=test_df,    # ← Held-out 20% for final evaluation
            predictors=PREDICTORS,
            results=count_results,
            model_name=model_name,
            is_log_target=False,
            n_estimators=combo['n_estimators'],
            learning_rate=combo['learning_rate'],
            max_depth=combo['max_depth'],
        )
        
        count_models.append({
            "model": count_model,
            "name": model_name,
            "rank": combo['rank'],
            "hyperparams": combo
        })
        
        # Graph each model
        is_best = (combo['rank'] == 1)  # Mark rank 1 as best
        graph_boosting_model(count_model, model_name, is_best=is_best)

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("✅ GRADIENT BOOSTING TRAINING COMPLETE!")
    print("=" * 70)
    
    print(f"\n{'='*70}")
    print("📊 FINAL RESULTS SUMMARY - TOP 3 MODELS")
    print(f"{'='*70}")
    
    # Summary for all top 3 models
    for model_info in count_models:
        model = model_info["model"]
        combo = model_info["hyperparams"]
        rank = model_info["rank"]
        
        print(f"\n🎯 COUNT MODEL #{rank} (Rank {rank} from grid search):")
        print(f"   Hyperparameters:")
        print(f"   • n_estimators: {combo['n_estimators']}")
        print(f"   • learning_rate: {combo['learning_rate']}")
        print(f"   • max_depth: {combo['max_depth']}")
        print(f"   • Grid Search CV RMSE: {combo['cv_rmse']:.4f}")
        print(f"\n   Final Model Performance:")
        print(f"   • CV R²: {model['10foldCV']['val_r2']:.4f}")
        print(f"   • Test R²: {model['10foldCV']['test_r2']:.4f}")
        print(f"   • CV RMSE: {np.sqrt(model['10foldCV']['val_mse']):.4f}")
        print(f"   • Test RMSE: {np.sqrt(model['10foldCV']['test_mse']):.4f}")
        print(f"   • Model name: {model_info['name']}")
    
    print(f"\n📁 All plots saved to: plots/boosting/")
    print(f"   • Rank 1 (best): boosting_count_rank1_* (marked as BEST)")
    print(f"   • Rank 2: boosting_count_rank2_*")
    print(f"   • Rank 3: boosting_count_rank3_*")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
