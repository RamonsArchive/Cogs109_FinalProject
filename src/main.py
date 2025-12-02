import os
from typing import Any
from load import load_data
from clean import clean_data
from clean import log_transform_data
from model import train_model, use_best_model
from graph import graph_model, exploratory_data_analysis
from exploratory import correlation_analysis
from feature_selection import backward_selection, compare_models_with_without_selection
import numpy as np
import matplotlib.pyplot as plt


def main():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(
        curr_dir,
        "../datasets/df_master_schedule_injury_surface_2019_23_weather_distance_days_since_last_game.csv",
    )
    # Define all predictor sets to test
    REGRESSION_PREDICTORS = [
        # Model 1: Baseline
        [
            "surface_type",
            "day",
            "week",
            "season",
            "stadium",
            "surface",
            "dome",
        ],
        # Model 2: Add Avg_Temp
        [
            "surface_type",
            "Avg_Temp",
            "day",
            "week",
            "season",
            "stadium",
            "surface",
            "dome",
        ],
        # Model 3: Add game intensity metrics
        [
            "surface_type",
            "Avg_Temp",
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
        ],
        # Model 4: Add rest/travel factors
        [
            "surface_type",
            "Avg_Temp",
            "day",
            "week",
            "season",
            "stadium",
            "surface",
            "dome",
            "HOME_day_since_last_game",
            "AWAY_day_since_last_game",
            "distance_miles",
        ],
        # Model 5: Add weather conditions
        [
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
        ],
        # Model 6: Kitchen sink (all potentially relevant features)
        [
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
        ],
    ]

    model_names = [
        "baseline",
        "Avg_Temp",
        "game_intensity",
        "rest_travel",
        "weather",
        "kitchen_sink_all",
    ]

    log_model_names = [
        "baseline_log",
        "Avg_Temp_log",
        "game_intensity_log",
        "rest_travel_log",
        "weather_log",
        "kitchen_sink_all_log",
    ]

    logistic_model_names = [
        "baseline_binary",
        "Avg_Temp_binary",
        "game_intensity_binary",
        "rest_travel_binary",
        "weather_binary",
        "kitchen_sink_all_binary",
    ]

    # Load data once
    df = load_data(data_dir)
    df = clean_data(df)

    target_column = "num_injuries"
    
    # ========================================================================
    # STEP 1: ONE SPLIT - Create held-out test set (NEVER touch until end)
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: DATA SPLITTING")
    print("=" * 70)
    from sklearn.model_selection import train_test_split
    from model import RANDOM_STATE
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
    print(f"📊 Outer split: Train={len(train_df)} ({len(train_df)/len(df)*100:.1f}%), Test={len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    print("⚠️  Test set will NOT be used for model training or selection!")
    print("⚠️  Test set will ONLY be used for final unbiased evaluation")
    print("=" * 70)

    # Create transformed datasets from the training split only
    log_train_df = log_transform_data(train_df, target_column)
    log_test_df = log_transform_data(test_df, target_column)
    
    # For full dataset operations (like correlation analysis), use full df
    log_df = log_transform_data(df, target_column)

    # Create binary target for logistic regression (1 if num_injuries > 4, else 0)
    binary_train_df = train_df.copy()
    binary_train_df[target_column] = (train_df[target_column] > 4).astype(int)
    binary_test_df = test_df.copy()
    binary_test_df[target_column] = (test_df[target_column] > 4).astype(int)
    
    # For full dataset operations
    binary_df = df.copy()
    binary_df[target_column] = (df[target_column] > 4).astype(int)
    
    print(f"\n✅ Created binary target: 1 if num_injuries > 4, else 0")
    print(f"   Class distribution (full): {np.bincount(binary_df[target_column])}")
    print(f"   Class distribution (train): {np.bincount(binary_train_df[target_column])}")
    print(f"   Class distribution (test): {np.bincount(binary_test_df[target_column])}")

    # Perform exploratory data analysis
    # exploratory_data_analysis(df, log_df, target_column)

    # Perform correlation analysis (on full dataset for exploration)
    # corr_original, corr_log = correlation_analysis(df, log_df, target_column)

    # ========================================================================
    # BACKWARD SELECTION - Find optimal features from Model 6 (kitchen sink)
    # ========================================================================
    # Run backward selection ONCE on logistic regression (binary classification)
    # Then apply the selected features to ALL three model types
    print("\n" + "=" * 70)
    print("🔍 PERFORMING BACKWARD SELECTION ON MODEL 6")
    print("=" * 70)
    print("Using LOGISTIC REGRESSION for feature selection")
    print(f"Significance level: α = 0.1")
    print(f"Starting with {len(REGRESSION_PREDICTORS[5])} predictors from Model 6")
    print("=" * 70 + "\n")

    # Backward selection uses training data only (no data leakage)
    selected_features = backward_selection(
        df=binary_train_df,  # Use training data only
        predictors=REGRESSION_PREDICTORS[5],  # Model 6: kitchen_sink_all
        target_column=target_column,
        significance_level=0.1, # Adjusted R² = 26.5% 
        model_type="logistic",
        is_binary=True,
        verbose=True,
    )

    # ========================================================================
    # ADD SELECTED FEATURES AS MODEL 7
    # ========================================================================
    # Model 7 will be trained alongside Models 1-6 in the training loops below.
    # The comparison between Model 6 (all features) and Model 7 (selected features)
    # will happen automatically when use_best_model() selects the best model based
    # on 10-fold CV performance.
    #
    # NOTE: We do NOT run a separate comparison here because:
    # 1. Model 6 and Model 7 will both be trained with full 10-fold CV
    # 2. use_best_model() will compare all models (including 6 vs 7) and select the best
    # 3. This avoids redundant computation and ensures consistent evaluation
    #
    REGRESSION_PREDICTORS.append(selected_features)
    model_names.append("backward_selected")
    log_model_names.append("backward_selected_log")
    logistic_model_names.append("backward_selected_binary")

    print(f"\n✅ Model 7 created with {len(selected_features)} selected features")
    print(
        f"   Will be tested across all three model types (Poisson, Logistic, Linear)\n"
    )

    # POISSON MODELS on original counts
    print("\n" + "=" * 70)
    print("TRAINING POISSON REGRESSION MODELS (Count Data)")
    print("=" * 70)
    poisson_results = {}
    poisson_errors = []

    for i, (model_name, predictors) in enumerate(
        zip(model_names, REGRESSION_PREDICTORS), 1
    ):
        result_model = train_model(
            df=train_df,  # Pass training df (for feature type detection)
            predictors=predictors,
            results=poisson_results,
            model_name=model_name,
            model_type="poisson",
            is_log_target=False,
            is_binary=False,
            train_df=train_df,  # Pre-split training data
            test_df=test_df,    # Pre-split test data
        )
        poisson_errors.append(result_model["10foldCV"]["val_primary_metric"])
        graph_model(result_model, model_name, model_number=i)

    # LOGISTIC REGRESSION MODELS on binary target
    print("\n" + "=" * 70)
    print("TRAINING LOGISTIC REGRESSION MODELS (Binary Classification)")
    print("=" * 70)
    logistic_results = {}
    logistic_errors = []

    for i, (logistic_model_name, predictors) in enumerate(
        zip(logistic_model_names, REGRESSION_PREDICTORS), 1
    ):
        result_model = train_model(
            df=binary_train_df,  # Pass training df (for feature type detection)
            predictors=predictors,
            results=logistic_results,
            model_name=logistic_model_name,
            model_type="logistic",
            is_log_target=False,
            is_binary=True,
            train_df=binary_train_df,  # Pre-split training data
            test_df=binary_test_df,    # Pre-split test data
        )
        logistic_errors.append(result_model["10foldCV"]["val_primary_metric"])
        graph_model(result_model, logistic_model_name, model_number=i)

    # LINEAR REGRESSION MODELS on log-transformed target
    print("\n" + "=" * 70)
    print("TRAINING LINEAR REGRESSION MODELS (Log-Transformed Target)")
    print("=" * 70)
    linear_results = {}
    linear_errors = []

    for i, (log_model_name, predictors) in enumerate(
        zip(log_model_names, REGRESSION_PREDICTORS), 1
    ):
        result_model = train_model(
            df=log_train_df,  # Pass training df (for feature type detection)
            predictors=predictors,
            results=linear_results,
            model_name=log_model_name,
            model_type="linear",
            is_log_target=True,
            is_binary=False,
            train_df=log_train_df,  # Pre-split training data
            test_df=log_test_df,    # Pre-split test data
        )
        linear_errors.append(result_model["10foldCV"]["val_primary_metric"])
        graph_model(result_model, log_model_name, model_number=i)

    # ========================================================================
    # SELECT BEST MODEL FOR EACH APPROACH
    # ========================================================================
    # This compares ALL models (1-7) including Model 6 vs Model 7.
    # The best model is selected based on 10-fold CV performance.
    # For Poisson: lowest Poisson Deviance
    # For Logistic: highest ROC-AUC
    # For Linear: lowest MSE
    #
    # NOTE: Model 6 (kitchen_sink_all) and Model 7 (backward_selected) are
    # both included in this comparison. The best one will be selected automatically.
    #
    best_poisson = use_best_model(
        results=poisson_results,
        df=train_df,  # Pass training data (for feature type detection)
        err_k10=poisson_errors,
        predictors=REGRESSION_PREDICTORS,  # Contains all 7 models including Model 6 and Model 7
        model_type="poisson",
        is_log_target=False,
        is_binary=False,
        train_df=train_df,  # Pre-split training data
        test_df=test_df,    # Pre-split test data
    )

    best_logistic = use_best_model(
        results=logistic_results,
        df=binary_train_df,  # Pass training data (for feature type detection)
        err_k10=logistic_errors,
        predictors=REGRESSION_PREDICTORS,
        model_type="logistic",
        is_log_target=False,
        is_binary=True,
        train_df=binary_train_df,  # Pre-split training data
        test_df=binary_test_df,    # Pre-split test data
    )

    best_linear = use_best_model(
        results=linear_results,
        df=log_train_df,  # Pass training data (for feature type detection)
        err_k10=linear_errors,
        predictors=REGRESSION_PREDICTORS,
        model_type="linear",
        is_log_target=True,
        is_binary=False,
        train_df=log_train_df,  # Pre-split training data
        test_df=log_test_df,    # Pre-split test data
    )

    # Get the best model names
    best_poisson_idx = best_poisson["best_model_idx"]
    best_poisson_name = model_names[best_poisson_idx]

    best_logistic_idx = best_logistic["best_model_idx"]
    best_logistic_name = logistic_model_names[best_logistic_idx]

    best_log_idx = best_linear["best_model_idx"]
    best_log_name = log_model_names[best_log_idx]

    # Graph the best models with special labeling
    graph_model(best_poisson, best_poisson_name, is_best=True)
    graph_model(best_logistic, best_logistic_name, is_best=True)
    graph_model(best_linear, best_log_name, is_best=True)

    print("\n" + "=" * 70)
    print("✅ ALL DONE!")
    print("=" * 70)
    print(f"Best Poisson model: {best_poisson_name} (Model {best_poisson_idx + 1})")
    print(f"Best Logistic model: {best_logistic_name} (Model {best_logistic_idx + 1})")
    print(f"Best Linear (log) model: {best_log_name} (Model {best_log_idx + 1})")
    print(f"Check the 'plots/' directory for all visualizations")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
