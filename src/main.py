import os
from load import load_data
from clean import clean_data
from clean import log_transform_data
from model import train_model, use_best_model
from graph import graph_model, exploratory_data_analysis
import numpy as np
import matplotlib.pyplot as plt


def main():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(
        curr_dir,
        "../datasets/df_master_schedule_injury_surface_2019_23_weather_distance_days_since_last_game.csv",
    )
    # Define all predictor sets to test
    Poisson_Regression_Predictors = [
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

    # Load data once
    df = load_data(data_dir)
    df = clean_data(df)

    target_column = "num_injuries"
    log_df = log_transform_data(df, target_column)

    exploratory_data_analysis(df, log_df, target_column)

    # POISSON MODELS on original counts
    print("\n" + "=" * 70)
    print("TRAINING POISSON REGRESSION MODELS (Count Data)")
    print("=" * 70)
    poisson_results = {}
    poisson_errors = []

    for i, (model_name, predictors) in enumerate(
        zip(model_names, Poisson_Regression_Predictors), 1
    ):
        result_model = train_model(
            df,
            predictors,
            poisson_results,
            model_name,
            model_type="poisson",
            is_log_target=False,
        )
        poisson_errors.append(result_model["10foldCV"]["val_primary_metric"])
        graph_model(result_model, model_name, model_number=i)

    # LINEAR REGRESSION MODELS on log-transformed target
    print("\n" + "=" * 70)
    print("TRAINING LINEAR REGRESSION MODELS (Log-Transformed Target)")
    print("=" * 70)
    linear_results = {}
    linear_errors = []

    for i, (log_model_name, predictors) in enumerate(
        zip(log_model_names, Poisson_Regression_Predictors), 1
    ):
        result_model = train_model(
            log_df,
            predictors,
            linear_results,
            log_model_name,
            model_type="linear",
            is_log_target=True,
        )
        linear_errors.append(result_model["10foldCV"]["val_primary_metric"])
        graph_model(result_model, log_model_name, model_number=i)

    # Select best from each approach
    best_poisson = use_best_model(
        poisson_results,
        df,
        poisson_errors,
        Poisson_Regression_Predictors,
        model_type="poisson",
        is_log_target=False,
    )

    best_linear = use_best_model(
        linear_results,
        log_df,
        linear_errors,
        Poisson_Regression_Predictors,
        model_type="linear",
        is_log_target=True,
    )
    # Get the best model name
    best_poisson_idx = best_poisson["best_model_idx"]
    best_poisson_name = model_names[best_poisson_idx]

    best_log_idx = best_linear["best_model_idx"]
    best_log_name = log_model_names[best_log_idx]

    # Graph the best model with special labeling
    graph_model(best_poisson, best_poisson_name, is_best=True)
    graph_model(best_linear, best_log_name, is_best=True)

    print("\n" + "=" * 70)
    print("✅ ALL DONE!")
    print("=" * 70)
    print(f"Best model: {best_poisson_name} (Model {best_poisson_idx + 1})")
    print(f"Best log model: {best_log_name} (Model {best_log_idx + 1})")
    print(f"Check the 'plots/' directory for all visualizations")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
