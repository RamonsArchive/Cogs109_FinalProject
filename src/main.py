import os
from load import load_data
from clean import clean_data
from model import train_model, use_best_model
from graph import graph_model
import numpy as np


def main():
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(
        curr_dir,
        "../datasets/df_master_schedule_injury_surface_2019_23_weather_distance_days_since_last_game.csv",
    )

    results = {}

    # Define all predictor sets to test
    Poisson_Regression_Predictors = [
        # Model 1: Baseline
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
        # Model 2: Add game intensity metrics
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
        # Model 3: Add rest/travel factors
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
        # Model 4: Add weather conditions
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
        # Model 5: Kitchen sink (all potentially relevant features)
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
        "game_intensity",
        "rest_travel",
        "weather",
        "kitchen_sink",
    ]

    # Load data once
    df = load_data(data_dir)
    df = clean_data(df)

    # Track CV errors for each model
    err_k10 = []

    print("\n" + "=" * 70)
    print("PHASE 1: Training and Evaluating All Models")
    print("=" * 70)

    # Train all models with 10-fold CV
    for i, (model_name, predictors) in enumerate(
        zip(model_names, Poisson_Regression_Predictors), 1
    ):
        result_model = train_model(df, predictors, results, model_name)
        err_k10.append(
            result_model["10foldCV"]["val_poisson_dev"]
        )  # Use CV error for selection
        graph_model(result_model, model_name, model_number=i)

    # Print comparison table
    print("\n" + "=" * 70)
    print("MODEL COMPARISON (10-fold CV Poisson Deviance)")
    print("=" * 70)
    for i, (name, error) in enumerate(zip(model_names, err_k10), 1):
        marker = " ⭐ BEST" if error == min(err_k10) else ""
        print(f"Model {i} ({name:20s}): {error:.4f}{marker}")
    print("=" * 70)

    # Select and retrain best model
    print("\n" + "=" * 70)
    print("PHASE 2: Retraining Best Model on Full Split")
    print("=" * 70)

    best_model_results = use_best_model(
        results, df, err_k10, Poisson_Regression_Predictors
    )

    # Get the best model name
    best_idx = best_model_results["best_model_idx"]
    best_name = model_names[best_idx]

    # Graph the best model with special labeling
    graph_model(best_model_results, best_name, is_best=True)

    print("\n" + "=" * 70)
    print("✅ ALL DONE!")
    print("=" * 70)
    print(f"Best model: {best_name} (Model {best_idx + 1})")
    print(f"Check the 'plots/' directory for all visualizations")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
