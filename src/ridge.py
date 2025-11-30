import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from load import load_data
from clean import clean_data
from model import train_model
from graph import graph_model

#loading data
curr_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(
    curr_dir,
    "../datasets/df_master_schedule_injury_surface_2019_23_weather_distance_days_since_last_game.csv",
)

df = load_data(data_dir)
df = clean_data(df)
    
target_column = "num_injuries"
predictors = [
    'surface_type', 'Avg_Temp', 'day', 'week', 'season', 'stadium', 'surface', 'dome', 'num_plays',
    'yds_w', 'yds_l', 'tov_w', 'tov_l'
]

#Ridge
results = {}
ridge_results = train_model(
    df = df,
    predictors = predictors,
    results = results,
    model_name = 'ridge_regression',
    model_type = 'ridge',
    is_log_target = False,
)

#graphing
ridge_result_model = {
    "10foldCV": {
        "val_primary_metric": ridge_results['10foldCV']['val_primary_metric'],
        "val_mse": ridge_results['10foldCV']['val_mse'],
        "val_mae": ridge_results['10foldCV']['val_mae'],
        "test_mse": ridge_results['10foldCV']['test_mse'],
        "test_mae": ridge_results['10foldCV']['test_mae'],
    },
    "y_test": ridge_results['y_test'],
    "y_pred": ridge_results['y_pred'],
    "df": df,
    "predictors": predictors,
    "model_type": "ridge",
    "is_log_target": False,
}

# Metrics
y_true = ridge_results['y_test']
y_pred = ridge_results['y_pred']
n = len(y_true)
p = len(predictors)

# R^2 and Adjusted R^2
from sklearn.metrics import r2_score
r2 = r2_score(y_true, y_pred)
r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)

# AIC and BIC
rss = np.sum((y_true - y_pred)**2)
aic = n * np.log(rss/n) + 2 * p
bic = n * np.log(rss/n) + p * np.log(n)

print(f"R^2: {r2:.4f} | Adjusted R^2: {r2_adj:.4f}")
print(f"AIC: {aic:.2f} | BIC: {bic:.2f}")


graph_model(ridge_result_model, "ridge_model", model_number=1)