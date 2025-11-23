import os
from load import load_data
from clean import clean_data
from graph import graph_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error



#loading data
curr_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(
    curr_dir,
    "../datasets/df_master_schedule_injury_surface_2019_23_weather_distance_days_since_last_game.csv",
)

df = load_data(data_dir)
df = clean_data(df)
    
target_column = "num_injuries"
        
#Polynomial regression
print("\n" + "=" * 70)
print("TRAINING POLYNOMIAL REGRESSION MODELS")
print("=" * 70)

poly_degrees = [1,2,3,4]
poly_errors = []
predictors = ["surface_type", "Avg_Temp", "day", "week", "season","stadium",
        "surface", "dome", "num_plays", "yds_w", "yds_l", "tov_w","tov_l"
        ] #model 3
    
# Split numeric and categorical predictors
num_cols = df[predictors].select_dtypes(include=np.number).columns.tolist()
cat_cols = [c for c in predictors if c not in num_cols]

# Loop over polynomial degrees
for i, degree in enumerate(poly_degrees, 1):
    model_name = f"poly_deg_{degree}"

    # Only apply polynomial to numeric predictors
    X_num = df[num_cols]
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X_num)

    # Include categorical variables as-is
    if cat_cols:
        X_cat = pd.get_dummies(df[cat_cols], drop_first=True)
        X = np.hstack([X_poly, X_cat.values])
    else:
        X = X_poly

    y = df[target_column].values

    # 10-fold cross-validation
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
    mse_scores = cross_val_score(LinearRegression(), X, y, cv=cv, scoring="neg_mean_squared_error")
    rmse = np.sqrt(-mse_scores.mean())
    poly_errors.append(rmse)

    print(f"[Degree {degree}] CV RMSE: {rmse:.4f}")
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)

    # Compute basic metrics
mse = mean_squared_error(y, y_pred)
mae = mean_absolute_error(y, y_pred)

result_model = {
    "10foldCV": {          
        "val_primary_metric": mse, 
        "val_mse": mse,
        "val_mae": mae,
        "test_mse": mse,
        "test_mae": mae,
    },
    "y_test": y,
    "y_pred": y_pred,
    "df": df,
    "predictors": predictors,
    "model_type": "linear",
    "is_log_target": False,
}

graph_model(result_model, model_name, model_number=i)

#plotting CV MSE vs Degrees (hw 6)
plt.figure(figsize=(8, 5))
plt.plot(poly_degrees, poly_errors, marker='o')
plt.xlabel("Polynomial Degree")
plt.ylabel("10-Fold CV Error (MSE)")
plt.title("Polynomial Degree vs Cross-Validated MSE")
plt.grid(True)
plots_dir = os.path.join(curr_dir, "plots")
os.makedirs(plots_dir, exist_ok=True)
plt.savefig(os.path.join(plots_dir, "poly_degree_vs_cv_error.png"))
plt.close()