import os
from load import load_data
from clean import clean_data
from graph import graph_model, generate_text_report, calculate_adjusted_r2
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score



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

# Get target column
y = df[target_column].values
X = df[predictors]

# CRITICAL: Split into train and test sets FIRST (before any preprocessing)
# This prevents data leakage - all preprocessing will happen inside the pipeline
# and will only be fit on training data during cross-validation and final training
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Loop over polynomial degrees
for i, degree in enumerate(poly_degrees, 1):
    model_name = f"poly_deg_{degree}"

    # Create pipeline with proper preprocessing to prevent data leakage
    # IMPORTANT: All preprocessing happens INSIDE the pipeline, so it's fit only on training data
    # This prevents data leakage that occurred when preprocessing was done before train/test split
    # Order: StandardScaler -> PolynomialFeatures (for numeric), OneHotEncoder (for categorical)
    numeric_transformer = Pipeline([
        ('scaler', StandardScaler()),
        ('poly', PolynomialFeatures(degree=degree, include_bias=False))
    ])
    
    # Build ColumnTransformer
    # Note: Using drop='first' to avoid dummy variable trap (better for linear regression)
    # This matches sklearn's default behavior and prevents multicollinearity
    if cat_cols:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_cols)
            ]
        )
    else:
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, num_cols)
            ]
        )
    
    # Full pipeline: preprocessor -> linear regression
    # cross_validate will fit this pipeline on each CV fold's training data only
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LinearRegression())
    ])

    # 10-fold cross-validation on training set
    cv = KFold(n_splits=10, shuffle=True, random_state=42)
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
    rmse = np.sqrt(mean_mse)
    poly_errors.append(rmse)

    print(f"[Degree {degree}] CV RMSE: {rmse:.4f}   (MSE: {mean_mse:.4f})")
    print(f"[Degree {degree}] CV MAE: {mean_mae:.4f}")
    print(f"[Degree {degree}] CV R²: {mean_r2:.4f}")
    
    # Fit on full training set and predict on test set
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Get actual number of features after transformation (for adjusted R²)
    # Access the fitted preprocessor from the pipeline
    X_train_transformed = pipeline.named_steps['preprocessor'].transform(X_train)
    n_features = X_train_transformed.shape[1]

    # Compute test metrics
    test_mse = mean_squared_error(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_r2 = r2_score(y_test, y_pred)
    
    print(f"[Degree {degree}] Test RMSE: {np.sqrt(test_mse):.4f}   (MSE: {test_mse:.4f})")
    print(f"[Degree {degree}] Test MAE: {test_mae:.4f}")
    print(f"[Degree {degree}] Test R²: {test_r2:.4f}")

    result_model = {
        "10foldCV": {          
            "val_primary_metric": mean_mse, 
            "val_mse": mean_mse,
            "val_mae": mean_mae,
            "val_r2": mean_r2,
            "test_mse": test_mse,
            "test_mae": test_mae,
        },
        "y_test": y_test,
        "y_pred": y_pred,
        "df": df,
        "predictors": predictors,
        "model_type": "linear",
        "is_log_target": False,
        "is_binary": False,
        "degree": degree,
    }

    graph_model(result_model, model_name, model_number=i)
    
    # Generate text report
    actual_mean = y_test.mean()
    pred_mean = y_pred.mean()
    
    # Calculate adjusted R² with correct number of features
    test_r2 = r2_score(y_test, y_pred)
    n_samples = len(y_test)
    adj_r2 = calculate_adjusted_r2(test_r2, n_samples, n_features)
    
    # Generate base report
    report = generate_text_report(
        f"Polynomial Degree {degree}",
        result_model["10foldCV"],
        predictors,  # Original predictors list (for display)
        y_test,
        y_pred,
        actual_mean,
        pred_mean,
        is_best=False,
        model_type="linear",
        is_log_target=False,
    )
    
    # Modify report to add polynomial-specific info and fix adjusted R²
    report_lines = report.split('\n')
    
    # Add polynomial transformation info after PREDICTORS section
    for i, line in enumerate(report_lines):
        if line.startswith("PREDICTORS ("):
            # Add polynomial transformation info after the predictor list
            # Find where the predictor list ends (next section starts)
            j = i + 1
            while j < len(report_lines) and not report_lines[j].startswith("CROSS-VALIDATION"):
                j += 1
            report_lines.insert(j, f"  • Polynomial Degree: {degree}")
            report_lines.insert(j+1, f"  • Actual Features After Transformation: {n_features}")
            report_lines.insert(j+2, f"  • Polynomial applied to numeric predictors only")
            break
    
    # Update ALL adjusted R² calculations in the report
    # There are TWO places: "TEST SET RESULTS" and "ADDITIONAL METRICS"
    # Both need to use the correct number of features (n_features) not len(predictors)
    # The generate_text_report function uses len(predictors)=13, but we have 78 actual features
    for i, line in enumerate(report_lines):
        if "Adjusted R²:" in line and "accounts for" not in line:
            # Replace with correct adjusted R² using actual feature count
            if not np.isnan(adj_r2):
                report_lines[i] = f"  • Adjusted R²:       {adj_r2:.4f} (using {n_features} features after polynomial transformation)"
    
    # Also update the MODEL INTERPRETATION section that mentions Adjusted R²
    for i, line in enumerate(report_lines):
        if "Adjusted R² = " in line and "accounts for" in line:
            if not np.isnan(adj_r2):
                # Extract the percentage format
                adj_r2_pct = adj_r2 * 100
                report_lines[i] = f"  • Adjusted R² = {adj_r2_pct:.1f}% accounts for model complexity (using {n_features} features)"
    
    report = '\n'.join(report_lines)
    
    # Save report
    report_dir = Path("plots") / "polynomial"
    report_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = report_dir / f"poly_deg_{degree}_{timestamp}_report.txt"
    with open(report_file, "w") as f:
        f.write(report)
    
    print(f"✓ Report saved: {report_file}\n")

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