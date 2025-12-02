import os
from load import load_data
from clean import clean_data
from graph import generate_text_report, calculate_adjusted_r2
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split, cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats

# Set style
sns.set_style("whitegrid")

# Constants
RANDOM_STATE = 42

# ========================================================================
# GRAPHING FUNCTIONS FOR POLYNOMIAL MODELS
# ========================================================================

def get_polynomial_plot_path(degree: int, plot_type: str):
    """Generate plot path for polynomial models"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path("plots") / "polynomial"
    base_dir.mkdir(parents=True, exist_ok=True)
    filename = f"poly_deg_{degree}_{timestamp}_{plot_type}.png"
    return base_dir / filename

def get_polynomial_report_path(degree: int):
    """Generate path for text report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path("plots") / "polynomial"
    base_dir.mkdir(parents=True, exist_ok=True)
    filename = f"poly_deg_{degree}_{timestamp}_report.txt"
    return base_dir / filename

def graph_polynomial_model(
    degree: int,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    cv_results: dict,
    test_results: dict,
    n_features: int,
):
    """Generate comprehensive visualizations for polynomial regression models"""
    
    print(f"\n{'='*70}")
    print(f"Graphing Polynomial Degree {degree} Model")
    print(f"{'='*70}")
    
    actual_mean = y_test.mean()
    pred_mean = y_pred.mean()
    
    # ========== 1. METRICS BAR CHART ==========
    fig1, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    metrics = ["r2", "mse", "mae"]
    titles = ["R²", "MSE", "MAE"]
    cv_vals = [cv_results["val_r2"], cv_results["val_mse"], cv_results["val_mae"]]
    test_vals = [test_results["test_r2"], test_results["test_mse"], test_results["test_mae"]]
    
    for ax, metric, title, cv_val, test_val in zip(axes, metrics, titles, cv_vals, test_vals):
        x = ["CV", "Test"]
        y = [cv_val, test_val]
        colors = ["#3498db", "#e74c3c"]
        
        bars = ax.bar(x, y, color=colors, alpha=0.7, edgecolor="black", linewidth=1.5)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f"{title}", fontsize=13, fontweight="bold")
        
        # Add value labels on bars
        for bar, val in zip(bars, y):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold"
            )
        
        # Add percentage difference
        if cv_val > 0:
            pct_diff = ((test_val - cv_val) / cv_val) * 100
            diff_text = f"Test vs CV: {pct_diff:+.1f}%"
            color = "lightcoral" if pct_diff > 10 else "lightgreen"
            ax.text(
                0.5, 0.95, diff_text,
                transform=ax.transAxes, ha="center", va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor=color, alpha=0.5)
            )
    
    plt.suptitle(
        f"Polynomial Degree {degree} Model Performance\n"
        f"Features: {n_features} (after polynomial transformation)",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    
    filename1 = get_polynomial_plot_path(degree, "metrics")
    plt.savefig(filename1, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {filename1}")
    plt.close()
    
    # ========== 2. SCATTER PLOT: ACTUAL VS PREDICTED ==========
    fig2, ax = plt.subplots(figsize=(10, 10))
    
    # Create scatter with jitter for better visibility
    jitter_x = y_test + np.random.normal(0, 0.1, size=len(y_test))
    jitter_y = y_pred + np.random.normal(0, 0.05, size=len(y_pred))
    
    ax.scatter(
        jitter_x, jitter_y,
        alpha=0.5, s=60, edgecolors="black", linewidth=0.8,
        c="#3498db", label="Predictions"
    )
    
    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot(
        [min_val, max_val], [min_val, max_val],
        "r--", lw=2.5, label="Perfect Prediction", zorder=5
    )
    
    # Add mean lines
    ax.axhline(
        y=pred_mean, color="green", linestyle=":", lw=2,
        label=f"Mean Prediction: {pred_mean:.2f}", alpha=0.7
    )
    ax.axvline(
        x=actual_mean, color="orange", linestyle=":", lw=2,
        label=f"Mean Actual: {actual_mean:.2f}", alpha=0.7
    )
    
    ax.set_xlabel("Actual Injuries", fontsize=13)
    ax.set_ylabel("Predicted Injuries", fontsize=13)
    ax.set_title(
        f"Actual vs Predicted: Polynomial Degree {degree}",
        fontsize=15, fontweight="bold"
    )
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)
    
    # Calculate and display metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    textstr = (
        f"R² = {r2:.4f}\n"
        f"MAE = {mae:.4f}\n"
        f"RMSE = {rmse:.4f}\n"
        f"Degree = {degree}\n"
        f"Features = {n_features}\n"
        f"Mean Pred = {pred_mean:.2f}\n"
        f"Mean Actual = {actual_mean:.2f}"
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    ax.text(
        0.05, 0.95, textstr,
        transform=ax.transAxes, fontsize=10,
        verticalalignment="top", bbox=props, fontweight="bold"
    )
    
    plt.tight_layout()
    filename2 = get_polynomial_plot_path(degree, "scatter")
    plt.savefig(filename2, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {filename2}")
    plt.close()
    
    # ========== 3. RESIDUALS ANALYSIS ==========
    residuals = y_test - y_pred
    
    fig3, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 3a. Residuals vs Predicted
    axes[0, 0].scatter(
        y_pred, residuals,
        alpha=0.6, s=60, edgecolors="black", linewidth=0.8, c="#e74c3c"
    )
    axes[0, 0].axhline(y=0, color="blue", linestyle="--", lw=2.5, label="Zero Residual")
    axes[0, 0].set_xlabel("Predicted Injuries", fontsize=12)
    axes[0, 0].set_ylabel("Residuals (Actual - Predicted)", fontsize=12)
    axes[0, 0].set_title("Residual Plot", fontsize=13, fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Add std bands
    std_resid = np.std(residuals)
    axes[0, 0].axhline(y=2 * std_resid, color="orange", linestyle=":", lw=1.5, alpha=0.7)
    axes[0, 0].axhline(y=-2 * std_resid, color="orange", linestyle=":", lw=1.5, alpha=0.7)
    
    # 3b. Residuals histogram
    axes[0, 1].hist(
        residuals, bins=30, edgecolor="black", alpha=0.7, color="#9b59b6"
    )
    axes[0, 1].axvline(x=0, color="red", linestyle="--", lw=2.5, label="Zero")
    axes[0, 1].axvline(
        x=residuals.mean(), color="green", linestyle=":", lw=2,
        label=f"Mean: {residuals.mean():.3f}"
    )
    axes[0, 1].set_xlabel("Residuals", fontsize=12)
    axes[0, 1].set_ylabel("Frequency", fontsize=12)
    axes[0, 1].set_title("Residuals Distribution", fontsize=13, fontweight="bold")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # 3c. Q-Q plot for normality check
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title("Q-Q Plot (Normality Check)", fontsize=13, fontweight="bold")
    axes[1, 0].grid(True, alpha=0.3)
    
    # 3d. Residuals vs Actual
    axes[1, 1].scatter(
        y_test, residuals,
        alpha=0.6, s=60, edgecolors="black", linewidth=0.8, c="#e67e22"
    )
    axes[1, 1].axhline(y=0, color="blue", linestyle="--", lw=2.5)
    axes[1, 1].set_xlabel("Actual Injuries", fontsize=12)
    axes[1, 1].set_ylabel("Residuals", fontsize=12)
    axes[1, 1].set_title("Residuals vs Actual", fontsize=13, fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(
        f"Residual Analysis: Polynomial Degree {degree}",
        fontsize=15, fontweight="bold"
    )
    plt.tight_layout()
    
    filename3 = get_polynomial_plot_path(degree, "residuals")
    plt.savefig(filename3, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {filename3}")
    plt.close()
    
    # ========== 4. PREDICTION ERROR BY ACTUAL COUNT ==========
    fig4, ax = plt.subplots(figsize=(12, 7))
    
    error_by_actual = {}
    for actual in np.unique(y_test):
        mask = y_test == actual
        error_by_actual[actual] = y_pred[mask] - actual
    
    positions = list(error_by_actual.keys())
    data = [error_by_actual[k] for k in positions]
    
    bp = ax.boxplot(
        data, positions=positions, widths=0.6, patch_artist=True,
        boxprops=dict(facecolor="lightblue", alpha=0.7),
        medianprops=dict(color="red", linewidth=2)
    )
    
    ax.axhline(y=0, color="black", linestyle="--", lw=1.5, alpha=0.5)
    ax.set_xlabel("Actual Number of Injuries", fontsize=12)
    ax.set_ylabel("Prediction Error", fontsize=12)
    ax.set_title(
        f"Prediction Error by Actual Injury Count: Polynomial Degree {degree}",
        fontsize=14, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)
    
    # Add sample sizes
    for pos in positions:
        count = len(error_by_actual[pos])
        ax.text(
            pos, ax.get_ylim()[1] * 0.95, f"n={count}",
            ha="center", fontsize=9, style="italic"
        )
    
    plt.tight_layout()
    filename4 = get_polynomial_plot_path(degree, "error_by_count")
    plt.savefig(filename4, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {filename4}")
    plt.close()
    
    print(f"\n{'='*70}")
    print(f"All visualizations saved to: plots/polynomial/")
    print(f"{'='*70}\n")

# ========================================================================
# MAIN CODE
# ========================================================================

# Loading data
curr_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(
    curr_dir,
    "../datasets/df_master_schedule_injury_surface_2019_23_weather_distance_days_since_last_game.csv",
)

df = load_data(data_dir)
df = clean_data(df)

target_column = "num_injuries"

# ========================================================================
# STEP 1: ONE SPLIT - Create held-out test set (NEVER touch until end)
# ========================================================================
print("\n" + "=" * 70)
print("STEP 1: DATA SPLITTING")
print("=" * 70)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
print(f"📊 Outer split: Train={len(train_df)} ({len(train_df)/len(df)*100:.1f}%), Test={len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
print("⚠️  Test set will NOT be used for model training or selection!")
print("⚠️  Test set will ONLY be used for final unbiased evaluation")
print("=" * 70)

# Polynomial regression
print("\n" + "=" * 70)
print("TRAINING POLYNOMIAL REGRESSION MODELS")
print("=" * 70)

poly_degrees = [1, 2, 3, 4]
poly_errors = []
poly_rmse_errors = []

# Model 6: Kitchen sink (all potentially relevant features)
predictors = [
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

# Split numeric and categorical predictors (using training data for consistency)
num_cols = train_df[predictors].select_dtypes(include=np.number).columns.tolist()
cat_cols = [c for c in predictors if c not in num_cols]

# Get target column from split data
X_train = train_df[predictors]
y_train = train_df[target_column].values
X_test = test_df[predictors]
y_test = test_df[target_column].values

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
                ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
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
    cv_rmse = np.sqrt(mean_mse)
    poly_errors.append(mean_mse)  # Store MSE for comparison plot
    poly_rmse_errors.append(cv_rmse)  # Store RMSE for display

    print(f"[Degree {degree}] CV RMSE: {cv_rmse:.4f}   (MSE: {mean_mse:.4f})")
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
    test_rmse = np.sqrt(test_mse)
    
    print(f"[Degree {degree}] Test RMSE: {test_rmse:.4f}   (MSE: {test_mse:.4f})")
    print(f"[Degree {degree}] Test MAE: {test_mae:.4f}")
    print(f"[Degree {degree}] Test R²: {test_r2:.4f}")

    # Store CV and test results (merge for generate_text_report which expects both)
    results_dict = {
        "val_primary_metric": mean_mse,
        "val_mse": mean_mse,
        "val_mae": mean_mae,
        "val_r2": mean_r2,
        "test_mse": test_mse,
        "test_mae": test_mae,
        "test_r2": test_r2,
    }
    
    # Separate dicts for graphing function
    cv_results = {
        "val_primary_metric": mean_mse,
        "val_mse": mean_mse,
        "val_mae": mean_mae,
        "val_r2": mean_r2,
    }
    
    test_results = {
        "test_mse": test_mse,
        "test_mae": test_mae,
        "test_r2": test_r2,
    }
    
    # Generate comprehensive graphs
    graph_polynomial_model(degree, y_test, y_pred, cv_results, test_results, n_features)
    
    # Generate text report
    actual_mean = y_test.mean()
    pred_mean = y_pred.mean()
    
    # Calculate adjusted R² with correct number of features
    n_samples = len(y_test)
    adj_r2 = calculate_adjusted_r2(test_r2, n_samples, n_features)
    
    # Generate base report (pass merged results dict)
    report = generate_text_report(
        f"Polynomial Degree {degree}",
        results_dict,  # Contains both CV and test metrics
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
    report_file = get_polynomial_report_path(degree)
    with open(report_file, "w") as f:
        f.write(report)
    
    print(f"✓ Report saved: {report_file}\n")

# ========== 5. POLYNOMIAL DEGREE COMPARISON PLOT ==========
print("\n" + "=" * 70)
print("GENERATING POLYNOMIAL DEGREE COMPARISON PLOT")
print("=" * 70)

fig5, axes = plt.subplots(1, 2, figsize=(15, 5))

# 5a. CV MSE vs Degrees
axes[0].plot(poly_degrees, poly_errors, marker='o', linewidth=2, markersize=10, color='#3498db')
axes[0].set_xlabel("Polynomial Degree", fontsize=12, fontweight="bold")
axes[0].set_ylabel("10-Fold CV MSE", fontsize=12, fontweight="bold")
axes[0].set_title("Polynomial Degree vs Cross-Validated MSE", fontsize=13, fontweight="bold")
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(poly_degrees)

# Add value labels
for deg, mse in zip(poly_degrees, poly_errors):
    axes[0].text(deg, mse, f'{mse:.4f}', ha='center', va='bottom', fontsize=9)

# 5b. CV RMSE vs Degrees
axes[1].plot(poly_degrees, poly_rmse_errors, marker='s', linewidth=2, markersize=10, color='#e74c3c')
axes[1].set_xlabel("Polynomial Degree", fontsize=12, fontweight="bold")
axes[1].set_ylabel("10-Fold CV RMSE", fontsize=12, fontweight="bold")
axes[1].set_title("Polynomial Degree vs Cross-Validated RMSE", fontsize=13, fontweight="bold")
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(poly_degrees)

# Add value labels
for deg, rmse in zip(poly_degrees, poly_rmse_errors):
    axes[1].text(deg, rmse, f'{rmse:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()

# Save to polynomial directory
polynomial_dir = Path("plots") / "polynomial"
polynomial_dir.mkdir(parents=True, exist_ok=True)
filename5 = polynomial_dir / "poly_degree_comparison.png"
plt.savefig(filename5, dpi=300, bbox_inches="tight")
print(f"✓ Saved: {filename5}")
plt.close()

print(f"\n{'='*70}")
print("✅ ALL POLYNOMIAL MODELS COMPLETE!")
print(f"📁 All plots and reports saved to: plots/polynomial/")
print(f"{'='*70}\n")