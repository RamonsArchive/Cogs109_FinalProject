"""
Gradient Boosting Model Visualization Functions
Saves plots to plots/boosting/ directory
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)

# Set style
sns.set_style("whitegrid")


def get_boosting_report_path(model_name: str, is_best: bool = False):
    """Generate path for text report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    base_dir = Path("plots") / "boosting"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    prefix = f"{model_name}_best_" if is_best else f"{model_name}_"
    filename = f"{prefix}{timestamp}_report.txt"
    
    return base_dir / filename


def get_boosting_plot_path(model_name: str, plot_type: str, is_best: bool = False):
    """
    Generate plot path for Boosting models
    
    Args:
        model_name: Name of the model (e.g., "boosting_count")
        plot_type: Type of plot (e.g., "metrics", "scatter", "residuals", "feature_importance")
        is_best: Whether this is the best model
    
    Returns:
        Path object for the plot file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Base directory
    base_dir = Path("plots") / "boosting"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename
    prefix = f"{model_name}_best_" if is_best else f"{model_name}_"
    filename = f"{prefix}{timestamp}_{plot_type}.png"
    
    return base_dir / filename


def calculate_adjusted_r2(r2: float, n: int, p: int) -> float:
    """Calculate adjusted R²"""
    if n <= p + 1:
        return np.nan
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def graph_boosting_model(
    model_results: dict,
    model_name: str,
    is_best: bool = False,
):
    """
    Generate comprehensive visualizations for Gradient Boosting model results
    
    Parameters:
    -----------
    model_results : dict
        Results dictionary from train_boosting_model
    model_name : str
        Name for this model run
    is_best : bool
        If True, label as "best"
    """
    title_prefix = "⭐ BEST MODEL - " if is_best else ""
    display_name = f"{model_name} (BEST)" if is_best else model_name
    
    print(f"\n{'='*70}")
    print(f"Graphing {display_name}")
    print(f"{'='*70}")
    
    # Extract data from results
    results = model_results["10foldCV"]
    y_test = np.array(model_results["y_test"])
    y_pred = np.array(model_results["y_pred"])
    df = model_results["df"]
    predictors = model_results["predictors"]
    is_log_target = model_results.get("is_log_target", False)
    y_test_original = model_results.get("y_test_original", y_test)
    n_estimators = model_results.get("n_estimators", 200)
    learning_rate = model_results.get("learning_rate", 0.1)
    max_depth = model_results.get("max_depth", 3)
    feature_importances = model_results.get("feature_importances", None)
    feature_names = model_results.get("feature_names", None)
    
    actual_mean = y_test.mean()
    pred_mean = y_pred.mean()
    
    # ========== 1. METRICS BAR CHART ==========
    fig1, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    metrics = ["r2", "mse", "mae"]
    titles = ["R²", "MSE", "MAE"]
    
    for ax, metric, title in zip(axes, metrics, titles):
        if metric == "r2":
            cv_val = results.get(f"val_{metric}", 0)
            test_val = results.get(f"test_{metric}", 0)
        else:
            cv_val = results.get(f"val_{metric}", 0)
            test_val = results.get(f"test_{metric}", 0)
        
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
    
    model_type_label = "Gradient Boosting (Log-Transformed Target)" if is_log_target else "Gradient Boosting (Count Data)"
    plt.suptitle(
        f"{title_prefix}Gradient Boosting Model Performance: {display_name} {model_type_label}\n"
        f"n_estimators={n_estimators}, learning_rate={learning_rate}, max_depth={max_depth}",
        fontsize=16, fontweight="bold"
    )
    plt.tight_layout()
    
    filename1 = get_boosting_plot_path(model_name, "metrics", is_best)
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
        f"{title_prefix}Actual vs Predicted: {display_name}",
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
        f"n_estimators = {n_estimators}\n"
        f"learning_rate = {learning_rate}\n"
        f"max_depth = {max_depth}\n"
        f"Mean Pred = {pred_mean:.2f}\n"
        f"Mean Actual = {actual_mean:.2f}"
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    ax.text(
        0.05, 0.95, textstr,
        transform=ax.transAxes, fontsize=11,
        verticalalignment="top", bbox=props, fontweight="bold"
    )
    
    plt.tight_layout()
    filename2 = get_boosting_plot_path(model_name, "scatter", is_best)
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
    from scipy import stats
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
        f"{title_prefix}Residual Analysis: {display_name}",
        fontsize=15, fontweight="bold"
    )
    plt.tight_layout()
    
    filename3 = get_boosting_plot_path(model_name, "residuals", is_best)
    plt.savefig(filename3, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {filename3}")
    plt.close()
    
    # ========== 4. FEATURE IMPORTANCE ==========
    if feature_importances is not None and feature_names is not None:
        # Get top 20 features by importance
        importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": feature_importances
        })
        importance_df = importance_df.sort_values("importance", ascending=False).head(20)
        
        fig4, ax = plt.subplots(figsize=(12, 8))
        
        bars = ax.barh(
            range(len(importance_df)), importance_df["importance"],
            color="#3498db", alpha=0.7, edgecolor="black"
        )
        
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df["feature"], fontsize=9)
        ax.set_xlabel("Feature Importance", fontsize=12)
        ax.set_title(
            f"{title_prefix}Top 20 Feature Importances: {display_name}",
            fontsize=14, fontweight="bold"
        )
        ax.grid(True, alpha=0.3, axis="x")
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, importance_df["importance"])):
            width = bar.get_width()
            ax.text(
                width, i, f"{val:.4f}",
                ha="left", va="center", fontsize=8
            )
        
        plt.tight_layout()
        filename4 = get_boosting_plot_path(model_name, "feature_importance", is_best)
        plt.savefig(filename4, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {filename4}")
        plt.close()
    
    # ========== 5. PREDICTION ERROR BY ACTUAL COUNT ==========
    fig5, ax = plt.subplots(figsize=(12, 7))
    
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
        f"{title_prefix}Prediction Error by Actual Injury Count: {display_name}",
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
    filename5 = get_boosting_plot_path(model_name, "error_by_count", is_best)
    plt.savefig(filename5, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {filename5}")
    plt.close()
    
    # ========== 7. GENERATE TEXT REPORT ==========
    report = generate_boosting_report(
        display_name,
        model_results,
        y_test,
        y_pred,
        actual_mean,
        pred_mean,
        is_best,
        n_estimators,
        learning_rate,
        max_depth,
    )
    
    report_file = get_boosting_report_path(model_name, is_best)
    with open(report_file, "w") as f:
        f.write(report)
    
    print(f"\n{'='*70}")
    print(f"All visualizations saved to: plots/boosting/")
    print(f"✓ Report saved: {report_file}")
    print(f"{'='*70}\n")


def generate_boosting_report(
    display_name: str,
    model_results: dict,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    actual_mean: float,
    pred_mean: float,
    is_best: bool = False,
    n_estimators: int = 200,
    learning_rate: float = 0.1,
    max_depth: int = 3,
):
    """Generate detailed text report for Boosting models"""
    best_marker = "⭐ BEST MODEL ⭐" if is_best else ""
    
    results = model_results["10foldCV"]
    predictors = model_results["predictors"]
    df = model_results["df"]
    is_log_target = model_results.get("is_log_target", False)
    y_test_original = model_results.get("y_test_original", y_test)
    
    model_type_label = "Gradient Boosting (Log-Transformed Target)" if is_log_target else "Gradient Boosting (Count Data)"
    
    # Calculate R²
    r2 = r2_score(y_test, y_pred)
    
    # Calculate adjusted R²
    n = len(y_test)
    p = len(predictors)
    adj_r2 = calculate_adjusted_r2(r2, n, p)
    
    # Calculate other metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    pred_range = y_pred.max() - y_pred.min()
    actual_range = y_test.max() - y_test.min()
    
    report = f"""
{'='*70}
{best_marker}
MODEL REPORT: {display_name}
MODEL TYPE: {model_type_label}
{'='*70}
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

HYPERPARAMETERS:
  • n_estimators (B trees):     {n_estimators}
  • learning_rate (shrinkage):  {learning_rate}
  • max_depth (splits):         {max_depth}

PREDICTORS ({len(predictors)}):
{', '.join(predictors)}

CROSS-VALIDATION RESULTS (10-fold):
  • R²:                {results['val_r2']:.4f}
  • MSE:               {results['val_mse']:.4f}
  • RMSE:              {np.sqrt(results['val_mse']):.4f}
  • MAE:               {results['val_mae']:.4f}
"""
    
    # Add original scale CV metrics for log-transformed models
    if is_log_target and "val_mse_original" in results:
        report += f"""
CROSS-VALIDATION RESULTS (10-fold) - ORIGINAL SCALE:
  • MSE:               {results['val_mse_original']:.4f}
  • RMSE:              {results['val_rmse_original']:.4f}
  • MAE:               {results['val_mae_original']:.4f}
"""
    
    report += f"""
TEST SET RESULTS:
  • R²:                {r2:.4f}
"""
    
    if not np.isnan(adj_r2):
        report += f"  • Adjusted R²:       {adj_r2:.4f}\n"
    
    report += f"""  • MSE:               {mse:.4f}
  • RMSE:              {rmse:.4f}
  • MAE:               {mae:.4f}

GENERALIZATION (Test/CV ratio):
"""
    
    # Generalization metrics
    mse_ratio = results["test_mse"] / results["val_mse"] if results["val_mse"] > 0 else np.nan
    mae_ratio = results["test_mae"] / results["val_mae"] if results["val_mae"] > 0 else np.nan
    
    report += f"""  • MSE:               {mse_ratio:.4f}
  • MAE:               {mae_ratio:.4f}
  
  Note: Ratio > 1.1 suggests overfitting
"""
    
    # Original scale metrics for log-transformed models
    if is_log_target:
        # Predictions are already in original scale (expm1 was applied)
        # But we need to show both log-scale and original-scale metrics
        y_test_log = np.log1p(y_test_original)
        y_pred_log = np.log1p(y_pred)
        
        original_pred_mean = y_pred.mean()
        original_mse = mean_squared_error(y_test_original, y_pred)
        original_rmse = np.sqrt(original_mse)
        original_mae = mean_absolute_error(y_test_original, y_pred)
        original_r2 = r2_score(y_test_original, y_pred)
        original_adj_r2 = calculate_adjusted_r2(original_r2, n, p)
        
        # Log-scale metrics
        log_mse = mean_squared_error(y_test_log, y_pred_log)
        log_rmse = np.sqrt(log_mse)
        log_r2 = r2_score(y_test_log, y_pred_log)
        
        report += f"""
LOG-SCALE METRICS (before expm1 transformation):
  • R²:                {log_r2:.4f}
  • RMSE:              {log_rmse:.4f}
  • MSE:               {log_mse:.4f}

ORIGINAL SCALE METRICS (after expm1 transformation):
  • Mean:              {original_pred_mean:.2f} injuries
  • MSE:               {original_mse:.4f} injuries
  • RMSE:              {original_rmse:.4f} injuries
  • MAE:               {original_mae:.4f} injuries
  • R²:                {original_r2:.4f}
"""
        if not np.isnan(original_adj_r2):
            report += f"  • Adjusted R²:       {original_adj_r2:.4f}\n"
        
        report += f"  • Pred Range:        [{y_pred.min():.2f}, {y_pred.max():.2f}]\n"
    
    # Additional metrics
    report += f"""
ADDITIONAL METRICS:
  • Mean Actual:       {actual_mean:.2f}
  • Mean Predicted:    {pred_mean:.2f}
  • Actual Range:      {actual_range:.2f}
  • Predicted Range:   {pred_range:.2f}
"""
    
    report += f"""
MODEL INTERPRETATION:
  • R² = {r2:.1%} means the model explains {r2:.1%} of variance
"""
    if not np.isnan(adj_r2):
        report += f"  • Adjusted R² = {adj_r2:.1%} accounts for model complexity\n"
    
    report += f"""  • Predictions range from {y_pred.min():.2f} to {y_pred.max():.2f}
  • Actual values range from {y_test.min():.2f} to {y_test.max():.2f}
"""
    
    # Performance warnings
    if r2 < 0.1:
        report += f"""
⚠️  WARNING: Very low R² ({r2:.4f})
   - Model barely outperforms predicting the mean
   - Consider: different features, non-linear models, or interaction terms
"""
    
    if mse_ratio > 1.2 or mae_ratio > 1.2:
        report += f"""
⚠️  WARNING: Potential overfitting detected
   - Test error is significantly higher than CV error
   - MSE ratio: {mse_ratio:.2f}, MAE ratio: {mae_ratio:.2f}
   - Consider: more regularization, simpler model, or more training data
"""
    
    if is_best:
        primary_metric = "MSE"
        report += f"""
{'='*70}
🎯 This is the BEST Boosting model based on 10-fold CV {primary_metric}
   - Retrained on full 80/20 train/test split
   - Use this model for final predictions
{'='*70}
"""
    
    report += f"\n{'='*70}\n"
    return report

