"""
Neural Network Model Visualization Functions
Saves plots to plots/neural_net/ directory
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


def get_neural_net_report_path(model_name: str, is_best: bool = False):
    """Generate path for text report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    base_dir = Path("plots") / "neural_net"
    base_dir.mkdir(parents=True, exist_ok=True)
    
    prefix = f"{model_name}_best_" if is_best else f"{model_name}_"
    filename = f"{prefix}{timestamp}_report.txt"
    
    return base_dir / filename


def get_neural_net_plot_path(model_name: str, plot_type: str, is_best: bool = False):
    """
    Generate plot path for Neural Network models
    
    Args:
        model_name: Name of the model (e.g., "nn_count")
        plot_type: Type of plot (e.g., "metrics", "scatter", "residuals")
        is_best: Whether this is the best model
    
    Returns:
        Path object for the plot file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Base directory
    base_dir = Path("plots") / "neural_net"
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


def graph_neural_net_model(
    model_results: dict,
    model_name: str,
    is_best: bool = False,
):
    """
    Generate comprehensive visualizations for Neural Network model results
    
    Parameters:
    -----------
    model_results : dict
        Results dictionary from train_final_model
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
    cv_results = model_results["10foldCV"]
    final_results = model_results["final_test"]
    predictions = model_results["predictions"]
    
    y_test = np.array(predictions["y_test_actual"])
    y_pred = np.array(predictions["y_test_pred"])
    
    # Get hyperparameters from model
    model = model_results["model"]
    hidden_layer_sizes = model.hidden_layer_sizes
    learning_rate_init = model.learning_rate_init
    alpha = model.alpha
    batch_size = model.batch_size
    
    actual_mean = y_test.mean()
    pred_mean = y_pred.mean()
    
    # ========== 1. METRICS BAR CHART ==========
    fig1, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    metrics = ["r2", "mse", "mae"]
    titles = ["R²", "MSE", "MAE"]
    
    for ax, metric, title in zip(axes, metrics, titles):
        if metric == "r2":
            cv_val = cv_results.get(f"val_{metric}", cv_results.get("val_r2", 0))
            test_val = final_results.get(f"test_{metric}", final_results.get("test_r2", 0))
        elif metric == "mse":
            cv_val = cv_results.get(f"val_{metric}", cv_results.get("val_mse", 0))
            test_val = final_results.get(f"test_{metric}", final_results.get("test_mse", 0))
        else:  # mae
            # Calculate MAE from predictions if not stored
            cv_val = cv_results.get(f"val_{metric}", 0)
            test_val = final_results.get(f"test_{metric}", mean_absolute_error(y_test, y_pred))
        
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
    
    hidden_layers_str = str(hidden_layer_sizes) if isinstance(hidden_layer_sizes, tuple) else f"({hidden_layer_sizes},)"
    model_type_label = "Neural Network (Log-Transformed Target)" if model_results.get("is_log_target", False) else "Neural Network (Count Data)"
    plt.suptitle(
        f"{title_prefix}Neural Network Model Performance: {display_name} {model_type_label}\n"
        f"Architecture: {hidden_layers_str}, lr={learning_rate_init}, alpha={alpha}, batch_size={batch_size}",
        fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    
    filename1 = get_neural_net_plot_path(model_name, "metrics", is_best)
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
        f"Architecture = {hidden_layers_str}\n"
        f"Learning Rate = {learning_rate_init}\n"
        f"Alpha (L2) = {alpha}\n"
        f"Batch Size = {batch_size}\n"
        f"Iterations = {model.n_iter_}\n"
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
    filename2 = get_neural_net_plot_path(model_name, "scatter", is_best)
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
    
    filename3 = get_neural_net_plot_path(model_name, "residuals", is_best)
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
    filename4 = get_neural_net_plot_path(model_name, "error_by_count", is_best)
    plt.savefig(filename4, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {filename4}")
    plt.close()
    
    # ========== 5. GENERATE TEXT REPORT ==========
    report = generate_neural_net_report(
        display_name,
        model_results,
        y_test,
        y_pred,
        actual_mean,
        pred_mean,
        is_best,
        hidden_layer_sizes,
        learning_rate_init,
        alpha,
        batch_size,
    )
    
    report_file = get_neural_net_report_path(model_name, is_best)
    with open(report_file, "w") as f:
        f.write(report)
    
    print(f"\n{'='*70}")
    print(f"All visualizations saved to: plots/neural_net/")
    print(f"✓ Report saved: {report_file}")
    print(f"{'='*70}\n")


def generate_neural_net_report(
    display_name: str,
    model_results: dict,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    actual_mean: float,
    pred_mean: float,
    is_best: bool = False,
    hidden_layer_sizes: tuple = (100, 50),
    learning_rate_init: float = 0.001,
    alpha: float = 0.0001,
    batch_size: int = 'auto',
):
    """Generate detailed text report for Neural Network models"""
    best_marker = "⭐ BEST MODEL ⭐" if is_best else ""
    
    cv_results = model_results["10foldCV"]
    final_results = model_results["final_test"]
    model = model_results["model"]
    is_log_target = model_results.get("is_log_target", False)
    
    model_type_label = "Neural Network (Log-Transformed Target)" if is_log_target else "Neural Network (Count Data)"
    
    # Calculate R²
    r2 = r2_score(y_test, y_pred)
    
    # Calculate adjusted R²
    n = len(y_test)
    # Estimate number of parameters (rough approximation)
    total_params = sum(w.size for w in model.coefs_) + sum(b.size for b in model.intercepts_)
    adj_r2 = calculate_adjusted_r2(r2, n, total_params)
    
    # Calculate other metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    pred_range = y_pred.max() - y_pred.min()
    actual_range = y_test.max() - y_test.min()
    
    hidden_layers_str = str(hidden_layer_sizes) if isinstance(hidden_layer_sizes, tuple) else f"({hidden_layer_sizes},)"
    
    report = f"""
{'='*70}
{best_marker}
MODEL REPORT: {display_name}
MODEL TYPE: {model_type_label}
{'='*70}
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

HYPERPARAMETERS:
  • Architecture (hidden_layer_sizes): {hidden_layers_str}
  • Learning rate (learning_rate_init): {learning_rate_init}
  • Alpha (L2 regularization): {alpha}
  • Batch size: {batch_size}
  • Activation: {model.activation}
  • Solver: {model.solver}
  • Iterations completed: {model.n_iter_}
  • Total parameters: {total_params:,}

CROSS-VALIDATION RESULTS (10-fold):
  • Training R²:       {cv_results['train_r2']:.4f}
  • Validation R²:    {cv_results['val_r2']:.4f}
  • Training RMSE:    {np.sqrt(cv_results['train_mse']):.4f}
  • Validation RMSE:  {np.sqrt(cv_results['val_mse']):.4f}
  • Training MSE:     {cv_results['train_mse']:.4f}
  • Validation MSE:   {cv_results['val_mse']:.4f}

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
    mse_ratio = final_results["test_mse"] / cv_results["val_mse"] if cv_results["val_mse"] > 0 else np.nan
    # Calculate CV MAE if not available
    cv_mae = cv_results.get("val_mae", np.sqrt(cv_results["val_mse"] * 2 / np.pi))  # Approximate MAE from RMSE
    mae_ratio = mae / cv_mae if cv_mae > 0 else np.nan
    
    report += f"""  • MSE:               {mse_ratio:.4f}
  • MAE:               {mae_ratio:.4f}
  
  Note: Ratio > 1.1 suggests overfitting
"""
    
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
   - Consider: different features, different architecture, or more training data
"""
    
    if mse_ratio > 1.2 or mae_ratio > 1.2:
        report += f"""
⚠️  WARNING: Potential overfitting detected
   - Test error is significantly higher than CV error
   - MSE ratio: {mse_ratio:.2f}, MAE ratio: {mae_ratio:.2f}
   - Consider: more regularization (higher alpha), simpler architecture, or more training data
"""
    
    if is_best:
        primary_metric = "MSE"
        report += f"""
{'='*70}
🎯 This is the BEST Neural Network model based on 10-fold CV {primary_metric}
   - Retrained on full 80/20 train/test split
   - Use this model for final predictions
{'='*70}
"""
    
    report += f"\n{'='*70}\n"
    return report

