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
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)

# Set style
sns.set_style("whitegrid")


def exploratory_data_analysis(
    df: pd.DataFrame, log_df: pd.DataFrame, target_column: str
):
    """Create side-by-side comparison of original vs log-transformed distributions"""

    print(f"\n{'='*70}")
    print("📊 EXPLORATORY DATA ANALYSIS")
    print(f"{'='*70}\n")

    # Create output directory
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create single figure with both subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Original distribution
    axes[0].hist(df[target_column], bins=50, color="blue", alpha=0.7, edgecolor="black")
    axes[0].axvline(
        df[target_column].mean(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {df[target_column].mean():.2f}",
    )
    axes[0].axvline(
        df[target_column].median(),
        color="orange",
        linestyle=":",
        linewidth=2,
        label=f"Median: {df[target_column].median():.0f}",
    )
    axes[0].set_title(
        f"Distribution of {target_column} (Original)", fontsize=14, fontweight="bold"
    )
    axes[0].set_xlabel("Number of Injuries", fontsize=12)
    axes[0].set_ylabel("Frequency", fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Log-transformed distribution
    axes[1].hist(
        log_df[target_column], bins=50, color="red", alpha=0.7, edgecolor="black"
    )
    axes[1].axvline(
        log_df[target_column].mean(),
        color="blue",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {log_df[target_column].mean():.2f}",
    )
    axes[1].axvline(
        log_df[target_column].median(),
        color="orange",
        linestyle=":",
        linewidth=2,
        label=f"Median: {log_df[target_column].median():.2f}",
    )
    axes[1].set_title(
        f"Distribution of {target_column} (Log-Transformed)",
        fontsize=14,
        fontweight="bold",
    )
    axes[1].set_xlabel("log1p(Number of Injuries)", fontsize=12)
    axes[1].set_ylabel("Frequency", fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(
        "Target Variable Analysis: Original vs Log-Transformed",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    # Save instead of show
    filename = output_dir / f"eda_distributions_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"✓ Saved EDA plot: {filename}")
    plt.close()  # ✅ Close the figure to free memory

    # Print statistics
    print(f"\n{'='*50}")
    print("ORIGINAL DISTRIBUTION STATS:")
    print(f"{'='*50}")
    print(f"Mean:     {df[target_column].mean():.2f}")
    print(f"Median:   {df[target_column].median():.0f}")
    print(f"Std Dev:  {df[target_column].std():.2f}")
    print(f"Min:      {df[target_column].min():.0f}")
    print(f"Max:      {df[target_column].max():.0f}")
    print(f"Skewness: {df[target_column].skew():.2f}")

    print(f"\n{'='*50}")
    print("LOG-TRANSFORMED DISTRIBUTION STATS:")
    print(f"{'='*50}")
    print(f"Mean:     {log_df[target_column].mean():.2f}")
    print(f"Median:   {log_df[target_column].median():.2f}")
    print(f"Std Dev:  {log_df[target_column].std():.2f}")
    print(f"Min:      {log_df[target_column].min():.2f}")
    print(f"Max:      {log_df[target_column].max():.2f}")
    print(f"Skewness: {log_df[target_column].skew():.2f}")
    print(f"{'='*50}\n")


def graph_model(
    model_results: dict,
    results_name: str,
    model_number: int = None,
    is_best: bool = False,
    add_classification_metrics: bool = True,
    injury_threshold: int = 3,
):
    """
    Generate comprehensive visualizations for model results

    Parameters:
    -----------
    model_results : dict
        Results dictionary from train_model (must contain keys: '10foldCV', 'y_test', 'y_pred', 'df', 'predictors')
    results_name : str
        Name for this model run (e.g., "baseline", "with_game_intensity")
    model_number : int, optional
        Model number (1, 2, 3, etc.)
    is_best : bool
        If True, label as "best" instead of using model_number
    add_classification_metrics : bool
        If True, convert to binary classification and add ROC/confusion matrix
    injury_threshold : int
        Threshold for binary classification
    """
    # Create filename prefix
    if is_best:
        file_prefix = f"{results_name}_best"
        display_name = f"{results_name} (BEST MODEL)"
    elif model_number is not None:
        file_prefix = f"{results_name}_{model_number}"
        display_name = f"{results_name} (Model {model_number})"
    else:
        file_prefix = results_name
        display_name = results_name

    print(f"\n{'='*70}")
    print(f"Graphing {display_name}")
    print(f"{'='*70}")

    # Create output directory
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extract data from results
    results = model_results["10foldCV"]
    y_test = np.array(model_results["y_test"])
    y_pred = np.array(model_results["y_pred"])
    df = model_results["df"]
    predictors = model_results["predictors"]
    model_type = model_results.get("model_type", "poisson")  # ✅ Get model type
    is_log_target = model_results.get(
        "is_log_target", False
    )  # ✅ Check if log-transformed

    actual_mean = y_test.mean()
    pred_mean = y_pred.mean()

    # ========== 1. METRICS BAR CHART ==========
    fig1, axes = plt.subplots(1, 3, figsize=(15, 4))

    # ✅ Adjust metrics based on model type
    if model_type == "poisson":
        metrics = ["poisson_dev", "mse", "mae"]
        titles = ["Poisson Deviance", "MSE", "MAE"]
    else:  # linear/ridge
        metrics = ["mse", "mae", "primary_metric"]
        titles = ["MSE", "MAE", "Primary Metric (MSE)"]

    for ax, metric, title in zip(axes, metrics, titles):
        # ✅ Handle missing Poisson deviance for linear models
        if metric == "poisson_dev" and model_type != "poisson":
            ax.text(
                0.5,
                0.5,
                "N/A for Linear Model",
                ha="center",
                va="center",
                fontsize=14,
                transform=ax.transAxes,
            )
            ax.set_title(title, fontsize=13, fontweight="bold")
            continue

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
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

        # Add percentage difference
        if cv_val > 0:
            pct_diff = ((test_val - cv_val) / cv_val) * 100
            diff_text = f"Test vs CV: {pct_diff:+.1f}%"
            color = "lightcoral" if pct_diff > 10 else "lightgreen"
            ax.text(
                0.5,
                0.95,
                diff_text,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor=color, alpha=0.5),
            )

    title_prefix = "⭐ BEST MODEL - " if is_best else ""
    model_type_label = f"({model_type.upper()}" + (
        " - Log Target)" if is_log_target else ")"
    )
    plt.suptitle(
        f"{title_prefix}Model Performance: {display_name} {model_type_label}",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()

    filename1 = output_dir / f"{file_prefix}_{timestamp}_metrics.png"
    plt.savefig(filename1, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {filename1}")
    plt.close()

    # ========== 2. ENHANCED PREDICTION SCATTER PLOT ==========
    fig2, ax = plt.subplots(figsize=(10, 10))

    # Create scatter with jitter for better visibility
    jitter_x = y_test + np.random.normal(0, 0.1, size=len(y_test))
    jitter_y = y_pred + np.random.normal(0, 0.05, size=len(y_pred))

    ax.scatter(
        jitter_x,
        jitter_y,
        alpha=0.5,
        s=60,
        edgecolors="black",
        linewidth=0.8,
        c="#3498db",
        label="Predictions",
    )

    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        lw=2.5,
        label="Perfect Prediction",
        zorder=5,
    )

    # Add mean prediction line
    ax.axhline(
        y=pred_mean,
        color="green",
        linestyle=":",
        lw=2,
        label=f"Mean Prediction: {pred_mean:.2f}",
        alpha=0.7,
    )
    ax.axvline(
        x=actual_mean,
        color="orange",
        linestyle=":",
        lw=2,
        label=f"Mean Actual: {actual_mean:.2f}",
        alpha=0.7,
    )

    ax.set_xlabel("Actual Injuries", fontsize=13)
    ax.set_ylabel("Predicted Injuries", fontsize=13)
    ax.set_title(
        f"{title_prefix}Actual vs Predicted: {display_name}",
        fontsize=15,
        fontweight="bold",
    )
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Calculate and display metrics
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # Prediction range
    pred_range = y_pred.max() - y_pred.min()

    textstr = (
        f"R² = {r2:.4f}\n"
        f"MAE = {mae:.4f}\n"
        f"RMSE = {rmse:.4f}\n"
        f"Pred Range = {pred_range:.2f}\n"
        f"Mean Pred = {pred_mean:.2f}\n"
        f"Mean Actual = {actual_mean:.2f}"
    )
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    ax.text(
        0.05,
        0.95,
        textstr,
        transform=ax.transAxes,
        fontsize=11,
        verticalalignment="top",
        bbox=props,
        fontweight="bold",
    )

    plt.tight_layout()

    filename2 = output_dir / f"{file_prefix}_{timestamp}_scatter.png"
    plt.savefig(filename2, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {filename2}")
    plt.close()

    # ========== 3. RESIDUALS ANALYSIS ==========
    residuals = y_test - y_pred

    fig3, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 3a. Residuals vs Predicted
    axes[0, 0].scatter(
        y_pred,
        residuals,
        alpha=0.6,
        s=60,
        edgecolors="black",
        linewidth=0.8,
        c="#e74c3c",
    )
    axes[0, 0].axhline(y=0, color="blue", linestyle="--", lw=2.5, label="Zero Residual")
    axes[0, 0].set_xlabel("Predicted Injuries", fontsize=12)
    axes[0, 0].set_ylabel("Residuals (Actual - Predicted)", fontsize=12)
    axes[0, 0].set_title("Residual Plot", fontsize=13, fontweight="bold")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Add std bands
    std_resid = np.std(residuals)
    axes[0, 0].axhline(
        y=2 * std_resid, color="orange", linestyle=":", lw=1.5, alpha=0.7
    )
    axes[0, 0].axhline(
        y=-2 * std_resid, color="orange", linestyle=":", lw=1.5, alpha=0.7
    )

    # 3b. Residuals histogram
    axes[0, 1].hist(residuals, bins=30, edgecolor="black", alpha=0.7, color="#9b59b6")
    axes[0, 1].axvline(x=0, color="red", linestyle="--", lw=2.5, label="Zero")
    axes[0, 1].axvline(
        x=residuals.mean(),
        color="green",
        linestyle=":",
        lw=2,
        label=f"Mean: {residuals.mean():.3f}",
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
        y_test,
        residuals,
        alpha=0.6,
        s=60,
        edgecolors="black",
        linewidth=0.8,
        c="#e67e22",
    )
    axes[1, 1].axhline(y=0, color="blue", linestyle="--", lw=2.5)
    axes[1, 1].set_xlabel("Actual Injuries", fontsize=12)
    axes[1, 1].set_ylabel("Residuals", fontsize=12)
    axes[1, 1].set_title("Residuals vs Actual", fontsize=13, fontweight="bold")
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(
        f"{title_prefix}Residual Analysis: {display_name}",
        fontsize=15,
        fontweight="bold",
    )
    plt.tight_layout()

    filename3 = output_dir / f"{file_prefix}_{timestamp}_residuals.png"
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
        data,
        positions=positions,
        widths=0.6,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", alpha=0.7),
        medianprops=dict(color="red", linewidth=2),
    )

    ax.axhline(y=0, color="black", linestyle="--", lw=1.5, alpha=0.5)
    ax.set_xlabel("Actual Number of Injuries", fontsize=12)
    ax.set_ylabel("Prediction Error", fontsize=12)
    ax.set_title(
        f"{title_prefix}Prediction Error by Actual Injury Count: {display_name}",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)

    # Add sample sizes
    for pos in positions:
        count = len(error_by_actual[pos])
        ax.text(
            pos,
            ax.get_ylim()[1] * 0.95,
            f"n={count}",
            ha="center",
            fontsize=9,
            style="italic",
        )

    plt.tight_layout()

    filename4 = output_dir / f"{file_prefix}_{timestamp}_error_by_count.png"
    plt.savefig(filename4, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {filename4}")
    plt.close()

    # ========== 5. CLASSIFICATION METRICS (if requested) ==========

    if add_classification_metrics:
        print(
            f"\n📊 Creating classification metrics with threshold: {injury_threshold}+ injuries..."
        )

        # Convert to binary classification
        y_test_binary = (y_test >= injury_threshold).astype(int)
        y_pred_binary = (y_pred >= injury_threshold).astype(int)

        # Check if we have both classes
        unique_test = np.unique(y_test_binary)
        unique_pred = np.unique(y_pred_binary)

        if len(unique_test) < 2:
            print(
                f"⚠️  Skipping classification metrics: All test samples are in the same class"
            )
            print(f"   Test class distribution: {np.bincount(y_test_binary)}")
        elif len(unique_pred) < 2:
            print(
                f"⚠️  Skipping classification metrics: All predictions are in the same class"
            )
            print(f"   Prediction class distribution: {np.bincount(y_pred_binary)}")
        else:
            # Normalize as proxy for probability
            if y_pred.max() > 0:
                y_pred_proba = y_pred / y_pred.max()
            else:
                y_pred_proba = y_pred

            # 5a. Confusion Matrix and ROC Curve
            fig5, axes = plt.subplots(1, 2, figsize=(15, 6))

            cm = confusion_matrix(y_test_binary, y_pred_binary)

            # ✅ Check confusion matrix shape
            if cm.shape == (2, 2):
                # Plot confusion matrix
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    ax=axes[0],
                    xticklabels=[f"<{injury_threshold}", f"{injury_threshold}+"],
                    yticklabels=[f"<{injury_threshold}", f"{injury_threshold}+"],
                    cbar_kws={"label": "Count"},
                )
                axes[0].set_ylabel("True Label", fontsize=12)
                axes[0].set_xlabel("Predicted Label", fontsize=12)
                axes[0].set_title(
                    f"Confusion Matrix\n(Threshold: {injury_threshold}+ injuries)",
                    fontsize=13,
                    fontweight="bold",
                )

                # Add metrics to confusion matrix
                tn, fp, fn, tp = cm.ravel()
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = (
                    2 * (precision * recall) / (precision + recall)
                    if (precision + recall) > 0
                    else 0
                )

                metrics_text = (
                    f"Accuracy: {accuracy:.3f}\n"
                    f"Precision: {precision:.3f}\n"
                    f"Recall: {recall:.3f}\n"
                    f"F1-Score: {f1:.3f}"
                )
                props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
                axes[0].text(
                    1.5, -0.3, metrics_text, fontsize=11, bbox=props, fontweight="bold"
                )

                # 5b. ROC Curve
                try:
                    fpr, tpr, thresholds = roc_curve(y_test_binary, y_pred_proba)
                    roc_auc = auc(fpr, tpr)

                    axes[1].plot(
                        fpr,
                        tpr,
                        color="darkorange",
                        lw=2.5,
                        label=f"ROC curve (AUC = {roc_auc:.3f})",
                    )
                    axes[1].plot(
                        [0, 1],
                        [0, 1],
                        color="navy",
                        lw=2,
                        linestyle="--",
                        label="Random Classifier",
                    )
                    axes[1].set_xlim([0.0, 1.0])
                    axes[1].set_ylim([0.0, 1.05])
                    axes[1].set_xlabel("False Positive Rate", fontsize=12)
                    axes[1].set_ylabel("True Positive Rate", fontsize=12)
                    axes[1].set_title(
                        f"ROC Curve\n(Threshold: {injury_threshold}+ injuries)",
                        fontsize=13,
                        fontweight="bold",
                    )
                    axes[1].legend(loc="lower right", fontsize=11)
                    axes[1].grid(True, alpha=0.3)
                except Exception as e:
                    print(f"⚠️  Could not generate ROC curve: {e}")
                    axes[1].text(
                        0.5,
                        0.5,
                        f"ROC curve unavailable\n{str(e)}",
                        ha="center",
                        va="center",
                        fontsize=12,
                        transform=axes[1].transAxes,
                    )

                plt.suptitle(
                    f"{title_prefix}Classification Metrics: {display_name}",
                    fontsize=15,
                    fontweight="bold",
                )
                plt.tight_layout()

                filename5 = output_dir / f"{file_prefix}_{timestamp}_classification.png"
                plt.savefig(filename5, dpi=300, bbox_inches="tight")
                print(f"✓ Saved: {filename5}")
                plt.close()

                # Print classification report
                print(f"\n📋 Classification Report (Threshold: {injury_threshold}+):")
                try:
                    print(
                        classification_report(
                            y_test_binary,
                            y_pred_binary,
                            target_names=[
                                f"<{injury_threshold} injuries",
                                f"{injury_threshold}+ injuries",
                            ],
                            zero_division=0,
                        )
                    )
                except Exception as e:
                    print(f"⚠️  Could not generate classification report: {e}")

            else:
                # Confusion matrix is not 2x2 (missing a class)
                print(
                    f"⚠️  Skipping classification metrics: Confusion matrix shape is {cm.shape}"
                )
                print(f"   Expected (2,2) but model only predicts one class")
                print(f"   Test distribution: {np.bincount(y_test_binary)}")
                print(f"   Pred distribution: {np.bincount(y_pred_binary)}")

    # ========== 6. DATA DISTRIBUTION (only for first model or best) ==========
    if model_number == 1 or is_best:
        target = "num_injuries"
        if target in df.columns:
            fig6, axes = plt.subplots(2, 2, figsize=(15, 12))

            # 6a. Histogram
            axes[0, 0].hist(
                df[target],
                bins=range(int(df[target].min()), int(df[target].max()) + 2),
                edgecolor="black",
                alpha=0.7,
                color="#2ecc71",
            )
            axes[0, 0].axvline(
                df[target].mean(),
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {df[target].mean():.2f}",
            )
            axes[0, 0].axvline(
                df[target].median(),
                color="blue",
                linestyle=":",
                linewidth=2,
                label=f"Median: {df[target].median():.0f}",
            )
            axes[0, 0].set_xlabel("Number of Injuries", fontsize=12)
            axes[0, 0].set_ylabel("Frequency", fontsize=12)
            axes[0, 0].set_title(
                "Distribution of Injuries in Dataset", fontsize=13, fontweight="bold"
            )
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

            # 6b. Value counts bar chart
            value_counts = df[target].value_counts().sort_index()
            axes[0, 1].bar(
                value_counts.index,
                value_counts.values,
                edgecolor="black",
                alpha=0.7,
                color="#e67e22",
            )
            axes[0, 1].set_xlabel("Number of Injuries", fontsize=12)
            axes[0, 1].set_ylabel("Count", fontsize=12)
            axes[0, 1].set_title(
                "Frequency of Each Injury Count", fontsize=13, fontweight="bold"
            )
            axes[0, 1].grid(True, alpha=0.3, axis="y")

            # 6c. Cumulative distribution
            sorted_injuries = np.sort(df[target])
            cumulative = (
                np.arange(1, len(sorted_injuries) + 1) / len(sorted_injuries) * 100
            )
            axes[1, 0].plot(sorted_injuries, cumulative, linewidth=2, color="#9b59b6")
            axes[1, 0].set_xlabel("Number of Injuries", fontsize=12)
            axes[1, 0].set_ylabel("Cumulative Percentage", fontsize=12)
            axes[1, 0].set_title(
                "Cumulative Distribution", fontsize=13, fontweight="bold"
            )
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].axhline(50, color="red", linestyle="--", alpha=0.5)

            # 6d. Statistics box
            axes[1, 1].axis("off")
            stats_text = f"""
            INJURY STATISTICS
            {'='*40}
            
            Count:        {len(df[target])}
            Mean:         {df[target].mean():.2f}
            Median:       {df[target].median():.0f}
            Mode:         {df[target].mode()[0]:.0f}
            Std Dev:      {df[target].std():.2f}
            Variance:     {df[target].var():.2f}
            
            Min:          {df[target].min():.0f}
            25th %ile:    {df[target].quantile(0.25):.0f}
            50th %ile:    {df[target].quantile(0.50):.0f}
            75th %ile:    {df[target].quantile(0.75):.0f}
            Max:          {df[target].max():.0f}
            
            Range:        {df[target].max() - df[target].min():.0f}
            IQR:          {df[target].quantile(0.75) - df[target].quantile(0.25):.0f}
            
            Skewness:     {df[target].skew():.2f}
            Kurtosis:     {df[target].kurtosis():.2f}
            """
            axes[1, 1].text(
                0.1,
                0.9,
                stats_text,
                transform=axes[1, 1].transAxes,
                fontsize=11,
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

            plt.suptitle(
                f"Target Variable Analysis: {display_name}",
                fontsize=15,
                fontweight="bold",
            )
            plt.tight_layout()

            filename6 = output_dir / f"{file_prefix}_{timestamp}_distribution.png"
            plt.savefig(filename6, dpi=300, bbox_inches="tight")
            print(f"✓ Saved: {filename6}")
            plt.close()

    # ========== 7. GENERATE TEXT REPORT ==========
    report = generate_text_report(
        display_name,
        results,
        predictors,
        y_test,
        y_pred,
        actual_mean,
        pred_mean,
        is_best,
        model_type=model_type,  # ✅ Add parameter
        is_log_target=is_log_target,
    )

    report_file = output_dir / f"{file_prefix}_{timestamp}_report.txt"
    with open(report_file, "w") as f:
        f.write(report)

    print("\n" + report)
    print(f"✓ Report saved: {report_file}")
    print(f"\n{'='*70}")
    print(f"All visualizations saved to: {output_dir}/")
    print(f"{'='*70}\n")


def generate_text_report(
    display_name,
    results,
    predictors,
    y_test,
    y_pred,
    actual_mean,
    pred_mean,
    is_best=False,
    model_type="poisson",
    is_log_target=False,
):
    """Generate detailed text report"""
    best_marker = "⭐ BEST MODEL ⭐" if is_best else ""

    model_type_label = f"{model_type.upper()}"
    if is_log_target:
        model_type_label += " (Log-Transformed Target)"

    report = f"""
{'='*70}
{best_marker}
MODEL REPORT: {display_name}
MODEL TYPE: {model_type_label}
{'='*70}
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

PREDICTORS ({len(predictors)}):
{', '.join(predictors)}

CROSS-VALIDATION RESULTS (10-fold):
"""

    # Add Poisson deviance only for Poisson models
    if model_type == "poisson" and "val_poisson_dev" in results:
        report += f"  • Poisson Deviance: {results['val_poisson_dev']:.4f}\n"

    report += f"""  • MSE:              {results['val_mse']:.4f}
  • RMSE:             {np.sqrt(results['val_mse']):.4f}
  • MAE:              {results['val_mae']:.4f}

TEST SET RESULTS:
"""

    # Add Poisson deviance only for Poisson models
    if model_type == "poisson" and "test_poisson_dev" in results:
        report += f"  • Poisson Deviance: {results['test_poisson_dev']:.4f}\n"

    report += f"""  • MSE:              {results['test_mse']:.4f}
  • RMSE:             {np.sqrt(results['test_mse']):.4f}
  • MAE:              {results['test_mae']:.4f}

GENERALIZATION (Test/CV ratio):
"""

    # Add Poisson dev ratio only for Poisson models
    if (
        model_type == "poisson"
        and "val_poisson_dev" in results
        and "test_poisson_dev" in results
    ):
        poisson_ratio = results["test_poisson_dev"] / results["val_poisson_dev"]
        report += f"  • Poisson Dev:      {poisson_ratio:.4f}\n"

    mse_ratio = results["test_mse"] / results["val_mse"]
    mae_ratio = results["test_mae"] / results["val_mae"]

    report += f"""  • MSE:              {mse_ratio:.4f}
  • MAE:              {mae_ratio:.4f}
  
  Note: Ratio > 1.1 suggests overfitting
"""

    # Inside your linear model evaluation/reporting:
    if is_log_target:
        # Convert back to original scale
        y_test_original = np.expm1(y_test)  # inverse of log1p
        y_pred_original = np.maximum(np.expm1(y_pred), 0)  # clip negatives to 0

        # Calculate metrics in original scale
        original_rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original))
        original_mae = mean_absolute_error(y_test_original, y_pred_original)
        original_r2 = r2_score(y_test_original, y_pred_original)

        print(f"\n[ORIGINAL SCALE METRICS]")
        print(f"RMSE: {original_rmse:.4f} injuries")
        print(f"MAE: {original_mae:.4f} injuries")
        print(f"R²: {original_r2:.4f}")
        print(f"Pred Range: [{y_pred_original.min():.2f}, {y_pred_original.max():.2f}]")

    r2 = r2_score(y_test, y_pred)
    pred_range = y_pred.max() - y_pred.min()
    actual_range = y_test.max() - y_test.min()

    report += f"""
ADDITIONAL METRICS:
  • R² Score:         {r2:.4f}
  • Mean Actual:      {actual_mean:.2f}
  • Mean Predicted:   {pred_mean:.2f}
  • Actual Range:     {actual_range:.2f}
  • Predicted Range:  {pred_range:.2f}

MODEL INTERPRETATION:
  • R² = {r2:.1%} means the model explains {r2:.1%} of variance
  • Predictions range from {y_pred.min():.2f} to {y_pred.max():.2f}
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
        primary_metric = "Poisson Deviance" if model_type == "poisson" else "MSE"
        report += f"""
{'='*70}
🎯 This is the BEST model based on 10-fold CV {primary_metric}
   - Retrained on full 80/20 train/test split
   - Use this model for final predictions
{'='*70}
"""

    report += f"\n{'='*70}\n"
    return report
