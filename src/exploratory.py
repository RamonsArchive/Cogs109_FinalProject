"""
Exploratory Data Analysis for NFL Injury Prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime


def correlation_analysis(df: pd.DataFrame, log_df: pd.DataFrame, target_column: str):
    """
    Perform correlation analysis between predictors and target variable.

    Parameters:
    -----------
    df : pd.DataFrame
        Original dataframe with count data
    log_df : pd.DataFrame
        Log-transformed dataframe
    target_column : str
        Name of target column (e.g., 'num_injuries')
    """
    print("\n" + "=" * 80)
    print("📊 CORRELATION ANALYSIS: Predictors vs Injury Outcomes")
    print("=" * 80)

    # Define all predictors
    all_predictors = [
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

    # Filter to only numeric columns that exist
    numeric_predictors = [
        col
        for col in all_predictors
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]

    # Categorical predictors
    categorical_predictors = [
        col
        for col in all_predictors
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])
    ]

    print(f"\n✓ Found {len(numeric_predictors)} numeric predictors")
    print(f"✓ Found {len(categorical_predictors)} categorical predictors")

    # === NUMERIC CORRELATIONS ===
    print("\n" + "=" * 80)
    print("NUMERIC PREDICTORS - Pearson Correlation with Injury Count")
    print("=" * 80)

    # Calculate correlations for original data
    correlations_original = []
    for predictor in numeric_predictors:
        corr = df[predictor].corr(df[target_column])
        correlations_original.append(
            {"Predictor": predictor, "Correlation": corr, "Abs_Correlation": abs(corr)}
        )

    # Sort by absolute correlation
    corr_df_original = pd.DataFrame(correlations_original).sort_values(
        "Abs_Correlation", ascending=False
    )

    print("\n📈 Original Data (Count):")
    print("-" * 80)
    for _, row in corr_df_original.iterrows():
        corr = row["Correlation"]
        predictor = row["Predictor"]

        # Strength indicator
        if abs(corr) > 0.3:
            strength = "🔴 STRONG"
        elif abs(corr) > 0.1:
            strength = "🟡 MODERATE"
        else:
            strength = "🔵 WEAK"

        # Direction
        direction = "↗️ Positive" if corr > 0 else "↘️ Negative"

        print(f"{strength:15} {predictor:35} r = {corr:7.4f}  {direction}")

    # Calculate correlations for log-transformed data
    print("\n📈 Log-Transformed Data:")
    print("-" * 80)
    correlations_log = []
    for predictor in numeric_predictors:
        corr = log_df[predictor].corr(log_df[target_column])
        correlations_log.append(
            {"Predictor": predictor, "Correlation": corr, "Abs_Correlation": abs(corr)}
        )

    corr_df_log = pd.DataFrame(correlations_log).sort_values(
        "Abs_Correlation", ascending=False
    )

    for _, row in corr_df_log.iterrows():
        corr = row["Correlation"]
        predictor = row["Predictor"]

        # Strength indicator
        if abs(corr) > 0.3:
            strength = "🔴 STRONG"
        elif abs(corr) > 0.1:
            strength = "🟡 MODERATE"
        else:
            strength = "🔵 WEAK"

        direction = "↗️ Positive" if corr > 0 else "↘️ Negative"
        print(f"{strength:15} {predictor:35} r = {corr:7.4f}  {direction}")

    # === CATEGORICAL ANALYSIS ===
    if categorical_predictors:
        print("\n" + "=" * 80)
        print("CATEGORICAL PREDICTORS - Mean Injury Count by Category")
        print("=" * 80)

        for predictor in categorical_predictors:
            print(f"\n📊 {predictor}:")
            print("-" * 60)
            grouped = df.groupby(predictor)[target_column].agg(["mean", "std", "count"])
            grouped = grouped.sort_values("mean", ascending=False)

            for category, row in grouped.iterrows():
                print(
                    f"  {str(category):20} → Avg: {row['mean']:.2f}  "
                    f"(±{row['std']:.2f}, n={int(row['count'])})"
                )

    # === VISUALIZATION ===
    create_correlation_plots(
        df,
        log_df,
        target_column,
        corr_df_original,
        corr_df_log,
        numeric_predictors,
        categorical_predictors,
    )

    # === SUMMARY ===
    print("\n" + "=" * 80)
    print("🎯 KEY FINDINGS SUMMARY")
    print("=" * 80)

    # Top 5 strongest correlations (original data)
    top_5 = corr_df_original.head(5)
    print("\n🏆 Top 5 Most Correlated Predictors (Original Data):")
    for i, row in enumerate(top_5.iterrows(), 1):
        _, data = row
        print(f"  {i}. {data['Predictor']:30} r = {data['Correlation']:7.4f}")

    # Check if any strong correlations exist
    strong_corrs = corr_df_original[corr_df_original["Abs_Correlation"] > 0.3]
    if len(strong_corrs) == 0:
        print("\n⚠️  WARNING: No strong correlations found (|r| > 0.3)")
        print("   This explains why your models have low predictive power!")
        print("   Consider:")
        print("   • Adding interaction terms (e.g., temp × humidity)")
        print("   • Creating aggregate features (e.g., team injury history)")
        print("   • Collecting different types of features")

    return corr_df_original, corr_df_log


def create_correlation_plots(
    df,
    log_df,
    target_column,
    corr_df_original,
    corr_df_log,
    numeric_predictors,
    categorical_predictors,
):
    """Create visualizations for correlation analysis"""

    output_dir = Path("plots") / "exploratory"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # === 1. Correlation Bar Chart ===
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Original data
    top_10_orig = corr_df_original.head(10)
    colors_orig = ["red" if x > 0 else "blue" for x in top_10_orig["Correlation"]]
    axes[0].barh(
        top_10_orig["Predictor"],
        top_10_orig["Correlation"],
        color=colors_orig,
        alpha=0.7,
    )
    axes[0].set_xlabel("Correlation Coefficient", fontsize=12)
    axes[0].set_title(
        "Top 10 Correlations - Original Data", fontsize=14, fontweight="bold"
    )
    axes[0].axvline(x=0, color="black", linestyle="--", linewidth=1)
    axes[0].grid(True, alpha=0.3)

    # Log data
    top_10_log = corr_df_log.head(10)
    colors_log = ["red" if x > 0 else "blue" for x in top_10_log["Correlation"]]
    axes[1].barh(
        top_10_log["Predictor"], top_10_log["Correlation"], color=colors_log, alpha=0.7
    )
    axes[1].set_xlabel("Correlation Coefficient", fontsize=12)
    axes[1].set_title(
        "Top 10 Correlations - Log-Transformed Data", fontsize=14, fontweight="bold"
    )
    axes[1].axvline(x=0, color="black", linestyle="--", linewidth=1)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    filename = output_dir / f"correlation_analysis_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"\n✓ Saved: {filename}")
    plt.close()

    # === 2. Correlation Heatmap ===
    if len(numeric_predictors) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(18, 10))

        # Select top predictors and target
        top_predictors = corr_df_original.head(10)["Predictor"].tolist()
        cols_to_plot = [col for col in top_predictors if col in df.columns] + [
            target_column
        ]

        # Original data heatmap
        corr_matrix_orig = df[cols_to_plot].corr()
        sns.heatmap(
            corr_matrix_orig,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"label": "Correlation"},
            ax=axes[0],
        )
        axes[0].set_title(
            "Correlation Heatmap - Original Data", fontsize=14, fontweight="bold"
        )

        # Log data heatmap
        corr_matrix_log = log_df[cols_to_plot].corr()
        sns.heatmap(
            corr_matrix_log,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"label": "Correlation"},
            ax=axes[1],
        )
        axes[1].set_title(
            "Correlation Heatmap - Log-Transformed Data", fontsize=14, fontweight="bold"
        )

        plt.tight_layout()
        filename = output_dir / f"correlation_heatmap_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {filename}")
        plt.close()

    # === 3. Scatter Plots for Top 3 Correlations ===
    top_3_predictors = corr_df_original.head(3)["Predictor"].tolist()

    if len(top_3_predictors) > 0:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for idx, predictor in enumerate(top_3_predictors):
            if predictor in df.columns:
                axes[idx].scatter(
                    df[predictor],
                    df[target_column],
                    alpha=0.5,
                    s=50,
                    edgecolors="black",
                    linewidth=0.5,
                )

                # Add trend line
                z = np.polyfit(
                    df[predictor].dropna(), df[target_column][df[predictor].notna()], 1
                )
                p = np.poly1d(z)
                x_trend = np.linspace(df[predictor].min(), df[predictor].max(), 100)
                axes[idx].plot(
                    x_trend, p(x_trend), "r--", linewidth=2, label=f"Trend line"
                )

                corr = df[predictor].corr(df[target_column])
                axes[idx].set_xlabel(predictor, fontsize=11)
                axes[idx].set_ylabel(target_column if idx == 0 else "", fontsize=11)
                axes[idx].set_title(
                    f"{predictor}\n(r = {corr:.3f})", fontsize=12, fontweight="bold"
                )
                axes[idx].grid(True, alpha=0.3)
                axes[idx].legend()

        plt.tight_layout()
        filename = output_dir / f"correlation_scatter_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {filename}")
        plt.close()

    print(f"\n{'='*80}")
    print(f"All correlation plots saved to: {output_dir}/")
    print(f"{'='*80}\n")
