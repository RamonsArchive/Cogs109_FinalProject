import os
"""
Exploratory Data Analysis for NFL Injury Prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from load import load_data
from clean import clean_data


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


def create_all_predictors_correlation_graph(df: pd.DataFrame, target_column: str, predictors: list):
    """
    Create a comprehensive correlation graph for ALL predictors
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with data
    target_column : str
        Name of target column
    predictors : list
        List of all predictor column names
    """
    output_dir = Path("plots") / "exploratory"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Filter to only columns that exist
    valid_predictors = [p for p in predictors if p in df.columns]
    
    # Calculate correlations for all numeric predictors
    correlations = []
    for predictor in valid_predictors:
        if pd.api.types.is_numeric_dtype(df[predictor]):
            corr = df[predictor].corr(df[target_column])
            correlations.append({
                "Predictor": predictor,
                "Correlation": corr,
                "Abs_Correlation": abs(corr)
            })
    
    # Sort by absolute correlation
    corr_df = pd.DataFrame(correlations).sort_values("Abs_Correlation", ascending=False)
    
    # Create figure - wider to accommodate labels
    fig = plt.figure(figsize=(16, max(10, len(corr_df) * 0.5)))
    ax = fig.add_subplot(111)
    
    # Color code by correlation strength
    colors = []
    for corr in corr_df["Correlation"]:
        if abs(corr) > 0.3:
            colors.append("#e74c3c")  # Red for strong
        elif abs(corr) > 0.1:
            colors.append("#f39c12")  # Orange for moderate
        else:
            colors.append("#3498db")  # Blue for weak
    
    # Create horizontal bar chart with more spacing
    bars = ax.barh(
        corr_df["Predictor"],
        corr_df["Correlation"],
        color=colors,
        alpha=0.7,
        edgecolor="black",
        linewidth=1.5,
        height=0.7  # Make bars thicker for better visibility
    )
    
    # Add value labels - positioned inside bars to avoid overflow
    for i, (bar, corr) in enumerate(zip(bars, corr_df["Correlation"])):
        width = bar.get_width()
        # Position label inside the bar if there's enough space, otherwise outside
        if abs(width) > 0.05:  # If bar is wide enough
            ax.text(
                width * 0.5, i, f"{corr:.3f}",
                ha="center", va="center", fontsize=10, fontweight="bold",
                color="white" if abs(corr) > 0.1 else "black"
            )
        else:  # If bar is too narrow, put label outside
            ax.text(
                width + (0.02 if width > 0 else -0.02), i, f"{corr:.3f}",
                ha="left" if width > 0 else "right", va="center", 
                fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", 
                facecolor="white", alpha=0.8, edgecolor="gray")
            )
    
    # Set x-axis limits with padding for labels
    x_min = corr_df["Correlation"].min()
    x_max = corr_df["Correlation"].max()
    x_range = x_max - x_min
    ax.set_xlim(x_min - x_range * 0.15, x_max + x_range * 0.15)
    
    ax.set_xlabel("Correlation Coefficient with " + target_column, fontsize=13, fontweight="bold")
    ax.set_ylabel("Predictors", fontsize=13, fontweight="bold")
    ax.set_title(
        f"Correlation Analysis: All Numerical Predictors vs {target_column}\n"
        f"Total Predictors: {len(corr_df)} | "
        f"Interpretation: Measures linear relationship strength (-1 to +1)",
        fontsize=14, fontweight="bold", pad=20
    )
    ax.axvline(x=0, color="black", linestyle="--", linewidth=2, alpha=0.5)
    ax.grid(True, alpha=0.3, axis="x", linestyle=":")
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#e74c3c", alpha=0.7, label="Strong (|r| > 0.3)"),
        Patch(facecolor="#f39c12", alpha=0.7, label="Moderate (0.1 < |r| ≤ 0.3)"),
        Patch(facecolor="#3498db", alpha=0.7, label="Weak (|r| ≤ 0.1)")
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=11, framealpha=0.9)

    
    plt.tight_layout()
    filename = output_dir / f"all_predictors_correlation_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {filename}")
    plt.close()
    
    return corr_df


def create_numerical_categorical_comparison(df: pd.DataFrame, target_column: str, predictors: list):
    """
    Create side-by-side graphs comparing numerical and categorical predictors
    
    Parameters:
    -----------
    df : pd.DataFrame
        Dataframe with data
    target_column : str
        Name of target column
    predictors : list
        List of all predictor column names
    """
    output_dir = Path("plots") / "exploratory"
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Filter to only columns that exist
    valid_predictors = [p for p in predictors if p in df.columns]
    
    # Separate numerical and categorical
    num_cols = [col for col in valid_predictors if pd.api.types.is_numeric_dtype(df[col])]
    cat_cols = [col for col in valid_predictors if not pd.api.types.is_numeric_dtype(df[col])]
    
    print(f"\n📊 Found {len(num_cols)} numerical predictors and {len(cat_cols)} categorical predictors")
    
    # Create figure with subplots - wider for better spacing
    fig, axes = plt.subplots(1, 2, figsize=(20, max(10, max(len(num_cols), len(cat_cols)) * 0.5)))
    
    # === LEFT PANEL: NUMERICAL PREDICTORS ===
    if len(num_cols) > 0:
        # Calculate correlations for numerical predictors
        num_correlations = []
        for predictor in num_cols:
            corr = df[predictor].corr(df[target_column])
            num_correlations.append({
                "Predictor": predictor,
                "Correlation": corr,
                "Abs_Correlation": abs(corr)
            })
        
        num_corr_df = pd.DataFrame(num_correlations).sort_values("Abs_Correlation", ascending=False)
        
        # Color code
        colors_num = []
        for corr in num_corr_df["Correlation"]:
            if abs(corr) > 0.3:
                colors_num.append("#e74c3c")
            elif abs(corr) > 0.1:
                colors_num.append("#f39c12")
            else:
                colors_num.append("#3498db")
        
        # Create horizontal bar chart with more spacing
        bars = axes[0].barh(
            num_corr_df["Predictor"],
            num_corr_df["Correlation"],
            color=colors_num,
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
            height=0.7
        )
        
        # Add value labels - positioned inside bars
        for i, (bar, corr) in enumerate(zip(bars, num_corr_df["Correlation"])):
            width = bar.get_width()
            if abs(width) > 0.05:
                axes[0].text(
                    width * 0.5, i, f"{corr:.3f}",
                    ha="center", va="center", fontsize=10, fontweight="bold",
                    color="white" if abs(corr) > 0.1 else "black"
                )
            else:
                axes[0].text(
                    width + (0.02 if width > 0 else -0.02), i, f"{corr:.3f}",
                    ha="left" if width > 0 else "right", va="center", 
                    fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3",
                    facecolor="white", alpha=0.8, edgecolor="gray")
                )
        
        # Set x-axis limits with padding
        x_min = num_corr_df["Correlation"].min()
        x_max = num_corr_df["Correlation"].max()
        x_range = x_max - x_min
        axes[0].set_xlim(x_min - x_range * 0.15, x_max + x_range * 0.15)
        
        axes[0].set_xlabel("Correlation Coefficient", fontsize=13, fontweight="bold")
        axes[0].set_ylabel("Predictors", fontsize=13, fontweight="bold")
        axes[0].set_title(
            f"Numerical Predictors: Pearson Correlation\n"
            f"({len(num_cols)} predictors) | Measures linear relationship",
            fontsize=13, fontweight="bold", pad=15
        )
        axes[0].axvline(x=0, color="black", linestyle="--", linewidth=2, alpha=0.5)
        axes[0].grid(True, alpha=0.3, axis="x", linestyle=":")
    else:
        axes[0].text(0.5, 0.5, "No numerical predictors", 
                     ha="center", va="center", fontsize=12)
        axes[0].set_title("Numerical Predictors", fontsize=13, fontweight="bold")
    
    # === RIGHT PANEL: CATEGORICAL PREDICTORS ===
    if len(cat_cols) > 0:
        # Calculate mean injury count by category for each categorical predictor
        cat_means = []
        cat_details = {}  # Store detailed info for each predictor
        
        for predictor in cat_cols:
            grouped = df.groupby(predictor)[target_column].agg(["mean", "count"])
            overall_mean = df[target_column].mean()
            max_diff = (grouped["mean"] - overall_mean).abs().max()
            
            # Store category-level details
            cat_details[predictor] = {
                "categories": grouped.index.tolist(),
                "means": grouped["mean"].tolist(),
                "counts": grouped["count"].tolist()
            }
            
            cat_means.append({
                "Predictor": predictor,
                "Max_Difference": max_diff,
                "Mean_Injury": grouped["mean"].mean()  # Average across all categories
            })
        
        cat_means_df = pd.DataFrame(cat_means).sort_values("Max_Difference", ascending=False)
        
        # Create horizontal bar chart showing mean injury count
        bars = axes[1].barh(
            cat_means_df["Predictor"],
            cat_means_df["Mean_Injury"],
            color="#9b59b6",
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
            height=0.7
        )
        
        # Add value labels
        for i, (bar, mean_val) in enumerate(zip(bars, cat_means_df["Mean_Injury"])):
            width = bar.get_width()
            axes[1].text(
                width * 0.5, i, f"{mean_val:.2f}",
                ha="center", va="center", fontsize=10, fontweight="bold",
                color="white"
            )
        
        # Add overall mean line
        overall_mean = df[target_column].mean()
        axes[1].axvline(x=overall_mean, color="red", linestyle="--", 
                       linewidth=2.5, label=f"Overall Mean: {overall_mean:.2f}", zorder=5)
        
        axes[1].set_xlabel("Average Mean Injury Count Across Categories", 
                          fontsize=13, fontweight="bold")
        axes[1].set_ylabel("Predictors", fontsize=13, fontweight="bold")
        axes[1].set_title(
            f"Categorical Predictors: Mean Injury Count\n"
            f"({len(cat_cols)} predictors) | Cannot use correlation (non-numeric)",
            fontsize=13, fontweight="bold", pad=15
        )
        axes[1].grid(True, alpha=0.3, axis="x", linestyle=":")
        axes[1].legend(fontsize=10, framealpha=0.9, loc="lower right")
    
    else:
        axes[1].text(0.5, 0.5, "No categorical predictors", 
                     ha="center", va="center", fontsize=12)
        axes[1].set_title("Categorical Predictors", fontsize=13, fontweight="bold")
    
    plt.suptitle(
        f"Predictor Analysis: Numerical vs Categorical\n"
        f"Target: {target_column} | "
        f"Left: Correlation (linear relationship) | Right: Mean Comparison (category differences)",
        fontsize=15, fontweight="bold", y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    filename = output_dir / f"numerical_categorical_comparison_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {filename}")
    plt.close()
    
    # Print detailed explanation
    print("\n" + "=" * 80)
    print("📖 INTERPRETATION GUIDE FOR CATEGORICAL PREDICTORS")
    print("=" * 80)
    print("\nThe 'Mean Injury Count' shown is the AVERAGE of mean injury counts")
    print("across all categories within each categorical predictor.\n")
    print("EXAMPLE: For 'week' predictor:")
    print("  • Week 1 might have mean = 2.5 injuries")
    print("  • Week 2 might have mean = 2.8 injuries")
    print("  • Week 3 might have mean = 2.4 injuries")
    print("  • ... (for all weeks)")
    print("  • The bar shows: (2.5 + 2.8 + 2.4 + ...) / number of weeks = 2.58\n")
    print("WHY CAN'T WE USE CORRELATION FOR CATEGORICAL VARIABLES?")
    print("  • Correlation measures LINEAR relationships between NUMBERS")
    print("  • Categorical variables have no inherent numerical order")
    print("  • Example: 'Monday' vs 'Tuesday' - which is 'higher'?")
    print("  • Solution: Compare mean injury counts across categories instead")
    print("=" * 80 + "\n")


def main():
    """
    Main function to generate exploratory data analysis graphs
    """
    print("\n" + "=" * 80)
    print("🔍 EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    
    # Load data
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(
        curr_dir,
        "../datasets/df_master_schedule_injury_surface_2019_23_weather_distance_days_since_last_game.csv",
    )
    
    print("\n📂 Loading data...")
    df = load_data(data_dir)
    df = clean_data(df)
    print(f"✓ Loaded {len(df)} rows, {len(df.columns)} columns")

    # Define all predictors (Model 6: Kitchen sink)
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
    
    target_column = "num_injuries"
    
    # Create output directory
    output_dir = Path("plots") / "exploratory"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("📊 GENERATING CORRELATION GRAPHS")
    print("=" * 80)
    
    # 1. Create correlation graph for ALL predictors
    print("\n1️⃣ Creating correlation graph for ALL predictors...")
    corr_df = create_all_predictors_correlation_graph(df, target_column, predictors)
    print(f"   ✓ Analyzed {len(corr_df)} numerical predictors")
    
    # 2. Create numerical vs categorical comparison
    print("\n2️⃣ Creating numerical vs categorical comparison...")
    create_numerical_categorical_comparison(df, target_column, predictors)
    
    print("\n" + "=" * 80)
    print(f"✅ ALL GRAPHS SAVED TO: {output_dir}/")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()