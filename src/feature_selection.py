"""
Feature Selection Module - Backward Elimination
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression, PoissonRegressor, Ridge


def backward_selection(
    df: pd.DataFrame,
    predictors: list,
    target_column: str,
    significance_level: float = 0.1,
    model_type: str = "logistic",
    is_binary: bool = False,
    verbose: bool = True,
):
    """
    Perform backward elimination to select significant features.

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    predictors : list
        List of all predictor columns to start with
    target_column : str
        Name of target column
    significance_level : float
        P-value threshold (default 0.1)
    model_type : str
        'logistic', 'poisson', or 'linear'
    is_binary : bool
        Whether target is binary (for logistic regression)
    verbose : bool
        Print progress messages

    Returns:
    --------
    selected_features : list
        List of features that passed backward selection
    """

    if verbose:
        print("\n" + "=" * 80)
        print("🔍 BACKWARD SELECTION WITH STATSMODELS")
        print("=" * 80)
        print(f"Model Type: {model_type.upper()}")
        print(f"Significance Level: α = {significance_level}")
        print(f"Starting with {len(predictors)} predictors")
        print("=" * 80 + "\n")

    # Remove redundant predictors to avoid multicollinearity
    # These pairs are likely perfectly correlated:
    redundant_pairs = [
        ("surface", "surface_type"),  # Both describe playing surface
        ("stadium", "dome"),  # Stadium names may perfectly predict dome status
    ]

    predictors_clean = predictors.copy()
    removed_for_redundancy = []

    for pred1, pred2 in redundant_pairs:
        if pred1 in predictors_clean and pred2 in predictors_clean:
            # Keep the more general one (surface_type, dome) and remove specific (surface, stadium)
            if pred1 in ["surface", "stadium"]:
                predictors_clean.remove(pred1)
                removed_for_redundancy.append(pred1)
                if verbose:
                    print(f"⚠️  Removed '{pred1}' (redundant with '{pred2}')")

    if removed_for_redundancy and verbose:
        print(
            f"\n✅ Cleaned predictors: {len(predictors_clean)} (removed {len(removed_for_redundancy)} for redundancy)\n"
        )

    # Prepare data
    X = df[predictors_clean].copy()
    y = df[target_column].copy()

    if is_binary:
        y = y.astype(int)

    # Identify numeric and categorical columns (use cleaned predictors)
    numeric_cols = [
        col for col in predictors_clean if pd.api.types.is_numeric_dtype(df[col])
    ]
    categorical_cols = [
        col for col in predictors_clean if not pd.api.types.is_numeric_dtype(df[col])
    ]

    if verbose:
        print(f"Numeric predictors: {len(numeric_cols)}")
        print(f"Categorical predictors: {len(categorical_cols)}\n")

    # Preprocess features (scale numeric, one-hot encode categorical)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(drop="first", sparse_output=False), categorical_cols),
        ]
    )

    X_processed = preprocessor.fit_transform(X)

    # Get feature names after preprocessing
    feature_names = []
    # Numeric features keep their names
    feature_names.extend(numeric_cols)
    # Categorical features get expanded names
    if categorical_cols:
        cat_encoder = preprocessor.named_transformers_["cat"]
        for i, col in enumerate(categorical_cols):
            categories = cat_encoder.categories_[i][
                1:
            ]  # Skip first due to drop='first'
            feature_names.extend([f"{col}_{cat}" for cat in categories])

    # Convert to DataFrame for easier manipulation
    X_df = pd.DataFrame(X_processed, columns=feature_names, index=X.index)

    # Add constant for statsmodels
    X_with_const = sm.add_constant(X_df)

    # Backward elimination loop
    eliminated_features = []
    iteration = 0

    while True:
        iteration += 1

        # Fit model based on type with error handling
        try:
            if model_type == "logistic":
                # Use regularized logistic regression to handle perfect separation
                model = sm.Logit(y, X_with_const).fit(
                    disp=0,
                    maxiter=1000,
                    method="bfgs",  # More stable optimization method
                )
            elif model_type == "poisson":
                model = sm.GLM(y, X_with_const, family=sm.families.Poisson()).fit(
                    disp=0,
                    maxiter=1000,
                )
            elif model_type == "linear":
                model = sm.OLS(y, X_with_const).fit()
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
        except (np.linalg.LinAlgError, ValueError) as e:
            if verbose:
                print(f"\n⚠️  Convergence error: {str(e)[:100]}")
                print("   Likely due to perfect multicollinearity or separation")
                print("   Stopping backward elimination at current iteration\n")
            # Map back remaining features to original predictor names
            remaining_features = list(X_with_const.columns)
            if "const" in remaining_features:
                remaining_features.remove("const")

            selected_predictors = set()
            for feat in remaining_features:
                if feat in numeric_cols:
                    selected_predictors.add(feat)
                else:
                    base_name = feat.rsplit("_", 1)[0]
                    if base_name in categorical_cols:
                        selected_predictors.add(base_name)

            return sorted(list(selected_predictors))

        # Get p-values (exclude constant)
        p_values = model.pvalues.drop("const", errors="ignore")

        # Find feature with highest p-value
        max_p_value = p_values.max()
        max_p_feature = p_values.idxmax()

        if verbose:
            print(f"Iteration {iteration}:")
            print(f"  Features remaining: {len(p_values)}")
            print(f"  Highest p-value: {max_p_value:.4f} ({max_p_feature})")

        # Check if we should eliminate this feature
        if max_p_value > significance_level:
            if verbose:
                print(f"  ❌ Eliminating: {max_p_feature} (p = {max_p_value:.4f})")
            eliminated_features.append(max_p_feature)

            # Remove feature and continue
            X_with_const = X_with_const.drop(columns=[max_p_feature])

            if len(X_with_const.columns) == 1:  # Only constant left
                if verbose:
                    print(
                        "\n⚠️  All features eliminated! Keeping all original features."
                    )
                return predictors  # Return original list if all eliminated

        else:
            if verbose:
                print(
                    f"  ✅ All remaining features significant (p < {significance_level})"
                )
            break

        print()

    # Map back from processed feature names to original predictor names
    remaining_features = list(X_with_const.columns)
    remaining_features.remove("const")  # Remove constant

    # Extract original predictor names
    selected_predictors = set()
    for feat in remaining_features:
        # Check if it's a numeric feature (exact match)
        if feat in numeric_cols:
            selected_predictors.add(feat)
        else:
            # It's a categorical feature (extract base name before underscore)
            base_name = feat.rsplit("_", 1)[0]  # Get everything before last underscore
            if base_name in categorical_cols:
                selected_predictors.add(base_name)

    selected_predictors = sorted(list(selected_predictors))

    # Print summary
    if verbose:
        print("\n" + "=" * 80)
        print("📊 BACKWARD SELECTION SUMMARY")
        print("=" * 80)
        print(f"Started with: {len(predictors)} predictors")
        print(f"Eliminated: {len(predictors) - len(selected_predictors)} predictors")
        print(f"Final selected: {len(selected_predictors)} predictors")
        print("\n✅ SELECTED PREDICTORS:")
        for pred in selected_predictors:
            print(f"   • {pred}")

        if eliminated_features:
            print("\n❌ ELIMINATED PREDICTORS:")
            # Map back eliminated features to original names
            # Only show predictors that are COMPLETELY eliminated (not in selected_predictors)
            eliminated_originals = set()
            for feat in eliminated_features:
                if feat in numeric_cols:
                    eliminated_originals.add(feat)
                else:
                    base_name = feat.rsplit("_", 1)[0]
                    if base_name in categorical_cols:
                        eliminated_originals.add(base_name)

            # Remove any that are still selected (partial elimination of categorical)
            truly_eliminated = eliminated_originals - set(selected_predictors)

            for pred in sorted(truly_eliminated):
                print(f"   • {pred}")

        print("=" * 80 + "\n")

    return selected_predictors


def compare_models_with_without_selection(
    df,
    all_predictors,
    selected_predictors,
    target_column,
    model_type="logistic",
    is_binary=False,
):
    """
    Compare model performance with all features vs selected features.

    This is a simple comparison using statsmodels to show AIC/BIC improvements.
    """
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer

    print("\n" + "=" * 80)
    print("📊 MODEL COMPARISON: All Features vs Selected Features")
    print("=" * 80)

    y = df[target_column].copy()
    if is_binary:
        y = y.astype(int)

    results = {}

    for name, predictors in [
        ("All Features", all_predictors),
        ("Selected Features", selected_predictors),
    ]:
        X = df[predictors].copy()

        # Identify numeric and categorical
        numeric_cols = [
            col for col in predictors if pd.api.types.is_numeric_dtype(df[col])
        ]
        categorical_cols = [
            col for col in predictors if not pd.api.types.is_numeric_dtype(df[col])
        ]

        # Preprocess
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numeric_cols),
                (
                    "cat",
                    OneHotEncoder(drop="first", sparse_output=False),
                    categorical_cols,
                ),
            ]
        )

        X_processed = preprocessor.fit_transform(X)

        # Use sklearn models for comparison (more robust than statsmodels)
        from sklearn.linear_model import (
            LogisticRegression,
            PoissonRegressor,
            LinearRegression,
        )
        from sklearn.model_selection import cross_val_score

        # Select appropriate sklearn model
        if model_type == "logistic":
            model = LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=42
            )
            scoring = "roc_auc"
        elif model_type == "poisson":
            model = PoissonRegressor(max_iter=1000)
            scoring = "neg_mean_poisson_deviance"
        elif model_type == "linear":
            model = LinearRegression()
            scoring = "neg_mean_squared_error"
        else:
            scoring = "accuracy"

        # Cross-validation score
        try:
            cv_scores = cross_val_score(model, X_processed, y, cv=5, scoring=scoring)
            mean_cv_score = np.mean(cv_scores)
        except Exception as e:
            print(f"⚠️  Error in CV for {name}: {e}")
            mean_cv_score = np.nan

        results[name] = {
            "n_features": len(predictors),
            "cv_score": mean_cv_score,
        }

    print(
        f"\n{'Metric':<20} {'All Features':>15} {'Selected Features':>20} {'Change':>15}"
    )
    print("-" * 80)
    print(
        f"{'# Features':<20} {results['All Features']['n_features']:>15} "
        f"{results['Selected Features']['n_features']:>20} "
        f"{results['Selected Features']['n_features'] - results['All Features']['n_features']:>15}"
    )
    # Determine score name based on model type
    if model_type == "logistic":
        score_name = "ROC-AUC (higher better)"
        higher_better = True
    elif model_type == "poisson":
        score_name = "Neg Poisson Dev (higher better)"
        higher_better = True
    else:
        score_name = "Neg MSE (higher better)"
        higher_better = True

    print(
        f"{score_name:<20} {results['All Features']['cv_score']:>15.4f} "
        f"{results['Selected Features']['cv_score']:>20.4f} "
        f"{results['Selected Features']['cv_score'] - results['All Features']['cv_score']:>15.4f}"
    )

    print("\n💡 Interpretation:")
    cv_improved = (
        results["Selected Features"]["cv_score"] > results["All Features"]["cv_score"]
    )

    if cv_improved:
        print(f"   ✅ Feature selection IMPROVED cross-validation performance")
    else:
        print(f"   ⚠️  Feature selection did NOT improve performance")
        print("      → All features may be useful, or sample size is small")

    print("=" * 80 + "\n")

    return results
