"""
Helper functions for handling class imbalance in logistic regression
"""

import numpy as np
import pandas as pd


def apply_smote(X_train, y_train):
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique)
    to balance the training data.

    Requires: pip install imbalanced-learn
    """
    try:
        from imblearn.over_sampling import SMOTE

        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

        print(f"   Original class distribution: {np.bincount(y_train)}")
        print(f"   After SMOTE: {np.bincount(y_resampled)}")

        return X_resampled, y_resampled
    except ImportError:
        print(
            "⚠️  Warning: imbalanced-learn not installed. Run: pip install imbalanced-learn"
        )
        return X_train, y_train


def apply_random_undersampling(X_train, y_train):
    """
    Randomly remove majority class samples to balance the dataset.
    """
    try:
        from imblearn.under_sampling import RandomUnderSampler

        rus = RandomUnderSampler(random_state=42)
        X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

        print(f"   Original class distribution: {np.bincount(y_train)}")
        print(f"   After undersampling: {np.bincount(y_resampled)}")

        return X_resampled, y_resampled
    except ImportError:
        print(
            "⚠️  Warning: imbalanced-learn not installed. Run: pip install imbalanced-learn"
        )
        return X_train, y_train


def find_optimal_threshold(y_test, y_pred_proba, metric="f1"):
    """
    Find the optimal classification threshold by maximizing a metric.

    Parameters:
    -----------
    y_test : array
        True labels
    y_pred_proba : array
        Predicted probabilities for class 1
    metric : str
        'f1', 'recall', 'precision', or 'balanced'

    Returns:
    --------
    optimal_threshold : float
        Best threshold value
    """
    from sklearn.metrics import precision_recall_curve, f1_score

    thresholds = np.linspace(0.1, 0.9, 81)  # Test 81 thresholds
    best_score = 0
    best_threshold = 0.5

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)

        if metric == "f1":
            score = f1_score(y_test, y_pred, zero_division=0)
        elif metric == "recall":
            from sklearn.metrics import recall_score

            score = recall_score(y_test, y_pred, zero_division=0)
        elif metric == "precision":
            from sklearn.metrics import precision_score

            score = precision_score(y_test, y_pred, zero_division=0)
        elif metric == "balanced":
            from sklearn.metrics import precision_score, recall_score

            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            score = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0

        if score > best_score:
            best_score = score
            best_threshold = threshold

    print(
        f"   Optimal threshold for {metric}: {best_threshold:.3f} (score: {best_score:.4f})"
    )
    return best_threshold
