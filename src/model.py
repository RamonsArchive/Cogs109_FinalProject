import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split,
    ShuffleSplit,
    LeaveOneOut,
    StratifiedKFold,
    cross_validate,
    KFold,
)
from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
    mean_poisson_deviance,
)
from sklearn.linear_model import PoissonRegressor

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def make_pipeline(num_cols, cat_cols):
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    # PoissonRegressor: no solver/random_state; ensure enough iterations
    clf = PoissonRegressor(max_iter=5000)
    return Pipeline([("preprocessor", pre), ("model", clf)])


# use Poisson Regression to predict the number of injuries
def train_model_1(df: pd.DataFrame):

    results = {}
    predictors = [
        "surface_type",
        "Avg_Temp",
        "day",
        "week",
        "season",
        "stadium",
        "surface",
        "dome",
    ]
    outcomes = ["num_injuries"]
    target = outcomes[0]  # grabs "num_injuries"

    num_cols_all = [c for c in predictors if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols_all = [c for c in predictors if not pd.api.types.is_numeric_dtype(df[c])]
    print("Data shape in model 1: ", df.shape)
    print("Number of numeric columns: ", len(num_cols_all))
    print("Number of categorical columns: ", len(cat_cols_all))

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)

    X_train = train_df[predictors]
    y_train = train_df[target].astype(
        int
    )  # train set that can be further split into train and validation set
    X_test = test_df[predictors]
    y_test = test_df[target].astype(int)

    print("Train size:", train_df.shape, "| Test size:", test_df.shape)
    print(train_df.head())

    # now we use 10-fold cross validation to evaluate the model
    pipeline = make_pipeline(num_cols_all, cat_cols_all)  # fit model
    cv = KFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    # Use appropriate regression metrics; Poisson deviance is most aligned
    scoring = {
        "neg_poisson_dev": "neg_mean_poisson_deviance",
        "neg_mse": "neg_mean_squared_error",
        "neg_mae": "neg_mean_absolute_error",
    }
    cv_out = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring)

    mean_poisson_dev = -np.mean(cv_out["test_neg_poisson_dev"])
    mean_mse = -np.mean(cv_out["test_neg_mse"])
    mean_mae = -np.mean(cv_out["test_neg_mae"])

    print(f"[CV] Mean Poisson deviance: {mean_poisson_dev:.4f}")
    print(f"[CV] RMSE: {np.sqrt(mean_mse):.4f}   (MSE: {mean_mse:.4f})")
    print(f"[CV] MAE : {mean_mae:.4f}")

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    test_mse = mean_squared_error(y_test, y_pred)
    test_mae = mean_absolute_error(y_test, y_pred)
    test_poisson_dev = mean_poisson_deviance(y_test, np.clip(y_pred, 1e-9, None))

    print(f"[Test] Poisson deviance: {test_poisson_dev:.4f}")
    print(f"[Test] RMSE: {np.sqrt(test_mse):.4f}   (MSE: {test_mse:.4f})")
    print(f"[Test] MAE : {test_mae:.4f}")

    results["10foldCV"] = {
        "val_poisson_dev": mean_poisson_dev,
        "val_mse": mean_mse,
        "val_mae": mean_mae,
        "test_poisson_dev": test_poisson_dev,
        "test_mse": test_mse,
        "test_mae": test_mae,
    }
    return results
