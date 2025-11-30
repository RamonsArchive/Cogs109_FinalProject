"""
Professional Neural Network Model for NFL Injury Prediction
Mirrors the structure of the gradient boosting implementation with proper:
- Train/test split (no data leakage)
- Hyperparameter tuning with GridSearchCV
- Cross-validation on training data only
- Final evaluation on held-out test set
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

# Import your existing functions
from load import load_data
from clean import clean_data
from graph_neural_net import graph_neural_net_model

RANDOM_STATE = 42


def create_preprocessing_pipeline(predictors, categorical_features):
    """
    Create preprocessing pipeline with proper handling of categorical variables
    
    Args:
        predictors: List of all predictor column names
        categorical_features: List of categorical column names
        
    Returns:
        ColumnTransformer for preprocessing
    """
    # Separate categorical and numerical features
    numerical_features = [f for f in predictors if f not in categorical_features]
    
    # Create preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
             categorical_features)
        ],
        remainder='passthrough'
    )
    
    return preprocessor


def tune_neural_network_hyperparameters(
    train_df,
    predictors,
    categorical_features,
    is_log_target=False,
    hidden_layer_sizes_grid=None,
    learning_rate_init_grid=None,
    alpha_grid=None,
    batch_size_grid=None,
    max_iter=5000
):
    """
    Perform grid search with cross-validation to find best hyperparameters
    
    HYPERPARAMETERS EXPLAINED:
    
    1. hidden_layer_sizes: Architecture of the network
       - Tuple like (100, 50) means 2 hidden layers with 100 and 50 neurons
       - Deeper = more complex patterns, but risk of overfitting
       - Wider = more capacity per layer
       
    2. learning_rate_init: Initial learning rate for weight updates
       - Controls how big steps the optimizer takes
       - Too high = unstable training, overshooting minima
       - Too low = very slow convergence
       - Typical range: 0.0001 to 0.01
       
    3. alpha: L2 regularization parameter (ridge penalty)
       - Penalizes large weights to prevent overfitting
       - Higher alpha = stronger regularization = simpler model
       - alpha=0 means no regularization
       - Typical range: 0.0001 to 0.01
       
    4. batch_size: Number of samples per gradient update
       - Smaller batches = more updates, noisier gradients, better generalization
       - Larger batches = faster training, more stable gradients
       - 'auto' = min(200, n_samples)
       - Common choices: 32, 64, 128, 256
       
    5. max_iter: Maximum number of iterations/epochs
       - How many times to go through the entire dataset
       - Use early_stopping to prevent wasting time
       
    OTHER IMPORTANT PARAMETERS (set to good defaults):
    
    6. activation='relu': Activation function for hidden layers
       - 'relu': Rectified Linear Unit, most common, f(x) = max(0, x)
       - 'tanh': Hyperbolic tangent, outputs [-1, 1]
       - 'logistic': Sigmoid function, outputs [0, 1]
       
    7. solver='adam': Optimization algorithm
       - 'adam': Adaptive Moment Estimation, best for large datasets
       - 'sgd': Stochastic Gradient Descent, requires tuning learning rate
       - 'lbfgs': Good for small datasets
       
    8. early_stopping=True: Stop when validation score stops improving
       - Prevents overfitting and wasted computation
       - Monitors validation loss and stops if no improvement
       
    9. validation_fraction=0.1: Portion of training data for early stopping
       - 10% of training data used to monitor convergence
       - This is SEPARATE from cross-validation
       
    10. n_iter_no_change=10: Patience for early stopping
        - Stop if no improvement for 10 consecutive epochs
        
    11. tol=1e-4: Tolerance for optimization
        - If improvement < tol, counts toward n_iter_no_change
    """
    
    # Default grid values if not provided
    if hidden_layer_sizes_grid is None:
        hidden_layer_sizes_grid = [
            (50,),           # 1 layer, 50 neurons (simple)
            (100,),          # 1 layer, 100 neurons
            (100, 50),       # 2 layers, decreasing size
            (100, 100),      # 2 layers, same size
            (200, 100),      # 2 layers, wider
            (100, 50, 25),   # 3 layers, decreasing (deeper)
        ]
    
    if learning_rate_init_grid is None:
        learning_rate_init_grid = [0.0001, 0.001, 0.01]
    
    if alpha_grid is None:
        alpha_grid = [0.0001, 0.001, 0.01, 0.1]
    
    if batch_size_grid is None:
        batch_size_grid = [32, 64, 128, 'auto']
    
    # Prepare data
    X_train = train_df[predictors].copy()
    
    if is_log_target:
        # Log transform target (add 1 to handle zeros)
        y_train = np.log1p(train_df["num_injuries"].values)
        print("Target: log(num_injuries + 1)")
    else:
        y_train = train_df["num_injuries"].values
        print("Target: num_injuries (count)")
    
    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(predictors, categorical_features)
    
    # Create base neural network
    base_nn = MLPRegressor(
        activation='relu',           # ReLU activation (best for most cases)
        solver='adam',               # Adam optimizer (adaptive learning rate)
        early_stopping=True,         # Stop when validation loss stops improving
        validation_fraction=0.1,     # 10% of training for early stopping validation
        n_iter_no_change=20,         # Patience: stop if no improvement for 20 epochs
        tol=1e-4,                    # Tolerance for improvement
        max_iter=max_iter,           # Maximum epochs
        random_state=RANDOM_STATE,
        verbose=False
    )
    
    # Create full pipeline
    base_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', base_nn)
    ])
    
    # Define parameter grid
    param_grid = {
        'model__hidden_layer_sizes': hidden_layer_sizes_grid,
        'model__learning_rate_init': learning_rate_init_grid,
        'model__alpha': alpha_grid,
        'model__batch_size': batch_size_grid,
    }
    
    # Calculate total combinations
    total_combinations = (
        len(hidden_layer_sizes_grid) * 
        len(learning_rate_init_grid) * 
        len(alpha_grid) * 
        len(batch_size_grid)
    )
    
    print(f"\n🔍 Grid Search Configuration:")
    print(f"   • Hidden layer sizes: {len(hidden_layer_sizes_grid)} options")
    print(f"   • Learning rates: {len(learning_rate_init_grid)} options")
    print(f"   • Alpha (L2 reg): {len(alpha_grid)} options")
    print(f"   • Batch sizes: {len(batch_size_grid)} options")
    print(f"   • Total combinations: {total_combinations}")
    print(f"   • CV folds: 10")
    print(f"   • Total model fits: {total_combinations * 10}")
    
    # Grid search with 10-fold CV
    grid_search = GridSearchCV(
        base_pipeline,
        param_grid,
        cv=10,
        scoring="neg_mean_squared_error",
        n_jobs=-1,  # Use all CPU cores
        verbose=2,
        return_train_score=True,
    )
    
    print(f"\n⏳ Starting grid search (this will take a while)...")
    grid_search.fit(X_train, y_train)
    
    # Get best parameters
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_  # Convert back to positive MSE
    
    print(f"\n✅ Grid search complete!")
    print(f"Best CV MSE: {best_score:.4f}")
    print(f"Best CV RMSE: {np.sqrt(best_score):.4f}")
    print(f"Best hyperparameters:")
    print(f"  • hidden_layer_sizes: {best_params['model__hidden_layer_sizes']}")
    print(f"  • learning_rate_init: {best_params['model__learning_rate_init']}")
    print(f"  • alpha (L2 reg): {best_params['model__alpha']}")
    print(f"  • batch_size: {best_params['model__batch_size']}")
    
    # Get top 5 combinations
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df['mean_test_rmse'] = np.sqrt(-results_df['mean_test_score'])
    top_5 = results_df.nsmallest(5, 'mean_test_rmse')[[
        'param_model__hidden_layer_sizes', 
        'param_model__learning_rate_init',
        'param_model__alpha',
        'param_model__batch_size',
        'mean_test_rmse', 
        'std_test_score'
    ]]
    
    print(f"\n📊 Top 5 hyperparameter combinations:")
    print(top_5.to_string(index=False))
    
    return {
        "best_hidden_layer_sizes": best_params['model__hidden_layer_sizes'],
        "best_learning_rate_init": best_params['model__learning_rate_init'],
        "best_alpha": best_params['model__alpha'],
        "best_batch_size": best_params['model__batch_size'],
        "best_cv_mse": best_score,
        "best_cv_rmse": np.sqrt(best_score),
        "grid_search": grid_search,
        "results_df": results_df,
    }


def train_final_model(
    train_df,
    test_df,
    predictors,
    categorical_features,
    results,
    model_name,
    is_log_target=False,
    hidden_layer_sizes=(100, 50),
    learning_rate_init=0.001,
    alpha=0.0001,
    batch_size='auto',
    max_iter=5000
):
    """
    Train final neural network model with best hyperparameters
    Evaluate with 10-fold CV on training data, then on held-out test set
    """
    
    # Prepare training data
    X_train = train_df[predictors].copy()
    
    if is_log_target:
        y_train = np.log1p(train_df["num_injuries"].values)
        y_test = np.log1p(test_df["num_injuries"].values)
    else:
        y_train = train_df["num_injuries"].values
        y_test = test_df["num_injuries"].values
    
    X_test = test_df[predictors].copy()
    
    # Create preprocessing and model pipeline
    preprocessor = create_preprocessing_pipeline(predictors, categorical_features)
    
    nn_model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        learning_rate_init=learning_rate_init,
        alpha=alpha,
        batch_size=batch_size,
        activation='relu',
        solver='adam',
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        tol=1e-4,
        max_iter=max_iter,
        random_state=RANDOM_STATE,
        verbose=False
    )
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', nn_model)
    ])
    
    # ========================================================================
    # 10-FOLD CROSS-VALIDATION ON TRAINING DATA
    # ========================================================================
    print(f"\n📊 Performing 10-fold CV on training data ({len(train_df)} samples)...")
    
    cv_results = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=10,
        scoring=['neg_mean_squared_error', 'r2'],
        return_train_score=True,
        n_jobs=-1
    )
    
    cv_train_mse = -cv_results['train_neg_mean_squared_error'].mean()
    cv_val_mse = -cv_results['test_neg_mean_squared_error'].mean()
    cv_train_r2 = cv_results['train_r2'].mean()
    cv_val_r2 = cv_results['test_r2'].mean()
    
    print(f"✅ 10-Fold CV Results:")
    print(f"   • Training RMSE: {np.sqrt(cv_train_mse):.4f}")
    print(f"   • Validation RMSE: {np.sqrt(cv_val_mse):.4f}")
    print(f"   • Training R²: {cv_train_r2:.4f}")
    print(f"   • Validation R²: {cv_val_r2:.4f}")
    
    # ========================================================================
    # TRAIN ON FULL TRAINING SET AND EVALUATE ON TEST SET
    # ========================================================================
    print(f"\n🎯 Training final model on full training set...")
    pipeline.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    
    # If log-transformed, convert back to original scale
    if is_log_target:
        y_train_pred = np.expm1(y_train_pred)  # Inverse of log1p
        y_test_pred = np.expm1(y_test_pred)
        y_train_actual = np.expm1(y_train)
        y_test_actual = np.expm1(y_test)
    else:
        y_train_actual = y_train
        y_test_actual = y_test
    
    # Calculate final metrics
    train_mse = mean_squared_error(y_train_actual, y_train_pred)
    test_mse = mean_squared_error(y_test_actual, y_test_pred)
    train_r2 = r2_score(y_train_actual, y_train_pred)
    test_r2 = r2_score(y_test_actual, y_test_pred)
    
    print(f"\n✅ Final Model Performance:")
    print(f"   • Train RMSE: {np.sqrt(train_mse):.4f}")
    print(f"   • Test RMSE: {np.sqrt(test_mse):.4f}")
    print(f"   • Train R²: {train_r2:.4f}")
    print(f"   • Test R²: {test_r2:.4f}")
    
    # Get info about neural network training
    final_model = pipeline.named_steps['model']
    print(f"\n📈 Training Details:")
    print(f"   • Iterations completed: {final_model.n_iter_}")
    print(f"   • Final loss: {final_model.loss_:.6f}")
    print(f"   • Network architecture: {hidden_layer_sizes}")
    print(f"   • Total parameters: {sum(w.size for w in final_model.coefs_)}")
    
    # Store results
    # Combine train_df and test_df for full dataset (for plotting)
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    
    results[model_name] = {
        "pipeline": pipeline,
        "model": final_model,
        "10foldCV": {
            "train_mse": cv_train_mse,
            "val_mse": cv_val_mse,
            "train_r2": cv_train_r2,
            "val_r2": cv_val_r2,
        },
        "final_test": {
            "train_mse": train_mse,
            "test_mse": test_mse,
            "train_r2": train_r2,
            "test_r2": test_r2,
        },
        "predictions": {
            "y_train_actual": y_train_actual,
            "y_train_pred": y_train_pred,
            "y_test_actual": y_test_actual,
            "y_test_pred": y_test_pred,
        },
        "is_log_target": is_log_target,
        "df": full_df,
        "predictors": predictors,
    }
    
    return results[model_name]


def main():
    """
    Main function for Neural Network models
    
    Uses Model 6 (kitchen sink) predictors
    Neural networks benefit from feature scaling (done automatically in pipeline)
    """
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(
        curr_dir,
        "../datasets/df_master_schedule_injury_surface_2019_23_weather_distance_days_since_last_game.csv",
    )

    # Model 6: Kitchen sink (all potentially relevant features)
    PREDICTORS = [
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
    
    # Categorical features (will be one-hot encoded)
    CATEGORICAL_FEATURES = [
        "surface_type",
        "day",
        "season",
        "stadium",
        "surface",
        "dome"
    ]

    print("\n" + "=" * 70)
    print("NEURAL NETWORK MODELS")
    print("=" * 70)
    print(f"Using {len(PREDICTORS)} predictors from Model 6 (kitchen sink)")
    print(f"  • {len(CATEGORICAL_FEATURES)} categorical features (one-hot encoded)")
    print(f"  • {len(PREDICTORS) - len(CATEGORICAL_FEATURES)} numerical features (standardized)")
    print("Neural networks require proper feature scaling and encoding")
    print("=" * 70)

    # Load and prepare data
    df = load_data(data_dir)
    df = clean_data(df)

    # ========================================================================
    # STEP 1: ONE SPLIT - Create held-out test set (NEVER touch until end)
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 1: DATA SPLITTING")
    print("=" * 70)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
    print(f"📊 Outer split: Train={len(train_df)} ({len(train_df)/len(df)*100:.1f}%), Test={len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    print("⚠️  Test set will NOT be used for hyperparameter tuning!")
    print("⚠️  Test set will ONLY be used for final unbiased evaluation at the end")

    # ========================================================================
    # STEP 2: GRID SEARCH WITH CV ON TRAINING DATA ONLY
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 2: HYPERPARAMETER TUNING WITH GRID SEARCH")
    print("=" * 70)
    print("Grid search performs 10-fold CV WITHIN the training set (80%)")
    print("The test set (20%) is never touched during this step")
    
    # Define grid search parameters
    HIDDEN_LAYER_SIZES_GRID = [
        (50,),           # Simple: 1 layer, 50 neurons
        (100,),          # Simple: 1 layer, 100 neurons
        (100, 50),       # Medium: 2 layers
        (100, 100),      # Medium: 2 layers, same size
        (200, 100),      # Wider: 2 layers
        (100, 50, 25),   # Deep: 3 layers
    ]
    LEARNING_RATE_INIT_GRID = [0.0001, 0.001, 0.01]
    ALPHA_GRID = [0.0001, 0.001, 0.01, 0.1]  # L2 regularization
    BATCH_SIZE_GRID = [32, 64, 128, 'auto']
    
    # Tune hyperparameters for COUNT model (on training data only)
    print("\n" + "=" * 70)
    print("COUNT MODEL: Hyperparameter Tuning")
    print("=" * 70)
    count_tuning = tune_neural_network_hyperparameters(
        train_df=train_df,
        predictors=PREDICTORS,
        categorical_features=CATEGORICAL_FEATURES,
        is_log_target=False,
        hidden_layer_sizes_grid=HIDDEN_LAYER_SIZES_GRID,
        learning_rate_init_grid=LEARNING_RATE_INIT_GRID,
        alpha_grid=ALPHA_GRID,
        batch_size_grid=BATCH_SIZE_GRID,
    )
    
    # ========================================================================
    # STEP 3: TRAIN FINAL MODELS WITH BEST HYPERPARAMETERS
    # ========================================================================
    print("\n" + "=" * 70)
    print("STEP 3: TRAINING FINAL MODELS WITH BEST HYPERPARAMETERS")
    print("=" * 70)
    
    # ========================================================================
    # NEURAL NETWORK ON COUNT DATA (Final Model)
    # ========================================================================
    print("\n" + "=" * 70)
    print("FINAL COUNT MODEL")
    print("=" * 70)
    print(f"Best hyperparameters from grid search:")
    print(f"  • hidden_layer_sizes: {count_tuning['best_hidden_layer_sizes']}")
    print(f"  • learning_rate_init: {count_tuning['best_learning_rate_init']}")
    print(f"  • alpha (L2 reg): {count_tuning['best_alpha']}")
    print(f"  • batch_size: {count_tuning['best_batch_size']}")
    print(f"  • Best CV RMSE: {count_tuning['best_cv_rmse']:.4f}")
    
    count_results = {}
    
    count_model = train_final_model(
        train_df=train_df,
        test_df=test_df,
        predictors=PREDICTORS,
        categorical_features=CATEGORICAL_FEATURES,
        results=count_results,
        model_name="nn_count",
        is_log_target=False,
        hidden_layer_sizes=count_tuning['best_hidden_layer_sizes'],
        learning_rate_init=count_tuning['best_learning_rate_init'],
        alpha=count_tuning['best_alpha'],
        batch_size=count_tuning['best_batch_size'],
    )
    
    graph_neural_net_model(count_model, "nn_count", is_best=False)

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("✅ NEURAL NETWORK TRAINING COMPLETE!")
    print("=" * 70)
    
    print(f"\n{'='*70}")
    print("📊 FINAL RESULTS SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n🧠 COUNT MODEL - Best Hyperparameters:")
    print(f"   • Architecture: {count_tuning['best_hidden_layer_sizes']}")
    print(f"   • Learning rate: {count_tuning['best_learning_rate_init']}")
    print(f"   • Alpha (L2): {count_tuning['best_alpha']}")
    print(f"   • Batch size: {count_tuning['best_batch_size']}")
    print(f"   • Grid Search Best CV RMSE: {count_tuning['best_cv_rmse']:.4f}")
    print(f"\n   Final Model Performance:")
    print(f"   • CV R²: {count_model['10foldCV']['val_r2']:.4f}")
    print(f"   • Test R²: {count_model['final_test']['test_r2']:.4f}")
    print(f"   • CV RMSE: {np.sqrt(count_model['10foldCV']['val_mse']):.4f}")
    print(f"   • Test RMSE: {np.sqrt(count_model['final_test']['test_mse']):.4f}")
    
    print(f"\n📁 All plots saved to: plots/neural_net/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()