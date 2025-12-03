# NFL Injury Prediction Analysis

## Project Overview

This project investigates the predictability of NFL game injuries using various machine learning and statistical modeling approaches. The analysis explores multiple modeling frameworks to understand which factors (weather, game intensity, rest days, travel distance, etc.) best predict the number of injuries in NFL games from 2019-2023.

**Key Finding:** Despite employing robust modeling techniques with extensive hyperparameter tuning and feature selection, the dataset proved too noisy to achieve strong predictive capabilities for the number of injuries. This suggests that injury occurrence in NFL games is highly stochastic and may be influenced by factors not captured in the available data.

---

## Author

**Ramon McDargh-Mitchell**

- **Email:** cltuchdev.apps@gmail.com
- **Portfolio:** https://clutchstudio.dev

---

## Acknowledgments

**Sherry Dela Cruz** (sgdelacruz@ucsd.edu) contributed the polynomial regression model implementation and provided valuable assistance throughout the project.

---

## File Structure

```
my-project/
├── datasets/                          # Data files
│   ├── df_master_schedule_injury_surface_2019_23_weather_distance_days_since_last_game.csv
│   └── contact_noncontact_2019_2020_full_seasons.csv
│
├── src/                               # Source code
│   ├── main.py                        # Main script - trains all linear models
│   ├── model.py                       # Core model training functions
│   ├── load.py                        # Data loading utilities
│   ├── clean.py                       # Data cleaning and preprocessing
│   ├── feature_selection.py          # Backward selection implementation
│   ├── graph.py                       # Visualization functions for linear models
│   │
│   ├── elastic_net.py                 # Elastic Net regression models
│   ├── elastic_graphs.py              # Elastic Net visualization
│   │
│   ├── boosting.py                    # Gradient Boosting models
│   ├── graph_boosting.py              # Boosting visualization
│   │
│   ├── random_forest.py               # Random Forest models
│   ├── graph_random_forest.py         # Random Forest visualization
│   │
│   ├── neural_net.py                  # Neural Network (MLP) models
│   ├── graph_neural_net.py            # Neural Network visualization
│   │
│   ├── polytest.py                    # Polynomial regression models (by Sherry Dela Cruz)
│   │
│   └── exploratory.py                 # Exploratory data analysis
│
└── plots/                             # Generated visualizations and reports
    ├── best/                          # Best model results
    ├── non_best/                      # All model variants
    ├── boosting/                      # Gradient Boosting results
    ├── elastic/                       # Elastic Net results
    ├── random_forest/                 # Random Forest results
    ├── neural_net/                    # Neural Network results
    ├── polynomial/                    # Polynomial regression results
    └── exploratory/                   # EDA visualizations
```

---

## Models Implemented

### 1. **Poisson Regression**
- **Purpose:** Models count data (number of injuries) using Poisson distribution
- **Features:** 
  - Evaluated using Poisson Deviance and Pseudo-R² (McFadden's)
  - Handles overdispersion in count data
- **Implementation:** `src/model.py`, `src/main.py`

### 2. **Linear Regression (OLS) with Log-Transformed Target**
- **Purpose:** Models log-transformed injury counts using ordinary least squares
- **Features:**
  - Uses R² and Adjusted R² for evaluation
  - Transforms predictions back to original scale for interpretation
- **Implementation:** `src/model.py`, `src/main.py`

### 3. **Logistic Regression (Binary Classification)**
- **Purpose:** Predicts high-injury games (1 if num_injuries > 4, else 0)
- **Features:**
  - Uses ROC-AUC, Accuracy, Precision, Recall, F1-Score
  - Includes naive baseline comparison (majority class predictor)
  - Class-weighted to handle class imbalance
- **Implementation:** `src/model.py`, `src/main.py`

### 4. **Elastic Net Regression**
- **Purpose:** Combines L1 (Lasso) and L2 (Ridge) regularization
- **Features:**
  - Automatic feature selection via L1 regularization
  - Grid search for optimal alpha and L1 ratio
  - Coefficient path visualization
  - Supports both Poisson and Linear variants
- **Implementation:** `src/elastic_net.py`, `src/elastic_graphs.py`

### 5. **Gradient Boosting (XGBoost-style)**
- **Purpose:** Ensemble method using gradient boosting for regression
- **Features:**
  - Grid search with 10-fold cross-validation
  - Hyperparameters: n_estimators, learning_rate, max_depth
  - Trains top 3 models for comparison
  - Feature importance visualization
- **Implementation:** `src/boosting.py`, `src/graph_boosting.py`

### 6. **Random Forest**
- **Purpose:** Ensemble method using bagging with decision trees
- **Features:**
  - Grid search with 10-fold cross-validation
  - Hyperparameters: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features
  - Feature importance visualization
- **Implementation:** `src/random_forest.py`, `src/graph_random_forest.py`

### 7. **Neural Networks (Multi-Layer Perceptron)**
- **Purpose:** Deep learning approach using MLP Regressor
- **Features:**
  - Grid search for architecture and hyperparameters
  - Hyperparameters: hidden_layer_sizes, learning_rate_init, alpha (L2 regularization)
  - Supports both count and log-transformed targets
- **Implementation:** `src/neural_net.py`, `src/graph_neural_net.py`

### 8. **Polynomial Regression**
- **Purpose:** Extends linear models with polynomial features
- **Features:**
  - Tests polynomial degrees 1-4
  - Cross-validated model selection
  - Comprehensive visualization of polynomial fits
- **Implementation:** `src/polytest.py` (by Sherry Dela Cruz)

---

## Methodology

### Data Preprocessing
- **Train/Test Split:** Single 80/20 split at the beginning to prevent data leakage
- **Feature Encoding:** 
  - Numerical features: StandardScaler
  - Categorical features: OneHotEncoder (drop='first')
- **Target Transformations:**
  - Original count data (for Poisson)
  - Log-transformed (log1p) for linear models
  - Binary classification (threshold: >4 injuries)

### Feature Selection
- **Backward Elimination:** 
  - Performed on logistic regression using Model 6 (kitchen sink)
  - Significance level: α = 0.1
  - Selected features applied to all model types
  - Prevents overfitting by removing non-significant predictors

### Model Evaluation
- **Cross-Validation:** 10-fold KFold cross-validation on training set
- **Hyperparameter Tuning:** GridSearchCV with 10-fold CV for:
  - Gradient Boosting
  - Random Forest
  - Neural Networks
  - Elastic Net
- **Final Evaluation:** Held-out test set (20%) for unbiased performance estimate
- **Metrics:**
  - **Poisson:** Deviance, Pseudo-R²
  - **Linear:** R², Adjusted R², MSE, RMSE, MAE
  - **Logistic:** ROC-AUC, Accuracy, Precision, Recall, F1-Score, Log Loss

### Model Comparison
- **7 Model Variants:** Baseline → Avg_Temp → Game Intensity → Rest/Travel → Weather → Kitchen Sink → Backward Selected
- **Best Model Selection:** Based on 10-fold CV performance
  - Poisson: Lowest Poisson Deviance
  - Linear: Lowest MSE
  - Logistic: Highest ROC-AUC

---

## Results Summary

### Overall Findings

Despite employing comprehensive modeling approaches with:
- ✅ Extensive hyperparameter tuning via grid search
- ✅ 10-fold cross-validation for robust evaluation
- ✅ Backward feature selection to reduce overfitting
- ✅ Multiple model families (linear, tree-based, neural networks)
- ✅ Proper train/test splitting to prevent data leakage

**The dataset proved too noisy to achieve strong predictive capabilities for the number of injuries.**

### Key Observations

1. **Low Predictive Power:** 
   - Best models achieved R²/Pseudo-R² values typically below 20-30%
   - This indicates that most variance in injury counts is unexplained by available predictors

2. **High Stochasticity:**
   - Injury occurrence appears highly random
   - Factors not captured in the dataset (player-specific, play-by-play dynamics, etc.) likely dominate

3. **Model Performance:**
   - **Best Linear Model:** Log-transformed OLS with backward-selected features
   - **Best Poisson Model:** Kitchen sink or backward-selected features
   - **Best Logistic Model:** Various models with ROC-AUC ~0.55-0.60 (slightly better than random)

4. **Feature Insights:**
   - Weather variables (temperature, humidity, wind) showed limited predictive power
   - Game intensity metrics (plays, yards, turnovers) had weak associations
   - Rest days and travel distance showed minimal impact

### Interpretation

The inability to achieve strong predictions suggests that:
- Injury occurrence in NFL games is fundamentally stochastic
- Important predictive factors may be missing from the dataset (e.g., player load, biomechanical factors, contact intensity)
- The signal-to-noise ratio is too low for reliable prediction with current data

This finding is valuable in itself, as it indicates that injury prevention strategies should focus on factors beyond game-level statistics.

---

## Usage

### Running the Main Analysis

```bash
cd src
python main.py
```

This will:
1. Load and clean the data
2. Perform backward feature selection
3. Train all 7 model variants for Poisson, Linear, and Logistic regression
4. Select best models based on CV performance
5. Generate comprehensive visualizations and reports

### Running Individual Model Scripts

```bash
# Elastic Net
python elastic_net.py

# Gradient Boosting
python boosting.py

# Random Forest
python random_forest.py

# Neural Networks
python neural_net.py

# Polynomial Regression
python polytest.py

# Exploratory Data Analysis
python exploratory.py
```

### Output

All results are saved to the `plots/` directory:
- **Best models:** `plots/best/`
- **All model variants:** `plots/non_best/`
- **Model-specific results:** `plots/{model_type}/`
- **Reports:** Text files with detailed metrics
- **Visualizations:** PNG files with graphs and charts

---

## Dependencies

```python
numpy
pandas
scikit-learn
matplotlib
seaborn
```

---

## Key Features

- ✅ **No Data Leakage:** Single train/test split, proper cross-validation
- ✅ **Robust Evaluation:** 10-fold CV with grid search for hyperparameter tuning
- ✅ **Comprehensive Models:** 8 different modeling approaches
- ✅ **Feature Selection:** Backward elimination for optimal feature sets
- ✅ **Extensive Visualization:** Metrics, scatter plots, residuals, feature importance
- ✅ **Professional Reporting:** Detailed text reports with all metrics

---

## License

This project is for academic/research purposes. Data usage should comply with NFL data policies.

---

## Contact

For questions or collaboration opportunities, please contact:
- **Ramon McDargh-Mitchell:** cltuchdev.apps@gmail.com
- **Portfolio:** https://clutchstudio.dev

---

*Last Updated: December 2025*

