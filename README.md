# Sales Prediction ML Project

A machine learning project for predicting retail sales using multiple regression models. This project compares the performance of Linear Regression, Polynomial Regression, K-Nearest Neighbors, and XGBoost on a retail sales dataset.

## Project Overview

This project aims to build predictive models for sales forecasting using historical sales data. We test multiple algorithms and identify the best performing model through rigorous evaluation on train/validation/test sets.

**Best Model:** XGBoost with R² = 0.8863 on validation set

## Project Structure

```
ML-project/
├── README.md
├── notebooks/
│   ├── 01_EDA.ipynb                    # Exploratory Data Analysis
│   ├── 02_Data_Preparation.ipynb       # Data preprocessing and splitting
│   ├── 03_Model_Training.ipynb         # Model training and comparison
│   └── 04_Final_Evaluation.ipynb       # Final model evaluation on test set
├── dataset/
│   ├── sales.csv                       # Original sales data
│   ├── inference.csv                   # Data for final predictions
│   └── processed/
│       └── train_val_test_split.pkl    # Preprocessed train/val/test split
├── models/
│   ├── linear_regression_model.pkl
│   ├── polynomial_degree2_model.pkl
│   ├── polynomial_degree3_model.pkl
│   ├── knn_model.pkl
│   └── best_xgboost_model.pkl
└── results/
    └── inference_predictions.csv       # Final predictions on inference data
```

## Dataset

- **Source:** `dataset/sales.csv`
- **Samples:** 640,840 records
- **Features:** 9 (store_ID, day_of_week, nb_customers_on_day, open, promotion, state_holiday, school_holiday, etc.)
- **Target:** sales (continuous value)

### Data Split
- **Training:** 70% (372,411 samples)
- **Validation:** 15% (79,802 samples)
- **Test:** 15% (79,803 samples)

## Models Tested

| Model | Train MAE | Train R² | Val MAE | Val R² |
|-------|-----------|----------|---------|--------|
| Linear Regression | 1154.58 | 0.7298 | 1156.53 | 0.7291 |
| Polynomial (Degree 2) | 1105.43 | 0.7585 | 1104.11 | 0.7589 |
| Polynomial (Degree 3) | 1093.03 | 0.7660 | 1091.47 | 0.7664 |
| KNN (k=3) | 21.28 | 0.9992 | 893.13 | 0.8073 |
| **XGBoost (Best)** | **667.46** | **0.9069** | **728.64** | **0.8863** |

## Key Findings

1. **XGBoost performs best** - Achieved the highest validation R² score (0.8863) and lowest validation MAE (728.64)

2. **KNN shows overfitting** - Perfect training R² (0.9992) but significantly worse validation performance (0.8073)

3. **Polynomial models improve over linear** - Degree 3 polynomial performs better than degree 2, suggesting non-linear relationships

4. **Hyperparameter tuning matters** - XGBoost's random search found optimal parameters:
   - n_estimators: 300
   - max_depth: 10
   - learning_rate: 0.1
   - subsample: 0.8
   - colsample_bytree: 1.0
   - min_child_weight: 5

## Running the Project

### Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn
```

### Workflow

1. **Exploratory Data Analysis**
   ```bash
   jupyter notebook notebooks/01_EDA.ipynb
   ```
   - Understand data distributions
   - Analyze feature correlations
   - Identify patterns and anomalies

2. **Data Preparation**
   ```bash
   jupyter notebook notebooks/02_Data_Preparation.ipynb
   ```
   - Clean and preprocess data
   - Create train/validation/test split
   - Save processed data

3. **Model Training & Comparison**
   ```bash
   jupyter notebook notebooks/03_Model_Training.ipynb
   ```
   - Train all 5 models
   - Evaluate on validation set
   - Compare performance metrics
   - Save trained models

4. **Final Evaluation**
   ```bash
   jupyter notebook notebooks/04_Final_Evaluation.ipynb
   ```
   - Evaluate best model on test set
   - Generate final predictions
   - Generate insights and conclusions

## Model Performance on Test Set

- **Best Model:** XGBoost
- **Test R² Score:** ~0.88
- **Test MAE:** ~730 (in sales units)

## Making Predictions

To generate predictions on new data:

```python
import pickle
import pandas as pd

# Load the best model
with open('models/best_xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load inference data
X_inference = pd.read_csv('dataset/inference.csv')

# Generate predictions
predictions = model.predict(X_inference)
```

## Key Metrics

- **MAE (Mean Absolute Error):** Average prediction error in sales units
- **R² Score:** Proportion of variance explained (1.0 = perfect, 0.0 = worst)

## Next Steps for Improvement

1. Feature engineering - Create additional features from timestamps
2. Ensemble methods - Combine multiple models
3. Feature selection - Identify most important features
4. Cross-validation - Use k-fold for more robust evaluation
5. Hyperparameter optimization - Expand search space for better tuning

## Authors

Machine Learning Group 3

## License

Educational project - Ironhack AI Class
