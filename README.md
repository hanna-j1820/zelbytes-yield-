# ZelBytes Yield Forecasting

## Environment Setup

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install pandas numpy matplotlib scikit-learn jupyter streamlit joblib
```

### Run Smoke Test

```bash
python src/smoke_test.py
```
## Train/Test Split Summary

- Train Period: 2025-01-01 to 2025-08-28
- Test Period: 2025-08-29 to 2025-10-27

- Train Rows: 240
- Test Rows: 60

### Features Used

- temperature
- humidity
- co2
- temp_humidity_interaction

### Scaling

MinMaxScaler was fitted on the training data only and then applied to the test data to prevent data leakage.

Scaler saved to:

models/scaler.joblib

# Task 6: Random Forest & Time-Series Cross Validation

## Objective
Train a Random Forest Regressor and compare its performance with the Linear Regression baseline using TimeSeriesSplit cross-validation.

## Methodology
The processed training and testing datasets were loaded and used to train a Random Forest Regressor. Model performance was evaluated using MAE, RMSE, and R² metrics on both training and testing datasets. TimeSeriesSplit cross-validation was applied to evaluate model stability across different time periods and prevent data leakage. Feature importance values were extracted from the trained model and visualized using a feature importance chart. The Random Forest results were then compared against the Linear Regression baseline.

## Random Forest Performance

### Train Metrics

| Metric | Value |
|----------|----------|
| MAE | 0.1578 |
| RMSE | 0.2026 |
| R² | 0.8667 |

### Test Metrics

| Metric | Value |
|----------|----------|
| MAE | 0.4950 |
| RMSE | 0.6216 |
| R² | 0.0327 |

## TimeSeriesSplit Cross Validation

CV Scores:

- -0.0694
- -0.5446
- 0.2139
- -0.0041
- -0.1428

Average CV R² Score:

- -0.1094

## Feature Importance Ranking

1. temp_humidity_interaction
2. temperature
3. co2
4. humidity

## Files Generated

- src/random_forest.py
- reports/random_forest_metrics.json
- reports/model_comparison.csv
- reports/figures/rf_feature_importance.png

## Conclusion

The Random Forest model was successfully trained and evaluated. The model achieved strong training performance but lower test and cross-validation scores, indicating possible overfitting. Feature importance analysis showed that the interaction between temperature and humidity was the most influential factor in yield prediction.

---

# Task 7: GridSearchCV & Champion Model Selection

## Objective

Tune Random Forest hyperparameters using GridSearchCV and select the best-performing model as the champion model.

## Methodology

The training dataset was used to tune a Random Forest Regressor using GridSearchCV with TimeSeriesSplit cross-validation. A hyperparameter grid consisting of n_estimators, max_depth, and min_samples_leaf was evaluated. The tuned Random Forest model was compared against the Linear Regression baseline and the default Random Forest model using MAE, RMSE, and R² metrics. The best-performing model was selected as the champion model and saved for future yield prediction.

## Best Hyperparameters

| Parameter | Value |
|------------|---------|
| max_depth | 3 |
| min_samples_leaf | 4 |
| n_estimators | 100 |

## Best Cross Validation Score

| Metric | Value |
|----------|----------|
| Best CV R² Score | 0.0574 |

## Model Comparison

| Model | MAE | RMSE | R² |
|---------|---------|---------|---------|
| Linear Regression | 0.4721 | 0.5805 | 0.1566 |
| Default Random Forest | 0.4950 | 0.6216 | 0.0328 |
| Tuned Random Forest | 0.4657 | 0.5793 | 0.1601 |

## Champion Model Selection

The Tuned Random Forest model was selected as the champion model because it achieved the lowest MAE, lowest RMSE, and highest R² score among all evaluated models. Hyperparameter tuning improved model performance and generalization compared to the default Random Forest model.

## Files Generated

- src/model_tuning.py
- models/champion.joblib
- reports/model_comparison_tuned.csv
- reports/champion_model.md

## Conclusion

GridSearchCV successfully identified an improved Random Forest configuration. The tuned Random Forest slightly outperformed both the Linear Regression baseline and the default Random Forest model, making it the final champion model for yield prediction.