import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# Create Output Folders
# ==========================================

Path("reports/figures").mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================
# Load Data
# ==========================================

X_train = pd.read_parquet(
    "data/processed/X_train.parquet"
)

X_test = pd.read_parquet(
    "data/processed/X_test.parquet"
)

y_train = pd.read_csv(
    "data/processed/y_train.csv"
).values.ravel()

y_test = pd.read_csv(
    "data/processed/y_test.csv"
).values.ravel()

# ==========================================
# Linear Regression Baseline
# ==========================================

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

linear_pred = linear_model.predict(
    X_test
)

linear_mae = mean_absolute_error(
    y_test,
    linear_pred
)

linear_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        linear_pred
    )
)

linear_r2 = r2_score(
    y_test,
    linear_pred
)

# ==========================================
# Random Forest Model
# ==========================================

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(
    X_train,
    y_train
)

# ==========================================
# Predictions
# ==========================================

train_pred = rf_model.predict(
    X_train
)

test_pred = rf_model.predict(
    X_test
)

# ==========================================
# Metrics
# ==========================================

train_mae = mean_absolute_error(
    y_train,
    train_pred
)

train_rmse = np.sqrt(
    mean_squared_error(
        y_train,
        train_pred
    )
)

train_r2 = r2_score(
    y_train,
    train_pred
)

test_mae = mean_absolute_error(
    y_test,
    test_pred
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_pred
    )
)

test_r2 = r2_score(
    y_test,
    test_pred
)

# ==========================================
# Time Series Cross Validation
# ==========================================

tscv = TimeSeriesSplit(
    n_splits=5
)

cv_scores = cross_val_score(
    rf_model,
    X_train,
    y_train,
    cv=tscv,
    scoring="r2"
)

# ==========================================
# Feature Importances
# ==========================================

feature_df = pd.DataFrame({
    "feature": X_train.columns,
    "importance": rf_model.feature_importances_
})

feature_df = feature_df.sort_values(
    by="importance",
    ascending=False
)

print("\nFeature Importances:")
print(feature_df)

# ==========================================
# Save Feature Importance Chart
# ==========================================

plt.figure(figsize=(8, 5))

plt.bar(
    feature_df["feature"],
    feature_df["importance"]
)

plt.title(
    "Random Forest Feature Importance"
)

plt.xlabel("Features")
plt.ylabel("Importance")

plt.xticks(rotation=20)

plt.tight_layout()

plt.savefig(
    "reports/figures/rf_feature_importance.png"
)

plt.close()

# ==========================================
# Comparison Table
# ==========================================

comparison_df = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Random Forest"
    ],
    "MAE": [
        linear_mae,
        test_mae
    ],
    "RMSE": [
        linear_rmse,
        test_rmse
    ],
    "R2": [
        linear_r2,
        test_r2
    ]
})

print("\nModel Comparison:")
print(comparison_df)

comparison_df.to_csv(
    "reports/model_comparison.csv",
    index=False
)

# ==========================================
# Save Metrics
# ==========================================

metrics = {
    "train_mae": float(train_mae),
    "train_rmse": float(train_rmse),
    "train_r2": float(train_r2),
    "test_mae": float(test_mae),
    "test_rmse": float(test_rmse),
    "test_r2": float(test_r2),
    "cv_scores": cv_scores.tolist(),
    "cv_mean_r2": float(cv_scores.mean())
}

with open(
    "reports/random_forest_metrics.json",
    "w"
) as f:
    json.dump(
        metrics,
        f,
        indent=4
    )

# ==========================================
# Print Summary
# ==========================================

print("\n===== RANDOM FOREST RESULTS =====")

print(f"\nTrain MAE: {train_mae:.4f}")
print(f"Train RMSE: {train_rmse:.4f}")
print(f"Train R2: {train_r2:.4f}")

print(f"\nTest MAE: {test_mae:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")
print(f"Test R2: {test_r2:.4f}")

print("\nTimeSeriesSplit CV Scores:")
print(cv_scores)

print(
    f"\nAverage CV R2: "
    f"{cv_scores.mean():.4f}"
)

print(
    "\nFeature importance chart saved to:"
)
print(
    "reports/figures/rf_feature_importance.png"
)

print(
    "\nMetrics saved to:"
)
print(
    "reports/random_forest_metrics.json"
)

print(
    "\nComparison table saved to:"
)
print(
    "reports/model_comparison.csv"
)