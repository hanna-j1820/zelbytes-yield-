import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ==========================================
# Create Output Folders
# ==========================================

Path("models").mkdir(
    parents=True,
    exist_ok=True
)

Path("reports").mkdir(
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
# Helper Function
# ==========================================

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    return mae, rmse, r2

# ==========================================
# Linear Regression Baseline
# ==========================================

linear_model = LinearRegression()

linear_model.fit(
    X_train,
    y_train
)

linear_mae, linear_rmse, linear_r2 = evaluate_model(
    linear_model,
    X_test,
    y_test
)

# ==========================================
# Default Random Forest
# ==========================================

rf_default = RandomForestRegressor(
    random_state=42
)

rf_default.fit(
    X_train,
    y_train
)

rf_mae, rf_rmse, rf_r2 = evaluate_model(
    rf_default,
    X_test,
    y_test
)

# ==========================================
# GridSearchCV Tuning
# ==========================================

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 10, None],
    "min_samples_leaf": [1, 2, 4]
}

tscv = TimeSeriesSplit(
    n_splits=5
)

grid_search = GridSearchCV(
    estimator=RandomForestRegressor(
        random_state=42
    ),
    param_grid=param_grid,
    cv=tscv,
    scoring="r2",
    n_jobs=-1
)

grid_search.fit(
    X_train,
    y_train
)

best_rf = grid_search.best_estimator_

# ==========================================
# Evaluate Tuned RF
# ==========================================

tuned_mae, tuned_rmse, tuned_r2 = evaluate_model(
    best_rf,
    X_test,
    y_test
)

# ==========================================
# Comparison Table
# ==========================================

comparison_df = pd.DataFrame({
    "Model": [
        "Linear Regression",
        "Default Random Forest",
        "Tuned Random Forest"
    ],
    "MAE": [
        linear_mae,
        rf_mae,
        tuned_mae
    ],
    "RMSE": [
        linear_rmse,
        rf_rmse,
        tuned_rmse
    ],
    "R2": [
        linear_r2,
        rf_r2,
        tuned_r2
    ]
})

print("\n===== MODEL COMPARISON =====")
print(comparison_df)

# ==========================================
# Save Comparison Table
# ==========================================

comparison_df.to_csv(
    "reports/model_comparison_tuned.csv",
    index=False
)

# ==========================================
# Save Champion Model
# ==========================================

joblib.dump(
    best_rf,
    "models/champion.joblib"
)

# ==========================================
# Save Markdown Report
# ==========================================

with open(
    "reports/champion_model.md",
    "w"
) as f:

    f.write("# Champion Model Selection\n\n")

    f.write(
        f"Best Parameters:\n\n"
        f"{grid_search.best_params_}\n\n"
    )

    f.write(
        "The tuned Random Forest model "
        "was selected as the champion model "
        "based on GridSearchCV performance.\n\n"
    )

    f.write(
        comparison_df.to_markdown(
            index=False
        )
    )

# ==========================================
# Print Results
# ==========================================

print("\n===== BEST PARAMETERS =====")
print(
    grid_search.best_params_
)

print(
    "\nBest CV Score:",
    grid_search.best_score_
)

print(
    "\nChampion model saved to:"
)

print(
    "models/champion.joblib"
)

print(
    "\nComparison table saved to:"
)

print(
    "reports/model_comparison_tuned.csv"
)

print(
    "\nChampion report saved to:"
)

print(
    "reports/champion_model.md"
)