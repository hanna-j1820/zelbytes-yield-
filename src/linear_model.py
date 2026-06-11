import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# Load Data
# =========================
X_train = pd.read_parquet("data/processed/X_train.parquet")
X_test = pd.read_parquet("data/processed/X_test.parquet")

y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()
y_test = pd.read_csv("data/processed/y_test.csv").values.ravel()

# =========================
# Train Model
# =========================
model = LinearRegression()
model.fit(X_train, y_train)

# =========================
# Predict
# =========================
y_pred = model.predict(X_test)

# =========================
# Metrics
# =========================
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)

# =========================
# Save Metrics
# =========================
metrics = {
    "MAE": float(mae),
    "RMSE": float(rmse),
    "R2": float(r2)
}

with open("reports/linear_metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

# =========================
# Coefficients
# =========================
coef_df = pd.DataFrame({
    "feature": X_train.columns,
    "coefficient": model.coef_
})

print("\nCoefficients:")
print(coef_df)

# =========================
# Residuals
# =========================
residuals = y_test - y_pred

# Residual vs Predicted
plt.scatter(y_pred, residuals)
plt.xlabel("Predicted")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted")
plt.show()

# Residual vs Humidity
plt.scatter(X_test["humidity"], residuals)
plt.xlabel("Humidity")
plt.ylabel("Residuals")
plt.title("Residuals vs Humidity")
plt.show()