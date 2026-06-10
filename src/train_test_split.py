import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib
from pathlib import Path

# ==========================================
# Create Models Folder
# ==========================================

Path("models").mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================
# Load and Sort Data
# ==========================================

df = pd.read_parquet(
    "data/processed/02_cleaned.parquet"
).sort_values("date")

# ==========================================
# Feature Engineering
# ==========================================

df["temp_humidity_interaction"] = (
    df["temperature"] * df["humidity"]
)

feature_cols = [
    "temperature",
    "humidity",
    "co2",
    "temp_humidity_interaction"
]

# ==========================================
# Chronological 80/20 Split
# ==========================================

split_idx = int(len(df) * 0.8)

train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

# ==========================================
# Scaling
# ==========================================

scaler = MinMaxScaler()

X_train = scaler.fit_transform(
    train[feature_cols]
)

X_test = scaler.transform(
    test[feature_cols]
)

# ==========================================
# Target
# ==========================================

y_train = train["yield"].values
y_test = test["yield"].values

# ==========================================
# Save Scaler
# ==========================================

joblib.dump(
    scaler,
    "data/processed/train_scaler.pkl"
)

# Required by Task 4
joblib.dump(
    scaler,
    "models/scaler.joblib"
)

# ==========================================
# Save Train/Test Files
# ==========================================

pd.DataFrame(
    X_train,
    columns=feature_cols
).to_parquet(
    "data/processed/X_train.parquet",
    index=False
)

pd.DataFrame(
    X_test,
    columns=feature_cols
).to_parquet(
    "data/processed/X_test.parquet",
    index=False
)

pd.Series(y_train).to_csv(
    "data/processed/y_train.csv",
    index=False
)

pd.Series(y_test).to_csv(
    "data/processed/y_test.csv",
    index=False
)

# ==========================================
# Verification
# ==========================================

print(
    f"\nTrain: {train['date'].min()} → {train['date'].max()}"
)

print(
    f"Test: {test['date'].min()} → {test['date'].max()}"
)

print(f"\nTrain rows: {len(train)}")
print(f"Test rows: {len(test)}")

print("\nScaler saved to:")
print(" - data/processed/train_scaler.pkl")
print(" - models/scaler.joblib")

print("\nFiles Saved Successfully")