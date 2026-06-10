# ==========================================
# Phase 2 - Task 1
# Feature Engineering & Scaling
# ==========================================

import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import joblib

# ==========================================
# File Paths
# ==========================================

input_file = Path("data/processed/02_cleaned.parquet")
output_file = Path("data/processed/features.parquet")
scaler_file = Path("data/processed/minmax_scaler.pkl")

# ==========================================
# Load Data
# ==========================================

df = pd.read_parquet(input_file)

print("\nDataset Loaded Successfully")

# ==========================================
# Verify Dataset
# ==========================================

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

# ==========================================
# Feature Engineering
# ==========================================

# Biological reasoning:
# Mushroom growth depends on both temperature
# and humidity together, not independently.
# This interaction feature captures that effect.

df["temp_humidity_interaction"] = (
    df["temperature"] * df["humidity"]
)

# ==========================================
# Define X (Features)
# ==========================================

X = df[
    [
        "temperature",
        "humidity",
        "co2",
        "temp_humidity_interaction"
    ]
]

# ==========================================
# Define y (Target)
# ==========================================

y = df["yield"]

print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print("yield")

# ==========================================
# Apply MinMax Scaling
# ==========================================

scaler = MinMaxScaler()

X_scaled = scaler.fit_transform(X)

# ==========================================
# Convert Back to DataFrame
# ==========================================

X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=X.columns
)

# ==========================================
# Check Scaling
# ==========================================

print("\nMinimum Values:")
print(X_scaled_df.min())

print("\nMaximum Values:")
print(X_scaled_df.max())

# ==========================================
# Add Target Back
# ==========================================

features_df = X_scaled_df.copy()

features_df["yield"] = y

# ==========================================
# Save Features Dataset
# ==========================================

features_df.to_parquet(
    output_file,
    index=False
)

# ==========================================
# Save Scaler
# ==========================================

joblib.dump(
    scaler,
    scaler_file
)

# ==========================================
# Smoke Test
# ==========================================

print("\nFeature Dataset Shape:")
print(features_df.shape)

print("\nPreview:")
print(features_df.head())

print("\nSaved:")
print(output_file)
print(scaler_file)

print("\nPhase 2 Task 1 Completed Successfully")