import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(42)

n = 300

# Generate realistic polyhouse data
temperature = np.random.normal(22, 1.5, n)
humidity = np.random.normal(87, 3, n)
co2 = np.random.normal(900, 80, n)

# Yield influenced by all factors + random noise
yield_kg = (
    14
    + 0.15 * (temperature - 22)
    + 0.04 * (humidity - 87)
    - 0.001 * np.abs(co2 - 900)
    + np.random.normal(0, 0.5, n)
)

# Create dates
dates = pd.date_range(
    start="2025-01-01",
    periods=n,
    freq="D"
)

df = pd.DataFrame({
    "date": dates,
    "temperature": np.round(temperature, 2),
    "humidity": np.round(humidity, 2),
    "co2": np.round(co2, 0),
    "yield": np.round(yield_kg, 2)
})

Path("data/processed").mkdir(parents=True, exist_ok=True)

df.to_parquet(
    "data/processed/02_cleaned.parquet",
    index=False
)

print("Dataset created successfully.")
print(df.head())
print(f"\nRows: {len(df)}")