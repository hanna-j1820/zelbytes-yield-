import pandas as pd

df = pd.read_csv("data/raw/polyhouse_sensor.csv")

print("Dataset loaded successfully")
print("Rows:", len(df))
print("Columns:", list(df.columns))
print(df.head())