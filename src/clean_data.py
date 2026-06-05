import pandas as pd
from pathlib import Path

raw_file = Path("data/raw/polyhouse_sensor.csv")
output_file = Path("data/processed/02_cleaned.parquet")

df = pd.read_csv(raw_file)

null_before = df.isnull().sum()

for col in df.columns:
    if df[col].dtype in ["float64", "int64"]:
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna(df[col].mode()[0])

df = df.drop_duplicates()

null_after = df.isnull().sum()

df.to_parquet(output_file, index=False)

with open("cleaning_log.md", "w") as f:
    f.write("# Cleaning Log\n\n")
    f.write("## Null Counts Before Cleaning\n\n")
    f.write(null_before.to_string())
    f.write("\n\n## Null Counts After Cleaning\n\n")
    f.write(null_after.to_string())

print("Cleaning complete.")