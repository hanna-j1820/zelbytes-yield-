import pandas as pd

df = pd.read_parquet("data/processed/02_cleaned.parquet")

print(df.head(50))

df.head(50).to_csv("sample_cleaned_data.csv", index=False)