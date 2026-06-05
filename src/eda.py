import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Load cleaned data
df = pd.read_parquet("data/processed/02_cleaned.parquet")

# ----------------------
# Summary Statistics
# ----------------------
summary = df.describe()

# Save summary statistics
with open("reports/summary_stats.txt", "w") as f:
    f.write(str(summary))

# ----------------------
# Correlation Heatmap
# ----------------------
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("reports/figures/correlation_heatmap.png")
plt.close()

# ----------------------
# Humidity vs Yield
# ----------------------
plt.figure(figsize=(6,4))
plt.scatter(df["humidity"], df["yield"])
plt.xlabel("Humidity (%)")
plt.ylabel("Yield (kg)")
plt.title("Humidity vs Yield")
plt.tight_layout()
plt.savefig("reports/figures/humidity_vs_yield.png")
plt.close()

# ----------------------
# CO2 vs Yield
# ----------------------
plt.figure(figsize=(6,4))
plt.scatter(df["co2"], df["yield"])
plt.xlabel("CO2 (ppm)")
plt.ylabel("Yield (kg)")
plt.title("CO2 vs Yield")
plt.tight_layout()
plt.savefig("reports/figures/co2_vs_yield.png")
plt.close()

print("EDA complete.")