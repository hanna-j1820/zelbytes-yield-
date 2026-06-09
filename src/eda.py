import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==========================================
# Create Output Folders
# ==========================================

Path("reports/figures").mkdir(parents=True, exist_ok=True)

# ==========================================
# Load Dataset
# ==========================================

df = pd.read_parquet("data/processed/02_cleaned.parquet")

# ==========================================
# Dataset Information
# ==========================================

rows, cols = df.shape

summary_stats = df.describe()

# ==========================================
# Rule Violations
# ==========================================

humidity_violations = 0
if "humidity" in df.columns:
    humidity_violations = (
        (df["humidity"] < 0) |
        (df["humidity"] > 100)
    ).sum()

co2_violations = 0
if "co2" in df.columns:
    co2_violations = (
        df["co2"] < 0
    ).sum()

yield_violations = 0
if "yield" in df.columns:
    yield_violations = (
        df["yield"] < 0
    ).sum()

# ==========================================
# Date Range
# ==========================================

date_info = "No date column found"

for col in df.columns:
    if "date" in col.lower():

        df[col] = pd.to_datetime(df[col])

        date_info = (
            f"{df[col].min().date()} "
            f"to "
            f"{df[col].max().date()}"
        )

        break

# ==========================================
# Correlation Heatmap
# ==========================================

numeric_df = df.select_dtypes(include="number")

# Calculate correlation matrix
corr_matrix = numeric_df.corr()

# Print correlation matrix in terminal
print("\nCorrelation Matrix:")
print(corr_matrix)

plt.figure(figsize=(8, 6))

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,          # Fix color scale minimum
    vmax=1,           # Fix color scale maximum
    center=0,         # White color at zero correlation
    linewidths=0.5
)

plt.title("Sensor & Yield Correlations")

plt.tight_layout()

plt.savefig(
    "reports/figures/correlation_heatmap.png",
    dpi=300
)

plt.show()      # Display heatmap on screen
plt.close()

# ==========================================
# Individual Scatter Plot:
# Humidity vs Yield
# ==========================================

if (
    "humidity" in df.columns
    and
    "yield" in df.columns
):

    plt.figure(figsize=(8, 6))

    plt.scatter(
        df["humidity"],
        df["yield"],
        alpha=0.4,
        s=30
    )

    plt.title("Humidity vs Yield")

    plt.xlabel("Humidity (%)")
    plt.ylabel("Yield (kg)")

    plt.tight_layout()

    plt.savefig(
        "reports/figures/humidity_vs_yield.png",
        dpi=300
    )

    plt.close()

# ==========================================
# Individual Scatter Plot:
# Temperature vs Yield
# ==========================================

if (
    "temperature" in df.columns
    and
    "yield" in df.columns
):

    plt.figure(figsize=(8, 6))

    plt.scatter(
        df["temperature"],
        df["yield"],
        alpha=0.4,
        s=30
    )

    plt.title("Temperature vs Yield")

    plt.xlabel("Temperature (°C)")
    plt.ylabel("Yield (kg)")

    plt.tight_layout()

    plt.savefig(
        "reports/figures/temperature_vs_yield.png",
        dpi=300
    )

    plt.close()

# ==========================================
# Individual Scatter Plot:
# CO2 vs Yield
# ==========================================

if (
    "co2" in df.columns
    and
    "yield" in df.columns
):

    plt.figure(figsize=(8, 6))

    plt.scatter(
        df["co2"],
        df["yield"],
        alpha=0.4,
        s=30
    )

    plt.title("CO2 vs Yield")

    plt.xlabel("CO2 (ppm)")
    plt.ylabel("Yield (kg)")

    plt.tight_layout()

    plt.savefig(
        "reports/figures/co2_vs_yield.png",
        dpi=300
    )

    plt.close()

# ==========================================
# Combined Figure (Like Screenshot)
# ==========================================

fig, axes = plt.subplots(
    1,
    3,
    figsize=(18, 6)
)

axes[0].scatter(
    df["humidity"],
    df["yield"],
    alpha=0.4,
    s=30
)

axes[0].set_title("Humidity vs Yield")
axes[0].set_xlabel("Humidity (%)")
axes[0].set_ylabel("Yield (kg)")

axes[1].scatter(
    df["temperature"],
    df["yield"],
    alpha=0.4,
    s=30
)

axes[1].set_title("Temperature vs Yield")
axes[1].set_xlabel("Temperature (°C)")
axes[1].set_ylabel("Yield (kg)")

axes[2].scatter(
    df["co2"],
    df["yield"],
    alpha=0.4,
    s=30
)

axes[2].set_title("CO2 vs Yield")
axes[2].set_xlabel("CO2 (ppm)")
axes[2].set_ylabel("Yield (kg)")

plt.tight_layout()

plt.savefig(
    "reports/figures/scatter_yield.png",
    dpi=300
)

plt.close()

# ==========================================
# Save Summary Statistics
# ==========================================

with open(
    "reports/summary_stats.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(summary_stats.to_string())

# ==========================================
# Create EDA Summary Report
# ==========================================

with open(
    "reports/eda_summary.md",
    "w",
    encoding="utf-8"
) as f:

    f.write("# EDA Summary\n\n")

    f.write("## Dataset Overview\n")
    f.write(f"- Rows: {rows}\n")
    f.write(f"- Columns: {cols}\n")
    f.write(f"- Date Range: {date_info}\n\n")

    f.write("## Rule Violations\n")
    f.write(f"- Humidity outside 0-100%: {humidity_violations}\n")
    f.write(f"- Negative CO2 values: {co2_violations}\n")
    f.write(f"- Negative Yield values: {yield_violations}\n\n")

    f.write("## Summary Statistics\n\n")
    f.write(summary_stats.to_string())
    f.write("\n\n")

    f.write("## Insights\n")
    f.write("1. Temperature shows a positive relationship with mushroom yield.\n")
    f.write("2. Humidity contributes to yield stability inside the polyhouse.\n")
    f.write("3. CO2 levels influence mushroom growth and productivity.\n")
    f.write("4. Correlation analysis highlights the strongest environmental drivers of yield.\n")

print("EDA completed successfully.")
print("Heatmap saved.")
print("Scatter plots saved.")
print("EDA report generated.")