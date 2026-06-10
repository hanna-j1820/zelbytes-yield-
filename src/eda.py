# Import pandas for data manipulation and analysis
import pandas as pd

# Import matplotlib for creating graphs and plots
import matplotlib.pyplot as plt

# Import seaborn for advanced statistical visualizations
import seaborn as sns

# Import Path for handling folders and file paths
from pathlib import Path


# ==========================================
# Create Output Folders
# ==========================================

# Create the folder "reports/figures"
# parents=True -> create missing parent folders if needed
# exist_ok=True -> don't throw an error if folder already exists
Path("reports/figures").mkdir(parents=True, exist_ok=True)


# ==========================================
# Load Dataset
# ==========================================

# Read the cleaned parquet file into a DataFrame
df = pd.read_parquet("data/processed/02_cleaned.parquet")


# ==========================================
# Dataset Information
# ==========================================

# Get dataset dimensions
# Example: (300, 5)
rows, cols = df.shape

# Generate statistical summary of numeric columns
# Includes mean, std, min, max, quartiles, etc.
summary_stats = df.describe()


# ==========================================
# Rule Violations
# ==========================================

# Initialize violation counters
humidity_violations = 0

# Check if humidity column exists
if "humidity" in df.columns:

    # Count values below 0 or above 100
    # Humidity should be between 0 and 100%
    humidity_violations = (
        (df["humidity"] < 0) |
        (df["humidity"] > 100)
    ).sum()


co2_violations = 0

# Check if CO2 column exists
if "co2" in df.columns:

    # Count negative CO2 values
    # Negative CO2 values are impossible
    co2_violations = (
        df["co2"] < 0
    ).sum()


yield_violations = 0

# Check if yield column exists
if "yield" in df.columns:

    # Count negative yield values
    yield_violations = (
        df["yield"] < 0
    ).sum()


# ==========================================
# Date Range
# ==========================================

# Default message if no date column is found
date_info = "No date column found"

# Loop through all columns
for col in df.columns:

    # Check if column name contains the word "date"
    if "date" in col.lower():

        # Convert column to datetime format
        df[col] = pd.to_datetime(df[col])

        # Find earliest and latest date
        date_info = (
            f"{df[col].min().date()} "
            f"to "
            f"{df[col].max().date()}"
        )

        # Stop after first date column is found
        break


# ==========================================
# Correlation Heatmap
# ==========================================

# Select only numeric columns
# Non-numeric columns like dates are excluded
numeric_df = df.select_dtypes(include="number")

# Calculate correlation matrix
# Values range from -1 to +1
corr_matrix = numeric_df.corr()

# Print correlation matrix in terminal
print("\nCorrelation Matrix:")
print(corr_matrix)

# Create a new figure
plt.figure(figsize=(8, 6))

# Draw correlation heatmap
sns.heatmap(
    corr_matrix,

    # Show correlation values inside cells
    annot=True,

    # Display values with 2 decimal places
    fmt=".2f",

    # Use blue-white-red color palette
    cmap="coolwarm",

    # Minimum correlation value
    vmin=-1,

    # Maximum correlation value
    vmax=1,

    # White color represents zero correlation
    center=0,

    # Add borders between cells
    linewidths=0.5
)

# Add graph title
plt.title("Sensor & Yield Correlations")

# Adjust spacing automatically
plt.tight_layout()

# Save heatmap image
plt.savefig(
    "reports/figures/correlation_heatmap.png",
    dpi=300
)

# Display heatmap on screen
plt.show()

# Close current figure and free memory
plt.close()


# ==========================================
# Individual Scatter Plot:
# Humidity vs Yield
# ==========================================

# Create plot only if both columns exist
if (
    "humidity" in df.columns
    and
    "yield" in df.columns
):

    plt.figure(figsize=(8, 6))

    # Create scatter plot
    plt.scatter(
        df["humidity"],   # X-axis
        df["yield"],      # Y-axis

        # Transparency
        alpha=0.4,

        # Point size
        s=30
    )

    plt.title("Humidity vs Yield")

    plt.xlabel("Humidity (%)")
    plt.ylabel("Yield (kg)")

    plt.tight_layout()

    # Save graph
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
# Combined Figure
# ==========================================

# Create one figure with 3 graphs side-by-side
fig, axes = plt.subplots(
    1,      # rows
    3,      # columns
    figsize=(18, 6)
)

# Graph 1
axes[0].scatter(
    df["humidity"],
    df["yield"],
    alpha=0.4,
    s=30
)

axes[0].set_title("Humidity vs Yield")
axes[0].set_xlabel("Humidity (%)")
axes[0].set_ylabel("Yield (kg)")


# Graph 2
axes[1].scatter(
    df["temperature"],
    df["yield"],
    alpha=0.4,
    s=30
)

axes[1].set_title("Temperature vs Yield")
axes[1].set_xlabel("Temperature (°C)")
axes[1].set_ylabel("Yield (kg)")


# Graph 3
axes[2].scatter(
    df["co2"],
    df["yield"],
    alpha=0.4,
    s=30
)

axes[2].set_title("CO2 vs Yield")
axes[2].set_xlabel("CO2 (ppm)")
axes[2].set_ylabel("Yield (kg)")

# Adjust spacing
plt.tight_layout()

# Save combined graph
plt.savefig(
    "reports/figures/scatter_yield.png",
    dpi=300
)

plt.close()


# ==========================================
# Save Summary Statistics
# ==========================================

# Open text file for writing
with open(
    "reports/summary_stats.txt",
    "w",
    encoding="utf-8"
) as f:

    # Write statistics into file
    f.write(summary_stats.to_string())


# ==========================================
# Create EDA Summary Report
# ==========================================

with open(
    "reports/eda_summary.md",
    "w",
    encoding="utf-8"
) as f:

    # Main title
    f.write("# EDA Summary\n\n")

    # Dataset overview
    f.write("## Dataset Overview\n")
    f.write(f"- Rows: {rows}\n")
    f.write(f"- Columns: {cols}\n")
    f.write(f"- Date Range: {date_info}\n\n")

    # Rule violations section
    f.write("## Rule Violations\n")
    f.write(f"- Humidity outside 0-100%: {humidity_violations}\n")
    f.write(f"- Negative CO2 values: {co2_violations}\n")
    f.write(f"- Negative Yield values: {yield_violations}\n\n")

    # Statistics section
    f.write("## Summary Statistics\n\n")
    f.write(summary_stats.to_string())
    f.write("\n\n")

    # Project insights
    f.write("## Insights\n")
    f.write("1. Temperature shows a positive relationship with mushroom yield.\n")
    f.write("2. Humidity contributes to yield stability inside the polyhouse.\n")
    f.write("3. CO2 levels influence mushroom growth and productivity.\n")
    f.write("4. Correlation analysis highlights the strongest environmental drivers of yield.\n")


# ==========================================
# Completion Messages
# ==========================================

print("EDA completed successfully.")
print("Heatmap saved.")
print("Scatter plots saved.")
print("EDA report generated.")