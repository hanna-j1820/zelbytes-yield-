\# Exploratory Data Analysis (EDA) Summary



\## Data Quality Report



\### Dataset Overview



\* Number of Rows: 60

\* Number of Columns: 4

\* Variables:



&#x20; \* Temperature

&#x20; \* Humidity

&#x20; \* CO2

&#x20; \* Yield



\### Date Range



No timestamp column was available in the dataset, therefore a date range could not be determined.



\### Summary Statistics



Summary statistics were generated using pandas `describe()` and include:



\* Mean

\* Standard Deviation

\* Minimum

\* Maximum

\* Quartiles (25%, 50%, 75%)



\### Rule Violations



\* Missing values were detected in Humidity, CO2, and Yield columns.

\* Missing values were imputed using median values.

\* Duplicate records were removed.

\* No missing values remain after cleaning.



\---



\## Visual Insights



\### Correlation Heatmap



The correlation heatmap highlights relationships between environmental variables and yield.



\### Humidity vs Yield



Yield generally increases as humidity increases. Maintaining stable humidity levels may improve mushroom growth and production.



\### CO2 vs Yield



Higher CO2 concentrations appear to be associated with higher mushroom yields, suggesting that CO2 plays an important role in crop development.



\---



\## Conclusions



1\. Humidity shows a positive relationship with yield.

2\. CO2 concentration also demonstrates a positive association with yield.

3\. Environmental conditions significantly influence mushroom production.

4\. Humidity and CO2 should be considered important features for future yield forecasting models.



