\# Imputation and Removal Rationale



\## Temperature

Missing values were replaced using the median because temperature sensors may occasionally fail or produce missing readings. The median is robust to extreme values.



\## Humidity

Missing values were replaced using the median to maintain environmental consistency within the polyhouse.



\## CO2

Missing values were replaced using the median because sensor interruptions can cause temporary missing values. Median reduces the influence of abnormal spikes.



\## Yield

Missing values were replaced using the median to avoid bias caused by unusually high or low harvest observations.



\## Duplicate Records

Duplicate rows were removed to prevent repeated sensor observations from affecting analysis and model training.

