\# Monitoring Plan



\## Objective



Monitor prediction activity of the mushroom yield forecasting application and identify conditions requiring model retraining.



\---



\## Prediction Log Sample



| Timestamp        | Temperature (°C) | Humidity (%) | CO₂ (ppm) | Predicted Yield (kg) |

| ---------------- | ---------------- | ------------ | --------- | -------------------- |

| 2026-06-18 10:00 | 24.5             | 85           | 650       | 14.7                 |

| 2026-06-18 10:05 | 25.0             | 87           | 680       | 15.1                 |

| 2026-06-18 10:10 | 23.8             | 84           | 640       | 14.2                 |



\---



\## Fields To Monitor



\* Timestamp

\* Temperature

\* Humidity

\* CO₂

\* Predicted Yield



\---



\## Model Artifact Handling



The deployed application uses:



\* models/champion.joblib

\* models/scaler.joblib



These artifacts are committed to the repository and loaded by the Streamlit application during startup.



\---



\## Retraining Triggers



The model should be retrained when:



1\. New sensor data exceeds 20% of the original training dataset.

2\. Environmental conditions move outside the training range.

3\. Prediction quality noticeably degrades.

4\. Seasonal environmental changes affect mushroom growth behavior.



\---



\## Monitoring Strategy



Prediction logs should be reviewed periodically to identify:



\* Data drift

\* Sensor anomalies

\* Unusual prediction distributions

\* Environmental changes not represented in training data



Retraining should be scheduled when significant drift is observed.



