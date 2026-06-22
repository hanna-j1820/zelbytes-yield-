\# Mushroom Yield Forecasting System



\## Final Technical Report



\---



\# 1. Problem Statement



Mushroom cultivation requires maintaining suitable environmental conditions inside growing facilities. Small variations in temperature, humidity, and CO₂ concentration can significantly affect mushroom yield.



The objective of this project was to develop a machine learning system capable of predicting mushroom yield using environmental sensor measurements.



The final system allows users to enter sensor values through a Streamlit web application and receive a yield prediction in kilograms.



\---



\# 2. Dataset Description



The project used a polyhouse environmental dataset containing:



\* Temperature (°C)

\* Humidity (%)

\* CO₂ (ppm)

\* Yield (kg)



The dataset contained approximately 300 observations representing environmental conditions and corresponding mushroom yields.



\## Features



\### Input Features



\* temperature

\* humidity

\* co2



\### Target Variable



\* yield\_kg



\---



\# 3. Data Cleaning



The raw dataset was cleaned before modeling.



Cleaning operations included:



\* Missing value detection

\* Missing value imputation

\* Data type validation

\* Duplicate record checks

\* Dataset consistency verification



Cleaned data was saved into processed files for downstream analysis.



\## Benefits



\* Improved data quality

\* Reduced noise

\* Consistent feature formats

\* Better model reliability



\---



\# 4. Exploratory Data Analysis (EDA)



EDA was performed to understand relationships between environmental conditions and mushroom yield.



\## Statistical Summary



Descriptive statistics were generated using:



\* Mean

\* Median

\* Standard deviation

\* Minimum values

\* Maximum values



\## Visualizations



Several plots were created:



\### Yield Distribution



Used to understand overall yield behavior.



\### Temperature vs Yield



Examined relationship between temperature and production.



\### Humidity vs Yield



Analyzed humidity influence on yield.



\### CO₂ vs Yield



Investigated impact of carbon dioxide concentration.



\### Correlation Heatmap



Measured linear relationships among variables.



\## Key Findings



\* Temperature showed moderate influence on yield.

\* Humidity contributed significantly to production.

\* CO₂ had measurable but weaker impact.

\* Environmental variables exhibited some correlation with yield.



\---



\# 5. Feature Engineering



Additional features were created to improve model performance.



\## Interaction Feature



A new feature was generated:



temp\_humidity\_interaction



Formula:



temperature × humidity



This feature captured combined environmental effects that individual variables could not represent independently.



\---



\# 6. Train-Test Split Strategy



A temporal train-test split was used.



\## Reason



Time-dependent datasets should preserve chronological order.



Random splitting may cause data leakage because future observations could appear in training data.



\## Split



Training Data:



\* First 80%



Testing Data:



\* Last 20%



This approach simulates real-world forecasting scenarios.



\---



\# 7. Feature Scaling



MinMaxScaler was applied.



\## Purpose



Scale features into a common range.



Benefits include:



\* Improved numerical stability

\* Better model consistency

\* Reduced feature magnitude differences



Scaler saved as:



models/scaler.joblib



\---



\# 8. Baseline Model



A Linear Regression model was trained as the baseline.



Metrics:



\* MAE

\* RMSE

\* R²



The baseline established a performance benchmark for future models.



\---



\# 9. Random Forest Model



A Random Forest Regressor was trained to capture nonlinear relationships.



\## Evaluation Metrics



\### Training Performance



\* MAE: 0.1578

\* RMSE: 0.2026

\* R²: 0.8667



\### Testing Performance



\* MAE: 0.4950

\* RMSE: 0.6216

\* R²: 0.0327



The gap between training and testing performance suggested overfitting.



\---



\# 10. Time Series Cross Validation



TimeSeriesSplit cross validation was used.



\## Purpose



Prevent temporal leakage.



\## Cross Validation Scores



\* -0.0694

\* -0.5446

\* 0.2139

\* -0.0041

\* -0.1428



Average Score:



\* -0.1094



\---



\# 11. Hyperparameter Tuning



GridSearchCV was used to tune Random Forest parameters.



\## Best Parameters



\* n\_estimators = 100

\* max\_depth = 3

\* min\_samples\_leaf = 4



Best CV Score:



0.0574



\---



\# 12. Champion Model Selection



Three models were compared.



| Model               | MAE    | RMSE   | R²     |

| ------------------- | ------ | ------ | ------ |

| Linear Regression   | 0.4721 | 0.5805 | 0.1566 |

| Random Forest       | 0.4950 | 0.6216 | 0.0328 |

| Tuned Random Forest | 0.4657 | 0.5793 | 0.1601 |



The Tuned Random Forest achieved the best overall performance.



Selected model:



Champion Model



Saved as:



models/champion.joblib



\---



\# 13. Deployment



The final model was deployed using Streamlit Community Cloud.



\## Live Application



https://mushroom-yield-forecast.streamlit.app



Features:



\* Interactive sliders

\* Real-time predictions

\* Cached model loading

\* Sensitivity analysis chart



\---



\# 14. Monitoring Strategy



Monitoring focuses on:



\* Input tracking

\* Prediction tracking

\* Drift detection

\* Performance degradation



Logged fields:



\* Timestamp

\* Temperature

\* Humidity

\* CO₂

\* Predicted Yield



Retraining occurs when:



\* New data exceeds 20% of training size

\* Significant drift occurs

\* Prediction quality declines



\---



\# 15. Limitations



Current limitations include:



\* Small dataset size

\* Synthetic data characteristics

\* Limited environmental variables

\* No real-world production validation



\---



\# 16. Future Work



Future improvements may include:



\* Larger datasets

\* Real sensor integration

\* Additional environmental variables

\* Deep learning approaches

\* Automated retraining pipelines

\* Cloud-based monitoring dashboards



\---



\# 17. Conclusion



This project successfully developed an end-to-end mushroom yield forecasting system.



The workflow included:



\* Data cleaning

\* Exploratory analysis

\* Feature engineering

\* Model training

\* Hyperparameter tuning

\* Model deployment

\* Monitoring planning



The final system provides real-time yield prediction through a publicly accessible Streamlit application.



