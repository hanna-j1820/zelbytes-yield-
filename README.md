# ZelBytes Yield Forecasting

## Environment Setup

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install pandas numpy matplotlib scikit-learn jupyter streamlit joblib
```

### Run Smoke Test

```bash
python src/smoke_test.py
```
## Train/Test Split Summary

- Train Period: 2025-01-01 to 2025-08-28
- Test Period: 2025-08-29 to 2025-10-27

- Train Rows: 240
- Test Rows: 60

### Features Used

- temperature
- humidity
- co2
- temp_humidity_interaction

### Scaling

MinMaxScaler was fitted on the training data only and then applied to the test data to prevent data leakage.

Scaler saved to:

models/scaler.joblib