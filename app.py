import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =====================================
# Page Configuration
# =====================================

st.set_page_config(
    page_title="Mushroom Yield Forecast",
    page_icon="🍄",
    layout="wide"
)

# =====================================
# Load Model and Scaler
# =====================================

@st.cache_resource
def load_artifacts():
    model = joblib.load("models/champion.joblib")
    scaler = joblib.load("models/scaler.joblib")
    return model, scaler

model, scaler = load_artifacts()

# =====================================
# App Title
# =====================================

st.title("🍄 Mushroom Yield Forecast App")

st.write("""
Predict mushroom yield using environmental sensor values.

Inputs:
- Temperature (°C)
- Humidity (%)
- CO₂ (ppm)
""")

# =====================================
# Sidebar Inputs
# =====================================

st.sidebar.header("Sensor Inputs")

temperature = st.sidebar.slider(
    "Temperature (°C)",
    min_value=15.0,
    max_value=35.0,
    value=24.0,
    step=0.1
)

humidity = st.sidebar.slider(
    "Humidity (%)",
    min_value=50.0,
    max_value=100.0,
    value=85.0,
    step=0.1
)

co2 = st.sidebar.slider(
    "CO₂ (ppm)",
    min_value=300,
    max_value=2000,
    value=650,
    step=10
)

# =====================================
# Feature Engineering
# =====================================

temp_humidity_interaction = temperature * humidity

# =====================================
# Create Input DataFrame
# =====================================

input_df = pd.DataFrame({
    "temperature": [temperature],
    "humidity": [humidity],
    "co2": [co2],
    "temp_humidity_interaction": [temp_humidity_interaction]
})

# =====================================
# Scale Features
# =====================================

scaled_input = scaler.transform(input_df)

# =====================================
# Predict Yield
# =====================================

prediction = model.predict(scaled_input)[0]

# =====================================
# Display Prediction
# =====================================

st.subheader("Predicted Yield")

st.success(f"{prediction:.2f} kg")

# =====================================
# Display Current Inputs
# =====================================

st.subheader("Current Sensor Values")

st.dataframe(input_df)

# =====================================
# Sensitivity Analysis
# =====================================

st.subheader("Yield Sensitivity to Humidity")

humidity_values = np.linspace(50, 100, 50)

test_df = pd.DataFrame({
    "temperature": [temperature] * 50,
    "humidity": humidity_values,
    "co2": [co2] * 50
})

test_df["temp_humidity_interaction"] = (
    test_df["temperature"]
    * test_df["humidity"]
)

scaled_test = scaler.transform(test_df)

predictions = model.predict(scaled_test)

chart_df = pd.DataFrame({
    "Humidity (%)": humidity_values,
    "Predicted Yield (kg)": predictions
})

st.line_chart(
    chart_df.set_index("Humidity (%)")
)

# =====================================
# Footer
# =====================================

st.caption(
    "Mushroom Yield Forecasting using Temperature, Humidity and CO₂"
)