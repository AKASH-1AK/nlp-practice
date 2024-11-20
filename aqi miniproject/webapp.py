import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# Load models and preprocessing objects
knn_regressor = joblib.load(r"D:\aqi miniproject\knn_regressor.pkl")
imputer_X = joblib.load(r"D:\aqi miniproject\imputer_X.pkl")

# Load the dataset for reference
df = pd.read_csv("D:/aqi miniproject/data.csv", encoding='latin1')  # Ensure this file contains 'date', 'location', 'so2', 'no2', 'rspm', 'spm'


# AQI Category Function
def categorize_aqi(aqi_value):
    if aqi_value <= 50:
        return 'Good'
    elif 51 <= aqi_value <= 100:
        return 'Moderate'
    elif 101 <= aqi_value <= 200:
        return 'Unhealthy for Sensitive Groups'
    elif 201 <= aqi_value <= 300:
        return 'Unhealthy'
    elif 301 <= aqi_value <= 400:
        return 'Very Unhealthy'
    else:
        return 'Hazardous'


# Disease Mapping Function
def map_disease(aqi_value):
    if aqi_value <= 50:
        return "No significant health risks expected."
    elif 51 <= aqi_value <= 100:
        return "Mild respiratory issues for sensitive individuals."
    elif 101 <= aqi_value <= 200:
        return "Potential respiratory discomfort and lung irritation."
    elif 201 <= aqi_value <= 300:
        return "Increased risk of bronchitis, asthma, or lung inflammation."
    elif 301 <= aqi_value <= 400:
        return "Severe respiratory problems, heart diseases for sensitive groups."
    else:
        return "Critical health conditions such as COPD, heart attacks, and lung cancer risks."


# Prediction Function
def predict_aqi(date, location):
    # Filter data for the given location
    location_data = df[df['location'].str.lower() == location.lower()]

    if location_data.empty:
        return None, None, f"No data available for the location '{location}'. Please check the input."

    # Compute average pollutant levels for the given location
    avg_pollutants = location_data[['so2', 'no2', 'rspm', 'spm']].mean().values.reshape(1, -1)

    # Impute missing values
    avg_pollutants_imputed = imputer_X.transform(avg_pollutants)

    # Predict AQI
    predicted_aqi = knn_regressor.predict(avg_pollutants_imputed)[0]
    predicted_category = categorize_aqi(predicted_aqi)
    predicted_disease = map_disease(predicted_aqi)

    return predicted_aqi, predicted_category, predicted_disease


# Streamlit Interface
st.title("Air Quality Index (AQI) Prediction with Health Insights")

# User Input
st.header("Enter Details for Prediction")
date_input = st.date_input("Select a date", value=datetime.now().date())
location_input = st.text_input("Enter the location", placeholder="E.g., Chennai, Mumbai")

# Prediction Button
if st.button("Predict AQI"):
    if location_input:
        predicted_aqi, category, disease = predict_aqi(str(date_input), location_input)

        if predicted_aqi is not None:
            st.success(f"Predicted AQI for {location_input} on {date_input}: {predicted_aqi:.2f}")
            st.info(f"AQI Category: {category}")
            st.warning(f"Health Impact: {disease}")
        else:
            st.error(disease)
    else:
        st.error("Please enter a valid location.")
