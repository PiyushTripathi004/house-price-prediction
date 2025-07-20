import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('real_estate_model.pkl')

st.title("California Housing Price Predictor")

features = [
    'MedInc',     # Median income in block group
    'HouseAge',   # Median house age
    'AveRooms',   # Average rooms per household
    'AveBedrms',  # Average bedrooms per household
    'Population', # Block group population
    'AveOccup',   # Average household occupancy
    'Latitude',   # Latitude
    'Longitude'   # Longitude
]
input_data = []
st.markdown("Enter details for one housing block group:")

for feature in features:
    value = st.number_input(f"Enter {feature}:", value=0.0)
    input_data.append(value)

if st.button("Predict"):
    input_df = pd.DataFrame([input_data], columns=features)
    prediction = model.predict(input_df)[0]
    st.balloons()
    st.success(f"Estimated Median House Value: ${prediction * 100000:.2f}")
