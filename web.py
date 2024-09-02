import streamlit as st
import joblib
import pandas as pd
import numpy as np
import requests
import json
import plotly.graph_objects as go

# Load the model
svm_poly_model = joblib.load('C:/Users/HP/Downloads/Crop_recommnedation_&_irrigation-20240831T163431Z-001/Crop_recommnedation_&_irrigation/svm_poly_model.pkl')


# Label encoding mappings for SVM model
crop_type_mapping = {'BANANA': 0, 'BEAN': 1, 'CABBAGE': 2, 'CITRUS': 3, 'COTTON': 4, 'MAIZE': 5, 'MELON': 6,
                     'MUSTARD': 7, 'ONION': 8, 'OTHER': 9, 'POTATO': 10, 'RICE': 11, 'SOYABEAN': 12, 'SUGARCANE': 13,
                     'TOMATO': 14, 'WHEAT': 15}
soil_type_mapping = {'DRY': 0, 'HUMID': 1, 'WET': 2}
weather_condition_mapping = {'NORMAL': 0, 'RAINY': 1, 'SUNNY': 2, 'WINDY': 3}

# Fetch weather data from the OpenWeatherMap API
def get_weather(city):
    api_key = "b3c62ae7f7ad5fc3cb0a7b56cb7cbda6"
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for any HTTP errors
    except requests.exceptions.HTTPError as err:
        st.error(f"Error: {err}")
        return None, None, None

    try:
        data = json.loads(response.text)
        if data['cod'] != 200:
            st.error(f"Error: {data['message']}")
            return None, None, None
    except json.JSONDecodeError as err:
        st.error(f"Error: Failed to parse response JSON - {err}")
        return None, None, None

    # Extract relevant weather information
    weather_description = data['weather'][0]['description']
    temperature = data['main']['temp']
    humidity = data['main']['humidity']
    pressure = data['main']['pressure']
    # Convert temperature from Kelvin to Celsius
    temperature = round(temperature - 273.15, 2)

    return temperature, humidity, weather_description, pressure

def plot_water_requirement_gauge(water_requirement):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=water_requirement,
        gauge={
            "axis": {"range": [None, 100]},  # Adjust range as needed
            "bar": {"color": "yellow"},
            "steps": [
                {"range": [0, 30], "color": "lightgreen"},
                {"range": [30, 70], "color": "orange"},
                {"range": [70, 100], "color": "red"}
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": water_requirement
            }
        }
    ))

    # Adjust the layout to centralize the gauge chart and set a single title
    fig.update_layout(
        height=400,  # Adjust height as needed
        width=600,   # Adjust width as needed
        margin={"r": 40, "t": 60, "l": 40, "b": 20},  # Adjust margins for proper centering
        paper_bgcolor="black",
        plot_bgcolor="black",
        title={
            'text': "Water Requirement (litres/sq.m)",  # Set the title text
            'font': {
                'size': 18,
                'color': 'white'
            },
            'x': 0.5,  # Center title horizontally
            'xanchor': 'center'  # Anchor title at the center
        },
        xaxis=dict(
            showticklabels=False,
            showgrid=False
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False
        ),
    )

    return fig


def main():
    st.set_page_config(page_title="Agricultural Prediction System", page_icon="🌾", layout="wide")

    # CSS for custom styling
    st.markdown("""
    <style>
    .stApp { 
        background-color: #000;
        color: #FFF;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton > button {
        border: 2px solid #FFF;
        border-radius: 5px;
        background-color: #003366;
        color: #FFD700;
        font-weight: bold;
    }
    .title {
        color: #FFD700;
        font-size: 36px;
        font-weight: bold;
        text-align: center;
    }
    .subheader {
        color: #FFD700;
        font-weight: bold;
        font-size: 24px;
        text-align: left;
    }
    .stSuccess > div {
        font-weight: bold;
        color: #000;
        background-color: #00FF00;
        border-radius: 5px;
        padding: 10px;
    }
    .stError > div {
        font-weight: bold;
        color: #000;
        background-color: #FF0000;
        border-radius: 5px;
        padding: 10px;
    }
    .container {
        display: flex;
        justify-content: space-between;
        margin-top: 20px;
        text-align: center;

    }
    .left-column {
        flex: 1;
        padding-right: 10px;
    }
    .right-column {
        flex: 1;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .map-img {
        width: 100%;
    }
    .map-container {
        width: 800px; /* Adjust width as needed */
        height: 600px; /* Adjust height as needed */
    }
    .prediction-box {
        border: 2px solid #FFD700;
        padding: 10px;
        border-radius: 10px;
        background-color: #003366;
        color: #FFD700;
        font-weight: bold;
        margin-top: 20px;
    }
    .weather-container {
        border: 2px solid #FFD700;
        padding: 15px;
        border-radius: 10px;
        background-color: #003366;
        color: #FFD700;
        font-weight: bold;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'water_requirement' not in st.session_state:
        st.session_state.water_requirement = 0.0

    if 'motor_capacity' not in st.session_state:
        st.session_state.motor_capacity = 0.0

    st.markdown("<div class='title'>Crop Water Requirement Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='subheader'>Enter the following details:</div>", unsafe_allow_html=True)

    crop_type = st.selectbox("Crop Type", list(crop_type_mapping.keys()))
    soil_type = st.selectbox("Soil Type", list(soil_type_mapping.keys()))

    city = st.text_input("Enter your city to get the weather details (optional)")

    if city and st.button("Get Weather"):
        temperature, humidity, weather_description, pressure = get_weather(city)
        if temperature is not None and humidity is not None:
            st.markdown(f"<div class='weather-container'>Weather in {city}: {weather_description.capitalize()}<br>"
                        f"Temperature: {temperature}°C<br>"
                        f"Humidity: {humidity}%<br>"
                        f"Pressure: {pressure}hg<br>",
                        unsafe_allow_html=True)

            # Auto-fill weather condition based on description
            if any(word in weather_description.lower() for word in ['sunny', 'clear', 'warm', 'hot']):
                auto_weather_condition = 'SUNNY'
            elif any(
                    word in weather_description.lower() for word in ['rain', 'drizzle', 'cloud', 'clouds', 'mist']):
                auto_weather_condition = 'RAINY'
            elif any(word in weather_description.lower() for word in ['wind', 'winds', 'haze']):
                auto_weather_condition = 'WINDY'
            else:
                auto_weather_condition = 'NORMAL'
        else:
            auto_weather_condition = 'NORMAL'
            temperature = 32.0

    else:
        auto_weather_condition = 'NORMAL'
        temperature = 32.0

    # User input for crop water requirement prediction
    col1, col2 = st.columns(2)
    with col1:
        weather_condition = st.selectbox("Weather Condition", list(weather_condition_mapping.keys()),
                                         index=list(weather_condition_mapping.keys()).index(auto_weather_condition))
    with col2:
        temperature_input = st.number_input("Temperature (°C)", value=temperature)

    # Predict button
    if st.button("Predict Water Requirement"):
        # Encode inputs for the model
        crop_type_encoded = crop_type_mapping[crop_type]
        soil_type_encoded = soil_type_mapping[soil_type]
        weather_condition_encoded = weather_condition_mapping[weather_condition]

        # Create the input feature array
        features = np.array([[crop_type_encoded, soil_type_encoded, weather_condition_encoded, temperature_input]])

        # Make the prediction
        water_requirement = svm_poly_model.predict(features)[0]
        st.session_state.water_requirement = water_requirement

        st.markdown(f"<div class='prediction-box'>The estimated water requirement is: {water_requirement:.2f} litres/sq.m</div>",
                    unsafe_allow_html=True)

    # Display the gauge chart if water requirement is predicted
    if st.session_state.water_requirement > 0.0:
        st.plotly_chart(plot_water_requirement_gauge(st.session_state.water_requirement))

    st.markdown(
        "<div class='subheader'>Motor Capacity (in Horsepower):</div>", unsafe_allow_html=True)
    motor_capacity = st.number_input("Motor Capacity (HP)", value=st.session_state.motor_capacity)

    # Only display if water requirement has been calculated
    if st.session_state.water_requirement > 0:
        if st.button("Calculate Irrigation Time"):
            # Assuming motor efficiency is 75% and flow rate is 0.1 litres per HP per second
            flow_rate = motor_capacity * 0.1 * 0.75
            irrigation_time = st.session_state.water_requirement / flow_rate

            st.markdown(f"<div class='prediction-box'>Estimated irrigation time: {irrigation_time:.2f} seconds/sq.m</div>",
                        unsafe_allow_html=True)

if __name__ == '__main__':
    main()
