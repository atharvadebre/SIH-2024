from threading import Thread

import streamlit as st
import joblib
import numpy as np
import requests
import json
import plotly.graph_objects as go
from fastapi import FastAPI, Request
import uvicorn

# Load the model and label encoders
data = joblib.load('final_ass.pkl')
model = data['model']
label_encoders = data['label_encoders']

# Define mappings for any required manual adjustments
crop_type_mapping = {key: idx for idx, key in enumerate(label_encoders['CROP TYPE'].classes_)}
soil_type_mapping = {key: idx for idx, key in enumerate(label_encoders['SOIL TYPE'].classes_)}
region_mapping = {key: idx for idx, key in enumerate(label_encoders['REGION'].classes_)}
weather_condition_mapping = {key: idx for idx, key in enumerate(label_encoders['WEATHER CONDITION'].classes_)}

app = FastAPI()

@app.post("/train")
async def train_model(request: Request):
    # Extract data from the POST request
    request_data = await request.json()
    features = request_data['features']
    target = request_data['target']

    # Convert data to the appropriate format (e.g., numpy array)
    X = np.array(features).reshape(1, -1)
    y = np.array(target)

    # Update or train the model (assuming the model supports partial_fit)
    model.partial_fit(X, y)

    # Optionally save the updated model
    joblib.dump({'model': model, 'label_encoders': label_encoders}, 'final_ass.pkl')

    return {"message": "Model updated successfully"}

def run_fastapi():
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Start FastAPI in a separate thread
fastapi_thread = Thread(target=run_fastapi)
fastapi_thread.start()

def get_weather(city):
    api_key = "b3c62ae7f7ad5fc3cb0a7b56cb7cbda6"
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for any HTTP errors
    except requests.exceptions.HTTPError as err:
        st.error(f"Error: {err}")
        return None, None, None, None, None

    try:
        data = json.loads(response.text)
        if data['cod'] != 200:
            st.error(f"Error: {data['message']}")
            return None, None, None, None, None
    except json.JSONDecodeError as err:
        st.error(f"Error: Failed to parse response JSON - {err}")
        return None, None, None, None, None

    # Extract relevant weather information
    weather_description = data['weather'][0]['description']
    temperature = data['main']['temp']
    humidity = data['main']['humidity']
    pressure = data['main']['pressure']
    # Rainfall is not provided in the response, need to use an appropriate field
    rainfall = data.get('rain', {}).get('1h', 0)  # Rainfall in the last 1 hour, default to 0 if not present
    # Convert temperature from Kelvin to Celsius
    temperature = round(temperature - 273.15, 2)

    return temperature, humidity, weather_description, pressure, rainfall

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

    fig.update_layout(
        height=400,
        width=600,
        margin={"r": 40, "t": 60, "l": 40, "b": 20},
        paper_bgcolor="black",
        plot_bgcolor="black",
        title={
            'text': "Water Requirement (litres/sq.m)",
            'font': {
                'size': 18,
                'color': 'white'
            },
            'x': 0.5,
            'xanchor': 'center'
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
    global rainfall
    st.set_page_config(page_title="Agricultural Prediction System", page_icon="🌾", layout="wide")

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

    if 'water_requirement' not in st.session_state:
        st.session_state.water_requirement = 0.0

    if 'motor_capacity' not in st.session_state:
        st.session_state.motor_capacity = 0.0

    st.markdown("<div class='title'>Crop Water Requirement Prediction</div>", unsafe_allow_html=True)
    st.markdown("<div class='subheader'>Enter the following details:</div>", unsafe_allow_html=True)

    crop_type = st.selectbox("Crop Type", list(crop_type_mapping.keys()))
    soil_type = st.selectbox("Soil Type", list(soil_type_mapping.keys()))
    region = st.selectbox("Region Type", list(region_mapping.keys()))

    city = st.text_input("Enter your city to get the weather details")

    if city and st.button("Get Weather"):
        temperature, humidity, weather_description, pressure, api_rainfall = get_weather(city)
        if temperature is not None and humidity is not None:
            st.markdown(f"<div class='weather-container'>Weather in {city}: {weather_description.capitalize()}<br>"
                        f"Temperature: {temperature}°C<br>"
                        f"Humidity: {humidity}%<br>"
                        f"Pressure: {pressure} hPa<br>"
                        f"Rainfall: {api_rainfall} mm<br>",
                        unsafe_allow_html=True)

            if any(word in weather_description.lower() for word in ['sunny', 'clear', 'warm', 'hot']):
                auto_weather_condition = 'SUNNY'
            elif any(word in weather_description.lower() for word in ['rain', 'drizzle', 'cloud', 'clouds', 'mist']):
                auto_weather_condition = 'RAINY'
            elif any(word in weather_description.lower() for word in ['wind', 'winds', 'haze']):
                auto_weather_condition = 'WINDY'
            else:
                auto_weather_condition = 'NORMAL'
        else:
            auto_weather_condition = 'NORMAL'
            temperature = 32.0
            api_rainfall = 0.0
    else:
        auto_weather_condition = 'NORMAL'
        temperature = 32.0
        api_rainfall = 0.0

    col1, col2 = st.columns(2)
    with col1:
        weather_condition = st.selectbox("Weather Condition", list(weather_condition_mapping.keys()),
                                         index=list(weather_condition_mapping.keys()).index(auto_weather_condition))
    with col2:
        temperature_input = st.number_input("Temperature (°C)", value=temperature)
    soil_moisture = st.number_input("Soil Moisture (%)", value=30.0)
    humidity_input = st.number_input("Humidity (%)", value=60.0)
    rainfall_input = st.number_input("Rainfall (mm)", value=api_rainfall)  # Use 0.0 if not availabl

    if st.button("Predict Water Requirement"):
        # Encode inputs for the model
        encoded_input = [
            label_encoders['CROP TYPE'].transform([crop_type])[0],
            label_encoders['SOIL TYPE'].transform([soil_type])[0],
            label_encoders['REGION'].transform([region])[0],
            temperature_input,
            label_encoders['WEATHER CONDITION'].transform([weather_condition])[0],
            soil_moisture,
            humidity_input,
            rainfall_input
        ]

        # Convert to numpy array and reshape for prediction
        encoded_input = np.array(encoded_input).reshape(1, -1)

        # Make the prediction
        water_requirement = model.predict(encoded_input)[0]
        st.session_state.water_requirement = water_requirement

        st.markdown(f"<div class='prediction-box'>The estimated water requirement is: {water_requirement:.2f} litres/sq.m</div>",
                    unsafe_allow_html=True)

    # Display the gauge chart if water requirement is predicted
    if st.session_state.water_requirement > 0.0:
        st.plotly_chart(plot_water_requirement_gauge(st.session_state.water_requirement))

    st.markdown("<div class='subheader'>Motor Capacity (in Horsepower):</div>", unsafe_allow_html=True)
    motor_capacity = st.number_input("Motor Capacity (HP)", value=st.session_state.motor_capacity)

    if st.session_state.water_requirement > 0:
        if st.button("Calculate Irrigation Time"):
            # Assuming motor efficiency is 75% and flow rate is 0.1 litres per HP per second
            flow_rate = motor_capacity * 0.1 * 0.75
            irrigation_time = st.session_state.water_requirement / flow_rate

            st.markdown(f"<div class='prediction-box'>Estimated irrigation time: {irrigation_time:.2f} seconds/sq.m</div>",
                        unsafe_allow_html=True)

if __name__ == '__main__':
    main()
