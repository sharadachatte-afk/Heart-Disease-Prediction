
import streamlit as st
import joblib
import pandas as pd

# Load saved files
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler_heart.pkl")
columns = joblib.load("columns_heart.pkl")

# Title
st.title("❤️ Heart Disease Prediction")

st.write("Enter the patient's information below:")

# Input fields
age = st.slider("Age", 18, 100, 40)

sex = st.selectbox("Sex", ["M", "F"])

chest_pain = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "ASY", "TA"]
)

resting_bp = st.number_input(
    "Resting Blood Pressure",
    min_value=0,
    max_value=250,
    value=120
)

cholesterol = st.number_input(
    "Cholesterol",
    min_value=0,
    max_value=600,
    value=200
)

fasting_bs = st.selectbox(
    "Fasting Blood Sugar",
    [0, 1]
)

resting_ecg = st.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"]
)

max_hr = st.slider(
    "Maximum Heart Rate",
    60,
    220,
    150
)

exercise_angina = st.selectbox(
    "Exercise Angina",
    ["N", "Y"]
)

oldpeak = st.number_input(
    "Oldpeak",
    min_value=0.0,
    max_value=10.0,
    value=1.0
)

st_slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)

# Prediction button
if st.button("Predict Heart Disease"):

    input_data = {
        "Age": age,
        "Sex": sex,
        "ChestPainType": chest_pain,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "RestingECG": resting_ecg,
        "MaxHR": max_hr,
        "ExerciseAngina": exercise_angina,
        "Oldpeak": oldpeak,
        "ST_Slope": st_slope
    }

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data])

    # Convert categorical variables
    input_df = pd.get_dummies(input_df, drop_first=True)

    # Match training columns
    input_df = input_df.reindex(
        columns=columns,
        fill_value=0
    )

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        st.error("⚠️ Heart Disease Detected")
    else:
        st.success("✅ No Heart Disease Detected")
