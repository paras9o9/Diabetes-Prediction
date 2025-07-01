import streamlit as st
import numpy as np
import pickle
import os

current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, 'knn_model.pkl')

# Load model
with open(model_path, 'rb') as f:
    scaler, model = pickle.load(f)

# App Title
st.title("🧪 Diabetes Prediction App")
st.write("Enter your health metrics below to predict your diabetes risk.")

# User inputs
preg = st.number_input("Pregnancies", min_value=0)
glucose = st.number_input("Glucose", min_value=0)
bp = st.number_input("Blood Pressure", min_value=0)
skin = st.number_input("Skin Thickness", min_value=0)
insulin = st.number_input("Insulin", min_value=0)
bmi = st.number_input("BMI", min_value=0.0, format="%.1f")
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, format="%.3f")
age = st.number_input("Age", min_value=1)

if st.button("Predict"):
    input_data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    prediction = model.predict(input_data)
    result = "🟢 Not Diabetic" if prediction[0] == 0 else "🔴 Diabetic"
    st.subheader(f"Prediction: {result}")
