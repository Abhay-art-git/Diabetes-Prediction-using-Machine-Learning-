import streamlit as st
import pickle
import pandas as pd

# Load the model
with open('diabetes_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the app title
st.title('Diabetes Prediction')

# User Inputs
age = st.slider('Age', min_value=0, max_value=80, value=30, step=1)
hypertension = st.selectbox('Hypertension', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
heart_disease = st.selectbox('Heart Disease', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
bmi = st.slider('BMI', min_value=10.0, max_value=95.0, value=27.3, step=0.1)
HbA1c_level = st.slider('HbA1c Level', min_value=3.5, max_value=9.0, value=5.5, step=0.1)
blood_glucose_level = st.slider('Blood Glucose Level', min_value=80.0, max_value=300.0, value=140.0, step=1.0)
smoking_history = st.selectbox('Smoking History', ['non-smoker', 'past_smoker', 'current'], index=0)
gender = st.selectbox('Gender', ['Female', 'Male'], index=0)

# Keep input in original format (no manual encoding!)
input_data = pd.DataFrame([{
    'age': age,
    'hypertension': hypertension,
    'heart_disease': heart_disease,
    'bmi': bmi,
    'HbA1c_level': HbA1c_level,
    'blood_glucose_level': blood_glucose_level,
    'smoking_history': smoking_history,
    'gender': gender
}])

# Prediction
if st.button('Predict'):
    try:
        prediction = model.predict(input_data)
        if prediction[0] == 1:
            st.markdown(
        "<div style='background-color:#ff0000;padding:10px;border-radius:5px;'>"
        "<strong>The model predicts that the person is likely to have diabetes.</strong>"
        "</div>",
        unsafe_allow_html=True
        )
        else:
            st.success("The model predicts that the person is unlikely to have diabetes.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")