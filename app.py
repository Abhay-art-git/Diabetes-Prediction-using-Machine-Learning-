import streamlit as st
import pickle
import numpy as np

# Load the model
with open('diabetes_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the app title
st.title('Diabetes Prediction')

# Add sliders and other input widgets for the user to enter data
age = st.slider('Age', min_value=0, max_value=80, value=30, step=1)
hypertension = st.selectbox('Hypertension', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
heart_disease = st.selectbox('Heart Disease', [0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
bmi = st.slider('BMI', min_value=10.0, max_value=95.0, value=27.3, step=0.1)
HbA1c_level = st.slider('HbA1c Level', min_value=3.5, max_value=9.0, value=5.5, step=0.1)
blood_glucose_level = st.slider('Blood Glucose Level', min_value=80.0, max_value=300.0, value=140.0, step=1.0)

# Arrange input data into a numpy array
input_data = np.array([[age, hypertension, heart_disease, bmi, HbA1c_level, blood_glucose_level]])

# Predict using the loaded model
if st.button('Predict'):
    prediction = model.predict(input_data)
    
    # Display the prediction
    if prediction[0] == 1:
        st.write("The model predicts that the person is likely to have diabetes.")
    else:
        st.write("The model predicts that the person is unlikely to have diabetes.")

# To run this app:
# streamlit run app.py 
