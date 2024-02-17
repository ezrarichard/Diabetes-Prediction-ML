# Import necessary libraries
import streamlit as st
import numpy as np
import joblib

# Load the trained model (ensure the model file is in the correct directory)
model = joblib.load('dia_risk_prediction_model.pkl')

def predict_diabetes(input_data):
    # Make prediction using the loaded model
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)
    return prediction, probability

# Streamlit app layout 
st.title('Diabetes Risk Prediction')

# Form for user input
with st.form("prediction_form"):
    st.subheader("Enter the details:")
    age = st.number_input('Age', min_value=1, max_value=120, value=30)
    
    # Assuming gender, polyuria, etc., are categorical and binary
    gender = st.selectbox('Gender', options=['Male', 'Female'])
    polyuria = st.selectbox('Polyuria', options=['Yes', 'No'])
    polydipsia = st.selectbox('Polydipsia', options=['Yes', 'No'])
    sudden_weight_loss = st.selectbox('Sudden Weight Loss', options=['Yes', 'No'])
    weakness = st.selectbox('Weakness', options=['Yes', 'No'])
    polyphagia = st.selectbox('Polyphagia', options=['Yes', 'No'])
    genital_thrush = st.selectbox('Genital Thrush', options=['Yes', 'No'])
    visual_blurring = st.selectbox('Visual Blurring', options=['Yes', 'No'])
    itching = st.selectbox('Itching', options=['Yes', 'No'])
    irritability = st.selectbox('Irritability', options=['Yes', 'No'])
    delayed_healing = st.selectbox('Delayed Healing', options=['Yes', 'No'])
    partial_paresis = st.selectbox('Partial Paresis', options=['Yes', 'No'])
    muscle_stiffness = st.selectbox('Muscle Stiffness', options=['Yes', 'No'])
    alopecia = st.selectbox('Alopecia', options=['Yes', 'No'])
    obesity = st.selectbox('Obesity', options=['Yes', 'No'])
    
    # Convert inputs to the format your model expects
    inputs = np.array([[age,
                        1 if gender == 'Male' else 0,
                        1 if polyuria == 'Yes' else 0,
                        1 if polydipsia == 'Yes' else 0,
                        1 if sudden_weight_loss == 'Yes' else 0,
                        1 if weakness == 'Yes' else 0,
                        1 if polyphagia == 'Yes' else 0,
                        1 if genital_thrush == 'Yes' else 0,
                        1 if visual_blurring == 'Yes' else 0,
                        1 if itching == 'Yes' else 0,
                        1 if irritability == 'Yes' else 0,
                        1 if delayed_healing == 'Yes' else 0,
                        1 if partial_paresis == 'Yes' else 0,
                        1 if muscle_stiffness == 'Yes' else 0,
                        1 if alopecia == 'Yes' else 0,
                        1 if obesity == 'Yes' else 0]])
    
    submitted = st.form_submit_button("Predict")
    if submitted:
        prediction, probability = predict_diabetes(inputs)
        
        # Display results
        st.write(f"Prediction: {'Positive' if prediction[0] == 1 else 'Negative'}")
        st.write(f"Probability of being Positive: {probability[0][1]:.2f}")

