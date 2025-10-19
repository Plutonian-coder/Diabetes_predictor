import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai
import os
import requests

# --- API Key Configuration ---
# It is highly recommended to store your API keys securely.
# In Colab, you can use "Secrets" (the key icon on the left sidebar).
# Add GEMINI_API_KEY and BYTEZ_API_KEY to your Colab Secrets.
# Alternatively, you can set them as environment variables in your system.

# Using the API key provided by the user directly for now.
# For better security, consider using Colab Secrets.
GEMINI_API_KEY = "AIzaSyB278rf_ZliONHPV-rc2wGE9k0INIGCiyE"
BYTEZ_API_KEY = os.getenv("BYTEZ_API_KEY") # Still loading Bytez key from environment variable

# Check if API keys are loaded
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found. Please set it in your Colab Secrets or environment variables.")
    st.stop()

if not BYTEZ_API_KEY:
    st.error("BYTEZ_API_KEY not found. Please set it in your Colab Secrets or environment variables.")
    st.stop()

# Configure Gemini API key
genai.configure(api_key=GEMINI_API_KEY)

# Define Bytez API endpoint
BYTEZ_API_URL = "https://api.bytez.ai/v1/image/generate" # Replace with actual Bytez API endpoint if different


# Load the trained model and scaler
try:
    with open('lgbm_model.pkl', 'rb') as model_file:
        lgbm_classifier = pickle.load(model_file)
except FileNotFoundError:
    st.error("Model file 'lgbm_model.pkl' not found. Please train and save the model first.")
    st.stop()

try:
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
except FileNotFoundError:
    st.error("Scaler file 'scaler.pkl' not found. Please train and save the scaler first.")
    st.stop()

# Function to generate image prompt based on prediction
def generate_image_prompt(prediction_text):
    if prediction_text == "not diabetic":
        return "a person enjoying a healthy lifestyle, vibrant and happy"
    else:
        return "a person managing diabetes with a healthy diet and exercise"

# Function to generate image using Bytez API
def generate_bytez_image(prompt, api_key):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "prompt": prompt,
        "n_images": 1,
        "size": "512x512" # Or other desired size
    }
    try:
        response = requests.post(BYTEZ_API_URL, headers=headers, json=data)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Bytez API error: {e}")
        return None


# Set the title and add introductory text
st.title('🩺 Diabetes Prediction App')
st.markdown("""
This application predicts the likelihood of diabetes based on the health metrics you provide, provides an explanation using Gemini, and generates a relevant image.
Please fill in the details below and click 'Predict Diabetes'.
""")

# Create input fields with improved layout
st.header('Enter Patient Health Metrics')

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=0, help='Number of times pregnant')
    glucose = st.number_input('Glucose', min_value=0, max_value=200, value=100, help='Plasma glucose concentration a 2 hours in an oral glucose tolerance test')
    blood_pressure = st.number_input('Blood Pressure', min_value=0, max_value=130, value=70, help='Diastolic blood pressure (mm Hg)')
    skin_thickness = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20, help='Triceps skin fold thickness (mm)')

with col2:
    insulin = st.number_input('Insulin', min_value=0, max_value=900, value=80, help='2-Hour serum insulin (mu U/ml)')
    bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0, help='Body mass index (weight in kg/(height in m)^2)')
    dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5, help='Diabetes pedigree function')
    age = st.number_input('Age', min_value=0, max_value=120, value=30, help='Age in years')

# Create a prediction button
if st.button('Predict Diabetes'):
    # Gather input values
    input_data = (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age)

    # Convert to numpy array and reshape
    input_np = np.asarray(input_data)
    input_data_reshaped = input_np.reshape(1, -1)

    # Standardize the input data
    if input_data_reshaped.shape[1] != scaler.n_features_in_:
         st.error(f"Input data has {input_data_reshaped.shape[1]} features, but the scaler was fitted with {scaler.n_features_in_} features.")
    else:
        std_data = scaler.transform(input_data_reshaped)

        # Make prediction
        prediction = lgbm_classifier.predict(std_data)

        # Display the result with enhanced visuals
        st.subheader('Prediction Result:')
        if prediction[0] == 0:
            st.balloons()
            st.success('🎉 Great news! Based on the provided metrics, the model predicts that the person is **NOT** diabetic.')
            prediction_text = "not diabetic"
        else:
            st.warning('⚠️ Based on the provided metrics, the model predicts that the person **IS** diabetic. Please consult a healthcare professional for diagnosis.')
            prediction_text = "diabetic"

        # Display Explanation and Image sections
        st.markdown("---") # Add a separator
        st.header('Additional Information')

        col3, col4 = st.columns(2)

        with col3:
            # Generate explanation using Gemini
            st.subheader('Explanation:')
            try:
                model = genai.GenerativeModel('gemini-pro')
                prompt = f"Explain the following diabetes prediction in simple terms for a user. The input features were: Pregnancies={pregnancies}, Glucose={glucose}, BloodPressure={blood_pressure}, SkinThickness={skin_thickness}, Insulin={insulin}, BMI={bmi}, DiabetesPedigreeFunction={dpf}, Age={age}. The prediction is that the person is {prediction_text}. Briefly mention which of these factors might have contributed to the prediction."
                response = model.generate_content(prompt)
                explanation = response.text
                st.write(explanation)
            except Exception as e:
                st.error(f"Failed to generate explanation: {e}")

        with col4:
            # Generate and display image using Bytez API
            st.subheader('Related Image:')
            image_prompt = generate_image_prompt(prediction_text)
            image_response = generate_bytez_image(image_prompt, BYTEZ_API_KEY)

            if image_response and image_response.get("data"):
                image_url = image_response["data"][0]["url"]
                st.image(image_url, caption=image_prompt)
            else:
                st.warning("Could not generate a relevant image at this time.")
