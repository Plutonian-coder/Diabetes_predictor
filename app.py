import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai
import os
import json # Import json for safe parsing
from gradio_client import Client
import re # For simple text parsing

# --- API Key Configuration ---
# *** CRITICAL: Using st.secrets for secure API key handling. 
# Make sure GEMINI_API_KEY is set in your Streamlit Cloud secrets. ***
try:
    GEMINI_API_KEY = "AIzaSyB278rf_ZliONHPV-rc2wGE9k0INIGCiyE"
    genai.configure(api_key=GEMINI_API_KEY)
except KeyError:
    st.error("🔑 **GEMINI_API_KEY** not found in Streamlit Secrets. Please add it to your app settings for the Gemini AI to function.")
    st.stop()
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}")
    st.stop()


# --- Gradio Client for Image Generation ---
# *** MODIFIED: Initialized Client with "NihalGazi/FLUX-Unlimited" ***
try:
    # Initialize the Gradio Client for the FLUX-Unlimited Space
    image_gen_client = Client("NihalGazi/FLUX-Unlimited")
    st.success("Successfully connected to NihalGazi/FLUX-Unlimited image generation service.")
except Exception as e:
    st.error(f"Failed to connect to NihalGazi/FLUX-Unlimited Gradio Space: {e}. Please ensure the Space is running and accessible.")
    st.stop()

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
    # Adjusting prompts for high-resolution generation
    if prediction_text == "not diabetic":
        return "A full-body shot of a vibrant, healthy person running in a sunny park with clear blue sky. Cinematic, highly detailed, photorealistic, 8k."
    else:
        return "A thoughtful portrait of a person monitoring their health, holding an apple, in a serene, modern kitchen. Focus on warmth and determination. Photorealistic, detailed, empowering mood, 8k."

# *** MODIFIED: Updated parameters for NihalGazi/FLUX-Unlimited model ***
def generate_image(prompt):
    st.info("Generating a relevant image... Please wait. (Using FLUX-Unlimited)")
    try:
        # Parameters specific to the NihalGazi/FLUX-Unlimited client call
        result = image_gen_client.predict(
            prompt=prompt,
            width=1280,
            height=1280,
            seed=0,
            randomize=True,
            server_choice="Google US Server",
            api_name="/generate_image" # Specific API endpoint for this model
        )

        # The FLUX-Unlimited model's output format is usually a file path string or a list containing one.
        if isinstance(result, (list, tuple)) and len(result) > 0:
            image_path = result[0]
        else:
            image_path = result

        if image_path:
            return image_path
        else:
            st.error(f"Image generator did not return a valid path. Raw result: {result}")
            return None
    except Exception as e:
        st.error(f"Error generating image: {e}")
        return None

# Centralized prediction and result display function
def perform_prediction_and_display(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age):
    try:
        input_data = (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age)
        input_np = np.asarray(input_data)
        input_data_reshaped = input_np.reshape(1, -1)

        if input_data_reshaped.shape[1] != scaler.n_features_in_:
            st.error(f"Input data has {input_data_reshaped.shape[1]} features, but the scaler was fitted with {scaler.n_features_in_} features.")
            return

        std_data = scaler.transform(input_data_reshaped)
        prediction = lgbm_classifier.predict(std_data)

        st.subheader('Prediction Result:')
        if prediction[0] == 0:
            st.balloons()
            st.success('🎉 Great news! Based on the provided metrics, the model predicts that the person is **NOT** diabetic.')
            prediction_text = "not diabetic"
        else:
            st.warning('⚠️ Based on the provided metrics, the model predicts that the person **IS** diabetic. Please consult a healthcare professional for diagnosis.')
            prediction_text = "diabetic"

        st.markdown("---")
        st.header('Additional Information')

        col_exp, col_img = st.columns(2)

        with col_exp:
            st.subheader('Explanation (powered by Gemini):')
            try:
                # Enhanced medical persona prompt
                model = genai.GenerativeModel('gemini-2.0-flash')
                prompt = f"""
                You are a compassionate medical AI assistant.
                A patient has just received a prediction from a machine learning model. Your role is to explain this result in a professional, empathetic, and easy-to-understand manner.
                IMPORTANT:
                1.  Do NOT give medical advice. Always end your explanation by advising the user to consult a healthcare professional.
                2.  Base your explanation *only* on the provided data.
                3.  Briefly mention which of the input factors might be contributing.
                Patient Data:
                -   Pregnancies: {pregnancies}, Glucose: {glucose}, Blood Pressure: {blood_pressure},
                -   Skin Thickness: {skin_thickness}, Insulin: {insulin}, BMI: {bmi},
                -   Diabetes Pedigree Function: {dpf}, Age: {age}
                Model Prediction: The person is **{prediction_text}**.
                Please provide your explanation now.
                """
                
                response = model.generate_content(prompt)
                explanation = response.text
                st.write(explanation)
            except Exception as e:
                st.error(f"Failed to generate explanation using Gemini. Error: {e}.")

        with col_img:
            st.subheader('Related Image (AI Generated):')
            image_prompt = generate_image_prompt(prediction_text)
            image_path = generate_image(image_prompt) # Using the new function

            if image_path:
                st.image(image_path, caption=image_prompt, use_column_width=True)
            else:
                st.warning("Could not generate a relevant image at this time.")
                
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.warning("The ML model could not run. Please check the inputs or contact support.")


# --- Streamlit UI Design ---
st.set_page_config(layout="wide", page_title="AI-Powered Diabetes Predictor")

st.title('⚕️ AI-Powered Diabetes Predictor & Wellness Assistant')
st.markdown("""
Welcome to your personal health companion. You can either use our **AI Assistant** to describe your health concerns naturally, or input your **Health Metrics Manually** for a precise prediction.
""")

# Create tabs for different modes
tab1, tab2 = st.tabs(["🤖 AI Assistant (Beta)", "✍️ Manual Health Metrics Input"])

with tab1:
    st.header("Tell Me About Your Health...")
    st.markdown("""
    Describe your current health status, any symptoms, or provide any known health metrics in a natural language.
    Our AI will provide a general assessment and then try to run a prediction if enough data is found.
    """)
    user_description = st.text_area(
        "Describe your health here:",
        "I'm a 35-year-old female. My glucose level was 140, and BMI is 30. I have no history of pregnancies. My blood pressure is around 85.",
        height=150
    )

    if st.button('Get AI Insights', key='ai_predict_button'):
        st.info("Processing your description with Gemini AI...")
        
        # --- Gemini as a "Fallback" Medical Practitioner ---
        try:
            gemini_model = genai.GenerativeModel('gemini-2.0-flash')
            prompt_for_assessment = f"""
            You are a medical assistant AI. A user has provided the following health description.
            Please provide a brief, general assessment based *only* on their text.
            - If they mention concerning symptoms (like high glucose, high BMI), gently acknowledge them.
            - **Do NOT diagnose them.**
            - **Always** recommend they speak to a healthcare professional for proper advice.
            - Keep it to one short paragraph.
            User description: "{user_description}"
            """
            response_assessment = gemini_model.generate_content(prompt_for_assessment)
            st.subheader("Gemini's General Assessment:")
            st.write(response_assessment.text)
            st.markdown("---")

        except Exception as e:
            st.warning(f"Could not get general assessment from Gemini: {e}")

        
        # --- Second, try to extract metrics for the ML model ---
        try:
            # Use Gemini to extract or confirm metrics
            prompt_for_parsing = f"""
            The user provided the following health description: "{user_description}"
            Please extract the following metrics.
            Format your response as a single, clean JSON string with keys: "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age".
            
            - If a value is not explicitly mentioned, use a typical default (e.g., SkinThickness: 20, Insulin: 80, DiabetesPedigreeFunction: 0.5).
            - For Pregnancies, if not mentioned, assume 0.
            - For Age, if not mentioned, assume 30.
            
            Example response: {{"Pregnancies": 1, "Glucose": 120, "BloodPressure": 70, "SkinThickness": 20, "Insulin": 80, "BMI": 25.0, "DiabetesPedigreeFunction": 0.5, "Age": 30}}
            
            Respond ONLY with the JSON string.
            """
            response_parsing = gemini_model.generate_content(prompt_for_parsing)
            extracted_metrics_str = response_parsing.text.strip()

            # Clean up potential markdown formatting
            if extracted_metrics_str.startswith("```json") and extracted_metrics_str.endswith("```"):
                extracted_metrics_str = extracted_metrics_str[7:-3].strip()

            st.subheader("AI's Interpretation of Your Metrics (for Model Prediction):")
            try:
                # Use json.loads() for safety
                extracted_metrics = json.loads(extracted_metrics_str) 
                st.json(extracted_metrics) 

                # Perform prediction with extracted metrics
                st.subheader("Running Diabetes Prediction based on AI-Parsed Data...")
                perform_prediction_and_display(
                    extracted_metrics.get("Pregnancies", 0),
                    extracted_metrics.get("Glucose", 100),
                    extracted_metrics.get("BloodPressure", 70),
                    extracted_metrics.get("SkinThickness", 20),
                    extracted_metrics.get("Insulin", 80),
                    extracted_metrics.get("BMI", 25.0),
                    extracted_metrics.get("DiabetesPedigreeFunction", 0.5),
                    extracted_metrics.get("Age", 30)
                )
            except Exception as json_e:
                st.error(f"AI had difficulty parsing the metrics into a usable format: {json_e}")
                st.warning("The ML model could not be run. Please review the General Assessment above or use the manual input tab.")
                st.code(extracted_metrics_str) # Show raw AI output for debugging
        except Exception as e:
            st.error(f"Error communicating with Gemini AI for insights: {e}")

with tab2:
    st.header('Enter Patient Health Metrics Manually')
    st.markdown("""
    Please input the specific health metrics below to get a precise diabetes prediction.
    """)

    # Use Streamlit columns for better layout
    col1_manual, col2_manual = st.columns(2)

    with col1_manual:
        pregnancies_manual = st.number_input('Pregnancies', min_value=0, max_value=20, value=0, help='Number of times pregnant', key='manual_pregnancies')
        glucose_manual = st.number_input('Glucose', min_value=0, max_value=200, value=100, help='Plasma glucose concentration a 2 hours in an oral glucose tolerance test', key='manual_glucose')
        blood_pressure_manual = st.number_input('Blood Pressure', min_value=0, max_value=130, value=70, help='Diastolic blood pressure (mm Hg)', key='manual_bp')
        skin_thickness_manual = st.number_input('Skin Thickness', min_value=0, max_value=100, value=20, help='Triceps skin fold thickness (mm)', key='manual_skin')

    with col2_manual:
        insulin_manual = st.number_input('Insulin', min_value=0, max_value=900, value=80, help='2-Hour serum insulin (mu U/ml)', key='manual_insulin')
        bmi_manual = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0, help='Body mass index (weight in kg/(height in m)^2)', key='manual_bmi')
        dpf_manual = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5, help='Diabetes pedigree function', key='manual_dpf')
        age_manual = st.number_input('Age', min_value=0, max_value=120, value=30, help='Age in years', key='manual_age')

    if st.button('Predict Diabetes Manually', key='manual_predict_button'):
        perform_prediction_and_display(
            pregnancies_manual, glucose_manual, blood_pressure_manual, skin_thickness_manual,
            insulin_manual, bmi_manual, dpf_manual, age_manual
        )