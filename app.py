import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
import google.generativeai as genai
import os
from gradio_client import Client # Import gradio_client
import re # For simple text parsing

# --- API Key Configuration ---
# It is highly recommended to store your API keys securely.
# In Colab, you can use "Secrets" (the key icon on the left sidebar).
# Add GEMINI_API_KEY to your Colab Secrets.
# Alternatively, you can set it as an environment variable in your system.

# Using the API key provided by the user directly for now.
# For better security, consider using Colab Secrets.
GEMINI_API_KEY = "AIzaSyB278rf_ZliONHPV-rc2wGE9k0INIGCiyE"

# Check if API keys are loaded
if not GEMINI_API_KEY:
    st.error("GEMINI_API_KEY not found. Please set it in your Colab Secrets or environment variables.")
    st.stop()

# Configure Gemini API key for text generation
genai.configure(api_key=GEMINI_API_KEY)

# --- Gradio Client for Image Generation ---
try:
    # Initialize the Gradio Client for FLUX.1-dev Space
    flux_client = Client("black-forest-labs/FLUX.1-dev")
    st.success("Successfully connected to FLUX.1-dev image generation service.")
except Exception as e:
    st.error(f"Failed to connect to FLUX.1-dev Gradio Space: {e}. Please ensure the Space is running and accessible.")
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
    if prediction_text == "not diabetic":
        return "A vibrant and happy person enjoying a healthy lifestyle, perhaps jogging in a park or eating a balanced meal. Photorealistic, high quality, optimistic mood, bright colors."
    else:
        return "A person thoughtfully managing diabetes, perhaps engaging in light exercise, preparing a healthy meal, or consulting with a healthcare professional. Photorealistic, detailed, supportive and empowering mood, serene colors."

# Function to generate image using FLUX.1-dev via Gradio Client
@st.cache_data(show_spinner="Generating a relevant image...") # Cache image generation for performance
def generate_flux_image(prompt):
    try:
        # Parameters as provided in your example
        result = flux_client.predict(
            prompt=prompt,
            seed=0,
            randomize_seed=True,
            width=768,
            height=768,
            guidance_scale=3.5,
            num_inference_steps=28,
            api_name="/infer"
        )

        if isinstance(result, (list, tuple)) and len(result) > 0:
            image_path = result[0]
        else:
            image_path = result

        if image_path and os.path.exists(image_path):
            return image_path
        else:
            st.error(f"FLUX.1-dev did not return a valid image path. Raw result: {result}")
            return None
    except Exception as e:
        st.error(f"Error generating image with FLUX.1-dev: {e}")
        return None

# Function to extract metrics from text description (simple parsing, can be enhanced with advanced NLP)
def extract_metrics_from_text(text_description):
    metrics = {
        "Pregnancies": 0, "Glucose": 100, "BloodPressure": 70, "SkinThickness": 20,
        "Insulin": 80, "BMI": 25.0, "DiabetesPedigreeFunction": 0.5, "Age": 30
    }
    # Using regex to find numbers near keywords
    # This is a basic example; a more robust solution would use a full NLP model
    # to understand context and extract entities accurately.

    pregnancies_match = re.search(r'(\d+)\s*(pregnancies|pregnant)', text_description, re.IGNORECASE)
    if pregnancies_match: metrics["Pregnancies"] = int(pregnancies_match.group(1))

    glucose_match = re.search(r'glucose\s+of\s+(\d+)|glucose\s+(\d+)', text_description, re.IGNORECASE)
    if glucose_match: metrics["Glucose"] = int(glucose_match.group(1) or glucose_match.group(2))

    bp_match = re.search(r'blood\s*pressure\s+of\s+(\d+)|blood\s*pressure\s+(\d+)', text_description, re.IGNORECASE)
    if bp_match: metrics["BloodPressure"] = int(bp_match.group(1) or bp_match.group(2))

    skin_match = re.search(r'skin\s*thickness\s+of\s+(\d+)|skin\s*thickness\s+(\d+)', text_description, re.IGNORECASE)
    if skin_match: metrics["SkinThickness"] = int(skin_match.group(1) or skin_match.group(2))

    insulin_match = re.search(r'insulin\s+level\s+of\s+(\d+)|insulin\s+(\d+)', text_description, re.IGNORECASE)
    if insulin_match: metrics["Insulin"] = int(insulin_match.group(1) or insulin_match.group(2))

    bmi_match = re.search(r'bmi\s+of\s+([\d.]+)|bmi\s+([\d.]+)', text_description, re.IGNORECASE)
    if bmi_match: metrics["BMI"] = float(bmi_match.group(1) or bmi_match.group(2))

    dpf_match = re.search(r'(diabetes\s+pedigree\s+function|dpf)\s+of\s+([\d.]+)|(diabetes\s+pedigree\s+function|dpf)\s+([\d.]+)', text_description, re.IGNORECASE)
    if dpf_match: metrics["DiabetesPedigreeFunction"] = float(dpf_match.group(2) or dpf_match.group(4))

    age_match = re.search(r'age\s+of\s+(\d+)|age\s+(\d+)', text_description, re.IGNORECASE)
    if age_match: metrics["Age"] = int(age_match.group(1) or age_match.group(2))

    return metrics

# Centralized prediction and result display function
def perform_prediction_and_display(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age):
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
            # Using 'gemini-pro' for text generation.
            model = genai.GenerativeModel('gemini-2.0-flash')
            prompt = f"Explain the following diabetes prediction in simple terms for a user. The input features were: Pregnancies={pregnancies}, Glucose={glucose}, BloodPressure={blood_pressure}, SkinThickness={skin_thickness}, Insulin={insulin}, BMI={bmi}, DiabetesPedigreeFunction={dpf}, Age={age}. The prediction is that the person is {prediction_text}. Briefly mention which of these factors might have contributed to the prediction."
            response = model.generate_content(prompt)
            explanation = response.text
            st.write(explanation)
        except Exception as e:
            st.error(f"Failed to generate explanation using Gemini ('gemini-pro'). Error: {e}. "
                     "Please ensure your API key is correct and 'gemini-pro' is available in your region.")

    with col_img:
        st.subheader('Related Image (FLUX.1-dev AI):')
        image_prompt = generate_image_prompt(prediction_text)
        image_path = generate_flux_image(image_prompt)

        if image_path:
            st.image(image_path, caption=image_prompt, use_column_width=True)
        else:
            st.warning("Could not generate a relevant image at this time using FLUX.1-dev.")


# --- Streamlit UI Design ---
st.set_page_config(layout="wide", page_title="AI-Powered Diabetes Predictor")

st.title('⚕️ AI-Powered Diabetes Predictor & Wellness Assistant')
st.markdown("""
Welcome to your personal health companion. You can either use our **AI Assistant** to describe your health concerns naturally, or input your **Health Metrics Manually** for a precise prediction.
""")

# Create tabs for different modes
tab1, tab2 = st.tabs(["🤖 AI Assistant", "✍️ Manual Health Metrics Input"])

with tab1:
    st.header("Tell Me About Your Health...")
    st.markdown("""
    Describe your current health status, any symptoms, or provide any known health metrics in a natural language.
    Our AI will try to understand and provide insights.
    """)
    user_description = st.text_area(
        "Describe your health here:",
        "I'm a 35-year-old female. My glucose level was 140, and BMI is 30. I have no history of pregnancies. My blood pressure is around 85.",
        height=150
    )

    if st.button('Get AI Insights', key='ai_predict_button'):
        st.info("Processing your description with Gemini AI...")
        try:
            # Use Gemini to extract or confirm metrics
            gemini_model = genai.GenerativeModel('gemini-2.0-flash') # Using gemini-pro for parsing
            prompt_for_parsing = f"""
            The user provided the following health description: "{user_description}"
            Please extract the following metrics if mentioned, otherwise suggest typical values or state if not found.
            Format your response as a JSON string with keys: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age.
            If a value is not explicitly mentioned, use a reasonable default. For DiabetesPedigreeFunction, if not mentioned, use 0.5.
            Example: {{"Pregnancies": 1, "Glucose": 120, "BloodPressure": 70, "SkinThickness": 20, "Insulin": 80, "BMI": 25.0, "DiabetesPedigreeFunction": 0.5, "Age": 30}}
            """
            response = gemini_model.generate_content(prompt_for_parsing)
            extracted_metrics_str = response.text.strip()

            # Clean up potential markdown formatting if Gemini adds it
            if extracted_metrics_str.startswith("```json") and extracted_metrics_str.endswith("```"):
                extracted_metrics_str = extracted_metrics_str[7:-3].strip()

            st.write("---")
            st.subheader("AI's Interpretation of Your Metrics:")
            try:
                extracted_metrics = eval(extracted_metrics_str) # Using eval for quick parsing of JSON-like string
                                                                # For production, consider `json.loads` after careful sanitization
                st.json(extracted_metrics) # Display the extracted metrics

                # Perform prediction with extracted metrics
                st.subheader("Running Diabetes Prediction based on AI-Parsed Data...")
                perform_prediction_and_display(
                    extracted_metrics.get("Pregnancies"),
                    extracted_metrics.get("Glucose"),
                    extracted_metrics.get("BloodPressure"),
                    extracted_metrics.get("SkinThickness"),
                    extracted_metrics.get("Insulin"),
                    extracted_metrics.get("BMI"),
                    extracted_metrics.get("DiabetesPedigreeFunction"),
                    extracted_metrics.get("Age")
                )
            except Exception as json_e:
                st.error(f"AI had difficulty parsing the metrics into a usable format: {json_e}")
                st.warning("Please try rephrasing your description or use the manual input tab.")
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