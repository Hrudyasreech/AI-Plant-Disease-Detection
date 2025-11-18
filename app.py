import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import google.generativeai as genai
import os
import io

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Plant Disease Advisor", page_icon="🌱", layout="wide")
st.title("🌱 AI-Based Plant Disease Advisor")
st.write("Upload a plant leaf image to detect disease and get treatment suggestions.")

# ---------------------------------------------------------
# SECURE API KEY HANDLING (Local + Streamlit Cloud)
# ---------------------------------------------------------
API_KEY = None

# 1. Try Streamlit Cloud secrets
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    pass

# 2. Try system environment variables (local machine)
if not API_KEY:
    API_KEY = os.getenv("GEMINI_API_KEY")

# 3. Ask user manually if still not available
if not API_KEY:
    st.warning("Google Gemini API Key not found. Enter it below:")
    API_KEY = st.text_input("Enter your Google Gemini API Key:", type="password")

# Configure Gemini API
if API_KEY:
    try:
        genai.configure(api_key=API_KEY)
    except Exception as e:
        st.error(f"Invalid API Key: {e}")
        API_KEY = None

# ---------------------------------------------------------
# LOAD GENERATIVE AI MODEL (CACHED)
# ---------------------------------------------------------
@st.cache_resource
def load_genai_model():
    if API_KEY:
        try:
            return genai.GenerativeModel("models/gemini-2.5-flash")
        except Exception as e:
            st.error(f"Error loading Gemini model: {e}")
            return None
    return None

genai_model = load_genai_model()

# ---------------------------------------------------------
# LOAD PLANT DISEASE MODEL (CACHED)
# ---------------------------------------------------------
MODEL_PATH = "models/Finetuned_Plant_Disease_Detector.h5"  # adjust path

@st.cache_resource
def load_detection_model(path):
    try:
        return tf.keras.models.load_model(path)
    except Exception as e:
        st.error(f"Failed to load detection model: {e}")
        return None

model = load_detection_model(MODEL_PATH)

# ---------------------------------------------------------
# CLASS NAMES (same order as training)
# ---------------------------------------------------------
CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 
    'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 
    'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
]

# ---------------------------------------------------------
# SUGGESTIONS DATABASE
# ---------------------------------------------------------
SUGGESTION_DICT = {
    "Pepper__bell___Bacterial_spot": "Avoid overhead watering. Use copper-based fungicides. Remove infected leaves.",
    "Pepper__bell___healthy": "Plant is healthy! Maintain regular care.",
    "Potato___Early_blight": "Use copper fungicides. Water at base. Rotate crops.",
    "Potato___Late_blight": "Remove infected plants immediately. Apply mancozeb or chlorothalonil.",
    "Potato___healthy": "Plant is healthy! Maintain soil moisture.",
    "Tomato_Bacterial_spot": "Avoid overhead watering. Use copper sprays.",
    "Tomato_Early_blight": "Prune lower leaves. Apply chlorothalonil or copper.",
    "Tomato_Late_blight": "Destroy infected plants. Apply mancozeb proactively.",
    "Tomato_Leaf_Mold": "Reduce humidity. Increase ventilation.",
    "Tomato_Septoria_leaf_spot": "Remove spotted leaves. Use fungicides.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Spray with neem or insecticidal soap.",
    "Tomato__Target_Spot": "Improve air circulation. Apply fungicide.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "VIRUS—No cure. Remove plant. Control whiteflies.",
    "Tomato__Tomato_mosaic_virus": "VIRUS—No cure. Sterilize tools. Remove plants.",
    "Tomato_healthy": "Plant is healthy! Continue monitoring."
}

# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.confidence = 0.0
    st.session_state.topk = []
    st.session_state.suggestion = ""

# ---------------------------------------------------------
# IMAGE PREPROCESSING
# ---------------------------------------------------------
def preprocess_image(image_pil):
    image = ImageOps.exif_transpose(image_pil.convert("RGB"))
    image = image.resize((224, 224))
    arr = np.array(image) / 255.0
    return np.expand_dims(arr, axis=0).astype(np.float32)

# ---------------------------------------------------------
# MAIN APP LOGIC
# ---------------------------------------------------------
upload = st.file_uploader("Upload a leaf image...", type=["jpg", "jpeg", "png"])

if upload:
    image = Image.open(upload)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=False, width = "content")

        if st.button("🔍 Classify Image"):
            with st.spinner("Analyzing..."):
                processed = preprocess_image(image)
                preds = model.predict(processed, verbose=0)[0]

                # Top-3 predictions
                idxs = preds.argsort()[-3:][::-1]
                st.session_state.topk = [(CLASS_NAMES[i], float(preds[i]) * 100) for i in idxs]

                best = idxs[0]
                st.session_state.prediction = CLASS_NAMES[best]
                st.session_state.confidence = preds[best] * 100
                st.session_state.suggestion = SUGGESTION_DICT.get(CLASS_NAMES[best], "No suggestion available.")

    with col2:
        if st.session_state.prediction:
            st.success(f"**Prediction:** {st.session_state.prediction}")
            st.info(f"**Confidence:** {st.session_state.confidence:.2f}%")

            st.write("### 🔎 Top Predictions")
            for name, conf in st.session_state.topk:
                st.write(f"- {name} — {conf:.2f}%")

            st.warning(f"### 🌿 Recommendation:\n{st.session_state.suggestion}")

            st.divider()

        if st.button(f"🤖 Ask AI Expert about {st.session_state.prediction}"):
            if not genai_model:
                    st.error("AI Expert unavailable — check API key.")
            else:
                    with st.spinner("Consulting AI Expert..."):
                        prompt = f"""
                          Give a short bullet-point treatment guide for {st.session_state.prediction}.
                          Rules:
                           - Only bullet points
                           - Max 6 bullets
                           - Each bullet under 8 words
                           - Include: immediate fix, organic remedy, chemical remedy, prevention
                         """
                        try:
                            response = genai_model.generate_content(prompt)
                            st.info(response.text)
                        except Exception as e:
                            st.error(f"Gemini Error: {e}")


