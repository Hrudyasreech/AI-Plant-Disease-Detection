import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import google.generativeai as genai
import os

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Plant Disease Advisor", page_icon="🌱", layout="wide")
st.title("🌱 AI-Based Plant Disease Advisor")
st.write("Upload a plant leaf image to detect disease and get treatment suggestions.")

# ---------------------------------------------------------
# SECURE API KEY HANDLING
# ---------------------------------------------------------
API_KEY = None

try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    pass

if not API_KEY:
    API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.warning("Google Gemini API Key not found. Enter it below:")
    API_KEY = st.text_input("Enter your Google Gemini API Key:", type="password")

if API_KEY:
    try:
        genai.configure(api_key=API_KEY)
    except Exception as e:
        st.error(f"Invalid API Key: {e}")
        API_KEY = None

# ---------------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------------
@st.cache_resource
def load_genai_model():
    if API_KEY:
        try:
            # Use the stable 1.5 model (2.5 is not yet standard)
            return genai.GenerativeModel("gemini-1.5-flash")
        except Exception as e:
            st.error(f"Error loading Gemini model: {e}")
            return None
    return None

genai_model = load_genai_model()

MODEL_PATH = "models/Finetuned_Plant_Disease_Detector.keras"

@st.cache_resource
def load_detection_model(path):
    try:
        # Copy and paste this ENTIRE function over your old one
@st.cache_resource
def load_detection_model(path):
    try:
        # 1. Build the Empty Architecture (MobileNetV2 + Your Layers)
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet', 
            include_top=False, 
            input_shape=(224, 224, 3)
        )
        
        local_model = tf.keras.models.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(15, activation='softmax') # 15 Classes
        ])

        # 2. Load the Weights into the Architecture
        local_model.load_weights(path)
        print("✅ Model weights loaded successfully!")
        return local_model

    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
    except Exception as e:
        st.error(f"Failed to load detection model: {e}")
        return None

model = load_detection_model(MODEL_PATH)

# ---------------------------------------------------------
# CLASS NAMES
# ---------------------------------------------------------
CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 
    'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 
    'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
]

# ---------------------------------------------------------
# SUGGESTIONS
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
# MAIN LOGIC
# ---------------------------------------------------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.confidence = 0.0
    st.session_state.topk = []
    st.session_state.suggestion = ""

def preprocess_image(image_pil):
    image = ImageOps.exif_transpose(image_pil.convert("RGB"))
    image = image.resize((224, 224))
    arr = np.array(image) / 255.0
    return np.expand_dims(arr, axis=0).astype(np.float32)

upload = st.file_uploader("Upload a leaf image...", type=["jpg", "jpeg", "png"])

if upload:
    image = Image.open(upload)
    col1, col2 = st.columns(2)

    with col1:
        # Fixed the width parameter here
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("🔍 Classify Image"):
            if model is None:
                st.error("Model not loaded. Check path.")
            else:
                with st.spinner("Analyzing..."):
                    processed = preprocess_image(image)
                    preds = model.predict(processed, verbose=0)[0]
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
                        prompt = f"Give a short 6-bullet treatment guide for {st.session_state.prediction} (Immediate/Organic/Chemical/Prevention)."
                        try:
                            response = genai_model.generate_content(prompt)
                            st.info(response.text)
                        except Exception as e:
                            st.error(f"Gemini Error: {e}")



