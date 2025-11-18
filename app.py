import os
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import google.generativeai as genai

os.environ["TF_USE_LEGACY_KERAS"] = "1" 

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(page_title="Plant Disease Advisor", page_icon="🌱", layout="wide")
st.title("🌱 AI-Based Plant Disease Advisor")
st.write("Upload a plant leaf image to detect disease and get treatment suggestions.")


# SECURE API KEY HANDLING
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


# CLASS NAMES (Required for model definition)
CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy', 'Potato___Early_blight', 
    'Potato___Late_blight', 'Potato___healthy', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 
    'Tomato_Late_blight', 'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot', 
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus', 'Tomato_healthy'
]


# LOAD GEMINI MODEL

@st.cache_resource
def load_genai_model():
    if API_KEY:
        try:
            return genai.GenerativeModel("gemini-2.5-flash")
        except Exception as e:
            st.error(f"Error loading Gemini model: {e}")
            return None
    return None

genai_model = load_genai_model()

# LOAD KERAS DETECTION MODEL
MODEL_PATH = "models/Finetuned_Plant_Disease_Detector.keras"

@st.cache_resource
def load_detection_model(path):
    try:
        # 1. Base Model Definition (MobileNetV2, weights=None to load from file)
        base_model = tf.keras.applications.MobileNetV2(
            weights=None, 
            include_top=False, 
            input_shape=(224, 224, 3)
        )
        # 2. Reconstruct the Sequential Model Head exactly as trained
        local_model = tf.keras.models.Sequential([
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')
        ])

        # 3. Load the Weights into the Reconstructed Architecture
        local_model.load_weights(path)
        
        # Pre-run a prediction to finalize the graph compilation 
        dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
        local_model.predict(dummy_input, verbose=0)
        
        st.success("✅ Detection model successfully loaded.")
        return local_model

    except Exception as e:
        st.error(f"Failed to load detection model: {e}")
        st.error("Please verify the model file path and the required dependencies.")
        return None
    
model = load_detection_model(MODEL_PATH)


# SUGGESTIONS
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

# MAIN LOGIC
if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.confidence = 0.0
    st.session_state.topk = []
    st.session_state.suggestion = ""

# Preprocessing the image
def preprocess_image(image_pil):
    image = ImageOps.exif_transpose(image_pil.convert("RGB"))
    image = image.resize((224, 224)) # Model input size
    arr = np.array(image) / 255.0
    return np.expand_dims(arr, axis=0).astype(np.float32)
upload = st.file_uploader("Upload a leaf image...", type=["jpg", "jpeg", "png"])

if upload:
    image_original = Image.open(upload)
    image_display = ImageOps.exif_transpose(image_original.convert("RGB")).resize((224, 224))
    
    col1, col2 = st.columns(2)

    with col1:
        st.image(image_display, caption="Uploaded Image (224x224)", use_container_width=True)

        if st.button("🔍 Classify Image"):
            if model is None:
                st.error("Model not loaded. Check path.")
            else:
                with st.spinner("Analyzing..."):
                    # Use the original image for processing (which resizes to 224x224)
                    processed = preprocess_image(image_original)
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

            if st.button(f"🤖 Ask AI Expert about {st.session_state.prediction}", key="ai_expert_button"):
                if not genai_model:
                        st.error("AI Expert unavailable — check API key.")
                else:
                        with st.spinner("Consulting AI Expert..."):
                            prompt = f"Provide a simple, easy-to-understand 6-point treatment plan for {st.session_state.prediction}. The points should be structured as clear bullet points covering Immediate action, Organic methods, Chemical options, and Prevention steps."
                            try:
                                response = genai_model.generate_content(prompt)
                                st.info(response.text)
                            except Exception as e:
                                st.error(f"Gemini Error: {e}")
