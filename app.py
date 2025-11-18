import os
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import google.generativeai as genai

os.environ["TF_USE_LEGACY_KERAS"] = "1"

# App layout
st.set_page_config(page_title="Plant Disease Advisor", page_icon="🌱", layout="wide")
st.title("🌱 Plant Disease Advisor")

# Subtle CSS to make the left column behave like sticky
st.markdown("""
<style>
.left-box {
    position: sticky;
    top: 80px;
}
</style>
""", unsafe_allow_html=True)

# API Key setup
API_KEY = None
try:
    API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    pass

if not API_KEY:
    API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.warning("Enter your Gemini API Key:")
    API_KEY = st.text_input("Gemini API Key:", type="password")

if API_KEY:
    try:
        genai.configure(api_key=API_KEY)
    except:
        st.error("Invalid API Key.")
        API_KEY = None

# Class labels
CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot', 'Pepper__bell___healthy',
    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
    'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite', 'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus', 'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# Load Gemini model
@st.cache_resource
def load_genai_model():
    try:
        return genai.GenerativeModel("gemini-2.5-flash")
    except:
        return None

genai_model = load_genai_model()

# Load trained MobileNetV2 model
MODEL_PATH = "models/Finetuned_Plant_Disease_Detector.keras"

@st.cache_resource
def load_detection_model(path):
    try:
        base = tf.keras.applications.MobileNetV2(
            weights=None,
            include_top=False,
            input_shape=(224, 224, 3)
        )

        model = tf.keras.models.Sequential([
            base,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(len(CLASS_NAMES), activation='softmax')
        ])

        model.load_weights(path)

        dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
        model.predict(dummy, verbose=0)

        return model

    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

model = load_detection_model(MODEL_PATH)

# Disease suggestions
SUGGESTION_DICT = {
    "Pepper__bell___Bacterial_spot": "Avoid overhead watering. Use copper sprays.",
    "Pepper__bell___healthy": "Healthy plant. Maintain care.",
    "Potato___Early_blight": "Use copper fungicide. Rotate crops.",
    "Potato___Late_blight": "Remove infected plants. Apply mancozeb.",
    "Potato___healthy": "Healthy plant. Maintain moisture.",
    "Tomato_Bacterial_spot": "Use copper sprays. Avoid splashing.",
    "Tomato_Early_blight": "Remove lower leaves. Apply fungicide.",
    "Tomato_Late_blight": "Remove infected leaves. Spray mancozeb.",
    "Tomato_Leaf_Mold": "Reduce humidity. Improve air flow.",
    "Tomato_Septoria_leaf_spot": "Remove spotted leaves. Use fungicide.",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Use neem oil/soap spray.",
    "Tomato__Target_Spot": "Improve ventilation. Apply fungicide.",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "No cure. Remove plant. Control whiteflies.",
    "Tomato__Tomato_mosaic_virus": "No cure. Remove plant. Sterilize tools.",
    "Tomato_healthy": "Healthy plant. Continue monitoring."
}

# Gemini AI treatment plan
def consult_ai_expert():
    if not genai_model:
        st.session_state.ai_response = "AI Expert unavailable."
        return

    with st.spinner("Preparing treatment plan..."):
        prompt = f"""
Give a short 6-bullet treatment plan for: {st.session_state.prediction}.
Rules:
- Only bullet points
- Max 1–2 lines per point
- No paragraphs
Include:
1. Immediate action
2. Organic method
3. Chemical method
4. Prevention tip
5. Do's
6. Don'ts
"""

        try:
            response = genai_model.generate_content(prompt)
            st.session_state.ai_response = response.text
        except Exception as e:
            st.session_state.ai_response = "Gemini Error: " + str(e)

# Initialize state
for key, val in {
    "prediction": None,
    "confidence": 0.0,
    "topk": [],
    "suggestion": "",
    "ai_response": ""
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# Image preprocessing
def preprocess_image(img):
    img = ImageOps.exif_transpose(img.convert("RGB"))
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0).astype(np.float32)

# Upload
upload = st.file_uploader("Upload a leaf image...", type=["jpg", "jpeg", "png"])

if upload:
    image_original = Image.open(upload)

    col1, col2 = st.columns([1, 2])

    # ---------------- Left Column (Fake Sticky) ----------------
    with col1:
        st.markdown("<div class='left-box'>", unsafe_allow_html=True)
        fixed_display = ImageOps.exif_transpose(image_original).resize((512, 512))
        st.image(fixed_display, caption="Uploaded Image (256×256)", width=512)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------------- Right Column ----------------
    with col2:
        if st.button("🔍 Classify Image"):
            st.session_state.ai_response = ""

            if model is None:
                st.error("Model not loaded.")
            else:
                with st.spinner("Analyzing..."):
                    processed = preprocess_image(image_original)
                    preds = model.predict(processed, verbose=0)[0]

                    idxs = preds.argsort()[-3:][::-1]
                    st.session_state.topk = [(CLASS_NAMES[i], float(preds[i])*100) for i in idxs]

                    best = idxs[0]
                    st.session_state.prediction = CLASS_NAMES[best]
                    st.session_state.confidence = preds[best]*100
                    st.session_state.suggestion = SUGGESTION_DICT.get(CLASS_NAMES[best], "No suggestion available.")

        if st.session_state.prediction:
            st.success(f"Prediction: {st.session_state.prediction}")
            st.info(f"Confidence: {st.session_state.confidence:.2f}%")

            st.write("### Top 3 Predictions")
            for name, conf in st.session_state.topk:
                st.write(f"- **{name}** — {conf:.2f}%")

            st.warning(f"### Quick Suggestion\n{st.session_state.suggestion}")
            st.divider()

            st.button(f"🤖 Get Treatment Plan", on_click=consult_ai_expert)

            if st.session_state.ai_response:
                st.write("### 🧠 AI Treatment Plan")
                st.markdown(st.session_state.ai_response)


