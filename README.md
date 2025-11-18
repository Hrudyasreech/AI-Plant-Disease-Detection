# AI Plant Disease Detection
### _Deep Learning-based Smart Plant Health Advisor using MobileNetV2_

---

## 1. Problem Statement
Crop diseases significantly reduce agricultural productivity, especially in areas lacking access to early detection tools. Manual inspection is slow, subjective, and often inaccurate.

This project proposes an **AI-powered system** that detects plant diseases from leaf images and provides treatment recommendations — supporting **sustainable, scalable agriculture**.

---

## 2. Proposed Solution
A **Convolutional Neural Network (CNN)** based on **MobileNetV2 (Transfer Learning)** is used to classify plant leaf images into healthy/diseased categories.

**Future extensions include:**
- Real-time detection using OpenCV  
- Farmer dashboard for uploading images  
- Mobile app integration

---

## 3. Dataset
**PlantVillage Dataset**  
Source: Kaggle  
Link: https://www.kaggle.com/datasets/emmarex/plantdisease

**Dataset Highlights:**
- 50,000+ images  
- 14 crops, 38 disease classes  
- High-quality leaf images  
- Well-suited for CNN & transfer learning

---

## 4. Key Features
- **Accurate Classification:** MobileNetV2 with fine-tuning  
- **Multi-class Support:** 15+ disease categories (Tomato, Potato, Pepper, etc.)  
- **Expert Treatment Advice:** (Optional) Gemini integration  
- **Clean, Simple Streamlit UI**  
- **Lightweight Model:** Suitable for mobile/edge deployment

---

## 5. Technical Results
| Metric | Score |
|--------|-------:|
| Training Accuracy | ~89–92% |
| Validation Accuracy | ~88–90% |
| Loss | ~0.28–0.32 |

> These results may vary depending on training epochs, augmentation, and fine-tuning depth.

---

## 6. Model Architecture
- **Base Model:** MobileNetV2 (pre-trained on ImageNet)  
- **include_top:** False  
- **Input Shape:** (224, 224, 3)  
- **Custom Layers:** GlobalAveragePooling2D → Dense (ReLU) → Dropout → Dense (Softmax)

---

## 7. How to Run Locally
**Step 1 — Clone the repository**
```bash
git clone <your-repo-link>
cd AI-Plant-Disease-Detection
```

**Step 2 — Install Dependencies**
```bash
pip install -r requirements.txt
```

**Step 3 — Run Streamlit App**
```bash
streamlit run app.py
```

If using Gemini API for treatment suggestions, enter your API key in the sidebar.

---

## 8. Live Demo
(Replace this with your deployed Streamlit link)

👉 **[https://ai-plant-disease-detection-hs.streamlit.app/](#)**

---

## 9. Project Structure
```
AI-Plant-Disease-Detection/
├── app.py                      # Streamlit Web App
├── PDD_Final.ipynb             # Full model training notebook
├── models/                     # Saved .h5 models
├── requirements.txt            # Dependencies
└── README.md                   # Project documentation
```

---

## 10. Future Improvements
- Optimize model for mobile devices  
- Deploy on AWS/GCP  
- Add Grad-CAM for interpretability  
- Expand to more crop species  
- Build multilingual farmer interface


