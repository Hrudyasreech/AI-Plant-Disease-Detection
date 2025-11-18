# AI-Plant-Disease-Detection
AI-Based Smart Plant Health Advisor for Sustainable Agriculture | CNN using MobileNetV2

Problem Statement :
Crop diseases are a major cause of reduced agricultural productivity worldwide. Farmers often lack quick, reliable, and affordable tools for early detection of plant diseases, which leads to delayed treatment and lower yields. Manual inspection is time-consuming and inaccurate, especially in large farms.

To address this, an AI-powered system is proposed to automatically identify plant diseases from leaf images and provide suitable treatment suggestions — contributing to sustainable farming practices.

Proposed Solution :
The proposed system uses a Convolutional Neural Network (CNN) architecture, specifically MobileNetV2, to classify leaf images into healthy or diseased categories.
After detection, the system can also display the possible cause and recommended treatment (based on the disease class).
Later stages may include:
1. Real-time detection using OpenCV and a camera feed.
2. A dashboard interface for farmers to upload images and get instant feedback.

Dataset Used:
Name: PlantVillage Dataset
Source: Kaggle (Link : https://www.kaggle.com/datasets/emmarex/plantdisease )

About the Dataset:
1. Contains over 50,000 labeled images of healthy and diseased plant leaves.
2. Covers 14 crops and 38 different classes of diseases.
3. Each image is a close-up of a single leaf in .jpg format.
4. Suitable for CNN-based image classification and transfer learning tasks.

**LIVE DEMO**
Click here to try the app instantly: [PASTE YOUR STREAMLIT APP LINK HERE]
(Note: If you run the app locally, you will need to provide your own Gemini API Key. The live demo has this configured.)

**Key Features**:
1. Accurate Detection: Uses a MobileNetV2 CNN model trained on the PlantVillage dataset (54,000+ images).
2. 15 Disease Classes: Can identify diseases in Potato, Tomato, and Pepper plants.
3. Expert Advice: Integrated with Google Gemini AI to provide detailed, step-by-step treatment plans (Organic & Chemical).
4. User-Friendly Interface: A simple, responsive web app built with Streamlit.

**Technical Details**: 
1. Model Architecture: MobileNetV2 (Transfer Learning) + Fine-Tuning.
2. Training Accuracy: ~89% - 92%.
3. Tech Stack: TensorFlow/Keras, Streamlit, Google Gemini API, Python.

How to Run Locally ?

If you want to run the code on your own machine instead of using the Live Demo:
1. Clone or Download this repository.
2. Install Dependencies:
   -> pip install -r requirements.txt
3. Run the App:
   -> streamlit run app.py
4. Enter API Key: When the app opens, enter your Google Gemini API Key in the sidebar to enable the AI advice feature.

Project Structure:
1. app.py: The main source code for the Streamlit web application.
2. Plant_Disease_Project_Report.ipynb: A complete report notebook showing the model training process, evaluation metrics, and confusion matrices.
3. models/: Contains the trained .h5 model files.
4. requirements.txt: List of Python libraries required.

