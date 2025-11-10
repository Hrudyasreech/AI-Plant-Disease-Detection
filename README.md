# AI-Plant-Disease-Detection
AI-Based Smart Plant Health Advisor for Sustainable Agriculture | CNN using MobileNetV2

Problem Statement :
Crop diseases are a major cause of reduced agricultural productivity worldwide. Farmers often lack quick, reliable, and affordable tools for early detection of plant diseases, which leads to delayed treatment and lower yields. Manual inspection is time-consuming and inaccurate, especially in large farms.

To address this, an AI-powered system is proposed to automatically identify plant diseases from leaf images and provide suitable treatment suggestions — contributing to sustainable farming practices.

Proposed Solution :
The proposed system uses a Convolutional Neural Network (CNN) architecture, specifically MobileNetV2, to classify leaf images into healthy or diseased categories.
After detection, the system can also display the possible cause and recommended treatment (based on the disease class).
Later stages may include:
1.Real-time detection using OpenCV and a camera feed.
2.A dashboard interface for farmers to upload images and get instant feedback.

Dataset Used:
Name: PlantVillage Dataset
Source: Kaggle (Link : https://www.kaggle.com/datasets/emmarex/plantdisease )

About the Dataset:
1.Contains over 50,000 labeled images of healthy and diseased plant leaves.
2.Covers 14 crops and 38 different classes of diseases.
3.Each image is a close-up of a single leaf in .jpg format.
4.Suitable for CNN-based image classification and transfer learning tasks.

