Model: CNN using MobileNetV2

**Week 2 Activities **:

1. Set up Google Colab environment with Kaggle API and Google Drive integration for dataset storage and model saving.
2. Downloaded and preprocessed the PlantVillage dataset using ImageDataGenerator for real-time data augmentation (rotation, flipping, scaling, etc.).
3. Visualized sample and augmented images to verify preprocessing correctness.
4. Implemented the CNN model using MobileNetV2 architecture (transfer learning approach with frozen base layers).
5. Trained the model on the dataset for 10 epochs using TensorFlow and Keras.
6. Monitored training progress and fixed runtime issues such as “input ran out of data” and GPU timeout errors.
7. Plotted graphs of training vs validation accuracy and loss to evaluate model performance.
8. Saved trained model (Plant_Disease_Detector_MobileNetV2.h5) to Google Drive for further testing and deployment.

**Learning Outcomes** :
1. Learned how to integrate Kaggle datasets into Colab using API keys and handle large datasets efficiently.
2. Understood the importance of data augmentation to prevent overfitting and improve model generalization.
3. Gained hands-on experience in transfer learning and how to use pretrained models like MobileNetV2 effectively.
4. Learned how to handle Colab GPU optimization and ensure long-running training sessions continue without interruption.
5. Understood model evaluation metrics and how to visualize performance trends using accuracy/loss curves.
