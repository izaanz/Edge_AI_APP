# Real-Time Age, Gender & Emotion Recognition on the Edge

This repository contains the source code and documentation for the "Project Edge AI" (DLBAIPEAI) course project. The project involves building and deploying deep learning models for real-time age, gender, and facial expression recognition in a cross-platform mobile application using Flutter and TensorFlow Lite.

## Project Overview

The primary goal of this project is to create a standalone mobile app that performs all AI inference directly on the user's device (on the edge), ensuring low latency and data privacy. The application can analyze a face from an image and predict:
- **Gender:** Male or Female
- **Age Group:** Child, Teen, Young Adult, Adult, or Senior
- **Emotion:** Angry, Happy, Sad, Neutral, etc. (8 classes)

## Repository Structure

- **/Flutter_App**: Contains the full source code for the Flutter mobile application.
- **/Model_Training**: Includes the Jupyter notebooks used for training the three separate deep learning models on the Kaggle platform.
- **README.md**: This file, providing an overview of the project.

## Models & Training

Three separate models were trained using transfer learning with the MobileNetV2 architecture in Python with TensorFlow/Keras.

1.  **Gender Model**: Trained on the UTKFace dataset. Achieved **91.0%** test accuracy.
2.  **Age Model**: Initially trained on the Adience dataset, then fine-tuned on a re-binned UTKFace dataset to improve accuracy from 22% to **68.0%**.
3.  **Emotion Model**: Trained on the FER+ dataset. Achieved **56.9%** test accuracy.

The training process is detailed in the Jupyter notebooks located in the `/Model_Training` directory.

## Edge Deployment & Mobile App

The trained Keras models were converted to the TensorFlow Lite (`.tflite`) format with post-training quantization to optimize for size and speed.

The mobile application was built with **Flutter** and uses the `tflite_flutter` package to run inference locally. The app allows users to select an image from their camera or gallery and displays the predictions in real-time.

### Screenshots

![App Screenshot 1](https://github.com/izaanz/Edge_AI_APP/blob/main/image/image2.png) 

## How to Run the Flutter App

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/izaanz/Edge-AI-Recognition-App.git
    ```
2.  **Navigate to the app directory:**
    ```bash
    cd Edge-AI-Recognition-App/Flutter_App
    ```
3.  **Install dependencies:**
    ```bash
    flutter pub get
    ```
4.  **Run the app:**
    Connect a physical device or start an emulator, then run:
    ```bash
    flutter run
    ```

## Tools and Technologies

-   **Model Training**: Python, TensorFlow, Keras, Scikit-learn, Jupyter Notebooks, Kaggle
-   **Mobile App**: Flutter, Dart, `tflite_flutter`, `image_picker`
-   **AI Assistance**: ChatGPT and Gemini were used to assist with debugging the Flutter application.
