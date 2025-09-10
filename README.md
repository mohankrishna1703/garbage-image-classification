Garbage Image Classification Using Deep Learning
📌 Introduction

This project is about classifying garbage images into six categories using Deep Learning.
The categories are:

1. Cardboard

2. Glass

3. Metal

4. Paper

5. Plastic

6. Trash

The main idea is to use Artificial Intelligence (AI) to help in waste management.
If machines can identify the type of garbage, it will help in automatic waste sorting, recycling, and keeping the environment clean.

🎯 Objective

The goal of this project is:

- To build a Convolutional Neural Network (CNN) model that can classify garbage images.

- To train the model on the given dataset.

- To create a simple web application using Streamlit where users can upload an image and get the predicted category.


📂 Project Structure

garbage_classification_project/
│
├── data/           # Dataset (images in 6 folders: cardboard, glass, etc.)
├── models/         # Saved trained model (garbage_cnn.h5)
├── train.py        # Training code
└── app.py          # Streamlit web app
│
├── requirements.txt # Dependencies
└── README.md        # Project report


📊 How It Works

1. Images are resized to 128x128 pixels.

2. Model uses a CNN (Convolutional Neural Network) with:

      - Convolution Layers

      - Pooling Layers

      - Dense Layers

3. Output layer predicts 1 out of 6 categories.


✅ Results

1. The model can classify most images from the dataset correctly.

2. Works best on clear garbage images.

3. On unrelated images (like human/animal pictures), it may still predict one of the garbage 
categories → because it only knows 6 classes.


⚠️ Limitations

1. Model is trained only on 6 categories → cannot say “Unknown”.

2. Accuracy depends on dataset quality.

3. Doesn’t perform well on real-world, messy, or blurry images.
