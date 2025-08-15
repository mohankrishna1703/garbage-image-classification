🗑 Garbage Image Classification Using Deep Learning
📌 Overview

This project classifies garbage images into 6 categories:

Cardboard

Glass

Metal

Paper

Plastic

Trash

It uses a Convolutional Neural Network (CNN) to recognize patterns in images and predict the correct category.

This can help in automated waste sorting, making recycling faster and more efficient.

📂 Dataset Structure

The dataset is stored inside the data/ folder.
data/
├── cardboard/
├── glass/
├── metal/
├── paper/
├── plastic/
└── trash/
Each folder contains sample images for that category.

🛠 Requirements

Install Python dependencies using:
**pip install -r requirements.txt**

Main libraries used:

tensorflow-cpu — to build and train the CNN model

streamlit — to make a simple web app for predictions

matplotlib — for visualizing training results

scikit-learn — for splitting data and evaluation metrics

pillow — for image handling

🏋️ Training the Model

Run this command to train the model:
**python train_cnn.py**

This will:

Load the dataset from data/

Preprocess and resize the images

Train a CNN model

Save the trained model in the models/ folder

🚀 Running the App

After training, you can test your model using the Streamlit app:
**streamlit run app.py**

Steps:

Open the browser link shown in the terminal

Upload a garbage image (jpg, png)

See the predicted category and confidence score

📊 Example Prediction

Input: Image of a crushed soda can

Model Prediction: Metal

🧠 Model Architecture

The CNN model consists of:

Conv2D Layers — Detect image features like edges and patterns

MaxPooling Layers — Reduce image size to make training faster

Flatten Layer — Convert 2D data to 1D

Dense Layers — Fully connected layers for prediction

Softmax Activation — Gives probability for each category

Loss Function: Categorical Crossentropy
Optimizer: Adam

🔍 Project Flow
**Image → Preprocessing → CNN Model → Prediction → Output Label**