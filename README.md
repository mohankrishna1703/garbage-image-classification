📘 Garbage Image Classification

🔹 Objective

This project uses Deep Learning (CNN with MobileNetV2) to classify garbage images into different types such as cardboard, glass, metal, paper, plastic, and trash.
It is built step-by-step for beginners — each script does one small task.

garbage_classification_project/

│

├── data/                 

│   ├── cardboard/

│   ├── glass/

│   ├── metal/

│   ├── paper/

│   ├── plastic/

│   └── trash/

│

├── models/               # trained model + labels.txt

├── load_data.py          # checks dataset and counts

├── preprocess.py         # prepares data generators

├── labels_utils.py       # saves / loads label names

├── train_model.py        # trains MobileNetV2 model

├── evaluate_simple.py    # prints accuracy & metrics

├── app_simple.py         # Streamlit web app

└── requirements.txt      # dependencies

🧠 Step-by-Step to Run


1️⃣ Install dependencies

2️⃣ Check dataset

3️⃣ Train the model

4️⃣ Evaluate the model

5️⃣ Run the Streamlit App
