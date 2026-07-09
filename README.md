# 🌿 Plant Disease Prediction Using CNN

A high-performance, deep learning-based web application engineered to diagnose crop health and classify plant leaf diseases from digital imagery. Built on top of a custom Convolutional Neural Network (CNN) leveraging **Keras 3**, the system acts as an accessible, localized precision agriculture tool. The application is production-optimized and fully deployed via Streamlit Community Cloud.

## 🚀 Live Application
Experience the real-time web deployment here:  
👉 **[Live App Link]**[deployed app link](https://plantdiseasedetectionusingcnn-vmvjy2ts8rtp82rvaj9rfc.streamlit.app/)

---


## 📊 Model Architecture & Performance
The custom CNN architecture uses sequential convolutional blocks with spatial feature extractors, downsampling max-pooling matrices, and dropout regularization layers to protect against overfitting.

The model was validated against a rigorous split of diverse, high-resolution samples, achieving highly resilient generalization boundaries:

* **Training Accuracy:** `98.73%` ($0.9873$)
* **Training Loss:** `0.7284`
* **Validation Accuracy:** `87.35%` 🚀

---

## 📁 Dataset & Taxonomy Reference
This project utilizes the **New Plant Diseases Dataset**, an open-source image repository hosted on Kaggle containing over 87,000 high-resolution images of crop leaves mapping across 38 distinct class indices.

The target classes capture critical crop groups including:
* **Solanaceae:** Tomato (Early Blight, Late Blight, Yellow Leaf Curl Virus, Mosaic Virus, Healthy, etc.) and Potato.
* **Rosaceae:** Apple, Strawberry, Peach, and Cherry variants.
* **Other Essential Crops:** Grape, Corn, Pepper, and Orange pathogens.

* **Dataset Source:** [New Plant Diseases Dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

---

## 🛠️ Repository Topography
```text
├── app/
│   ├── main.py                             # Core Streamlit application & prediction layout
│   ├── class_indices.json                  # Strict index-to-disease categorical string dictionary
│   ├── plant_disease_prediction_model.keras # Native Keras 3 saved weights archive
│   └── requirements.txt                    # Pinned lightweight cloud-native dependencies
├── notebook/
│   └── plant_disease_prediction_using_CNN.ipynb # Complete experimental notebook (EDA, training, validation)
└── .gitignore                              # Filters out local virtual environments (.venv) and bytecode cache
