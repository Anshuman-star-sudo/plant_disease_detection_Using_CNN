import io
import json
from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# 1. Handle dynamic paths
working_dir = Path(__file__).resolve().parent
model_path = working_dir / "plant_disease_prediction_model.keras"

# Custom GlorotUniform subclass to ignore input_axes/output_axes serialization bugs in older Keras versions
class SafeGlorotUniform(tf.keras.initializers.GlorotUniform):
    def __init__(self, seed=None, **kwargs):
        kwargs.pop('input_axes', None)
        kwargs.pop('output_axes', None)
        super().__init__(seed=seed, **kwargs)

# 2. Optimized model caching
def load_model():
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    custom_objects = {
        'GlorotUniform': SafeGlorotUniform
    }
    return tf.keras.models.load_model(model_path, custom_objects=custom_objects)

model = load_model()

# Load class labels index mapping
with open(working_dir / "class_indices.json", encoding="utf-8") as f:
    class_indices = json.load(f)

def load_and_preprocess_image(pil_image, target_size=(128, 128)):
    """Resizes and converts a PIL Image to a batch-expanded numpy array.
    
    Leaves values unscaled as the model contains an internal Rescaling layer.
    """
    img = pil_image.resize(target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Shape becomes (1, 128, 128, 3)
    return img_array

def predict_image_class(model, pil_image, class_indices):
    preprocessed_img = load_and_preprocess_image(pil_image)
    predictions = model(preprocessed_img, training=False)
    predicted_class_index = np.argmax(predictions.numpy(), axis=1)[0]
    
    # Map index to class label safely
    str_idx = str(int(predicted_class_index))
    predicted_class_name = class_indices.get(str_idx, f"Unknown Class ({str_idx})")
    
    return predicted_class_name

# 3. Streamlit UI Config
st.set_page_config(page_title="Plant Disease Classifier", layout="wide")
st.title("Plant Disease Classifier")

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Load raw file stream to PIL Image
    pil_image = Image.open(io.BytesIO(uploaded_image.getvalue())).convert("RGB")
    
    col1, col2 = st.columns(2)

    with col1:
        st.image(pil_image, caption="Uploaded image", use_container_width=True)

    with col2:
        if st.button("Classify"):
            prediction = predict_image_class(model, pil_image, class_indices)
            st.success(f"Prediction: {prediction}")
