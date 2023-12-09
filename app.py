# Import necessary libraries
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import cv2

# Load the trained Quantum Neural Network (QNN) model
loaded_model = load_model("quantum_neural_network_model.h5")

# Function to preprocess an image for prediction
def preprocess_image(img):
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    img = img / 255.0  # Normalize pixel values
    img = np.reshape(img, (1, 64, 64, 1))  # Reshape for model input
    return img

# Streamlit app
st.title("Optimal Character Recognition Streamlit App")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Preprocess the image
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    processed_image = preprocess_image(image)
    
    # Make predictions using the loaded model
    prediction = loaded_model.predict(processed_image)

    # Display the prediction
    st.write("Prediction:", label_encoder.classes_[np.argmax(prediction)])

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
