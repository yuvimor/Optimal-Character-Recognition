# Import necessary libraries
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model

# Load the trained Quantum Neural Network (QNN) model
model = load_model("quantum_neural_network_model.h5")

# Streamlit app
st.title("Optimal Character Recognition")
st.markdown("Upload an image of the character")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
submit = st.button('Recognize')

#On recognize button click
if submit:
    if uploaded_file is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        #Resize
        image = cv2.resize(image, (64, 64))
        # Convert to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
        # Normalize pixel values
        image = image / 255.0  
        #Convert image to 4 Dimension
        image = np.reshape(image, (1, 64, 64, 1))
        
        # Make predictions using the loaded model
        prediction = model.predict(image)

    # Display the prediction
    st.write("Prediction:", label_encoder.classes_[np.argmax(prediction)])

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)
