import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import streamlit as st

# Load the trained model
model = load_model('model.h5')

# Define class labels
labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

def getResult(image):
    img = image.resize((225, 225))  # Resize the image
    x = img_to_array(img)
    x = x.astype('float32') / 255.0
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    predictions = model.predict(x)[0]
    return labels[np.argmax(predictions)]  # Get predicted label

# Streamlit UI
st.title("Plant Disease Classification")
st.write("Upload an image to classify the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open and display image
    image = load_img(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Get prediction
    prediction = getResult(image)
    st.write("### Prediction: ", prediction)
