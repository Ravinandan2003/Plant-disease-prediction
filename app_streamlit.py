import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import streamlit as st

# Define class labels
labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

# Initialize model variable
model = None

# Function to load .h5 model
def load_h5_model():
    return load_model('model.h5')

# Function to load .tflite model
def load_tflite_model():
    interpreter = tf.lite.Interpreter(model_path="model_quantized.tflite")
    interpreter.allocate_tensors()
    return interpreter

# Load the selected model
def load_selected_model(model_type):
    if model_type == 'H5':
        return load_h5_model()
    elif model_type == 'TFLite':
        return load_tflite_model()

# Predict function for .h5 model
def get_result_h5(image):
    img = image.resize((225, 225))  # Resize the image
    x = img_to_array(img)
    x = x.astype('float32') / 255.0
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    predictions = model.predict(x)[0]
    return labels[np.argmax(predictions)]  # Get predicted label

# Predict function for .tflite model
def get_result_tflite(image, interpreter):
    img = image.resize((225, 225))  # Resize the image
    x = img_to_array(img)
    x = x.astype('float32') / 255.0
    x = np.expand_dims(x, axis=0)  # Add batch dimension

    # Get tensor details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the tensor input
    interpreter.set_tensor(input_details[0]['index'], x)

    # Run inference
    interpreter.invoke()

    # Get the prediction output
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    return labels[np.argmax(predictions)]  # Get predicted label

# Streamlit UI
st.title("Plant Disease Classification")
st.write("Upload an image to classify the disease.")

# Select model type: H5 or TFLite
model_type = st.selectbox("Choose Model Type", ["H5", "TFLite"])

# Load the selected model
if model_type == 'H5':
    model = load_selected_model('H5')
elif model_type == 'TFLite':
    model = load_selected_model('TFLite')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Open and display image
    image = load_img(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Get prediction based on selected model
    if model_type == 'H5':
        prediction = get_result_h5(image)
    elif model_type == 'TFLite':
        prediction = get_result_tflite(image, model)

    st.write("### Prediction: ", prediction)
