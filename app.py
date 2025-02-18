import os
import numpy as np
import streamlit as st
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model('model.h5')
print('Model loaded.')

# Define class labels
labels = {0: 'Healthy', 1: 'Powdery', 2: 'Rust'}

# Folder to hold uploaded images
UPLOAD_FOLDER = 'uploads'

# Check if the folder exists, if not create it
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to process and predict image
def getResult(image):
    img = image.resize((225, 225))  # Resize to match model input size
    x = np.array(img)
    x = x.astype('float32') / 255.  # Normalize the image
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    predictions = model.predict(x)[0]
    return predictions

# Streamlit App Interface
st.title("Plant Disease Classification Using Deep Learning")
st.write("Select an image from the uploads folder to classify the disease.")

# Get list of image files in the uploads folder
image_files = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith(('.png', '.jpg', '.jpeg'))]

# Select image from the folder
selected_image = st.selectbox("Choose an image", image_files)

if selected_image:
    # Load and display the selected image
    img_path = os.path.join(UPLOAD_FOLDER, selected_image)
    img = Image.open(img_path)

    # Show the selected image
    st.image(img, caption="Uploaded Image", use_column_width=True)
    st.write("")

    # Get predictions
    predictions = getResult(img)
    predicted_label = labels[np.argmax(predictions)]

    # Display the predicted label
    st.write(f"**Prediction**: {predicted_label}")
    st.write(f"Prediction Probabilities: {predictions}")
