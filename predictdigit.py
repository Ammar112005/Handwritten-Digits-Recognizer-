 import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("mnist_cnn_model.h5")
    return model

model = load_model()

# Streamlit page setup
st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("Handwritten Digit Recognizer")
st.markdown("Upload a digit image (28x28 pixels, white digit on black background).")

# Upload and process the image
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
        image = ImageOps.invert(image)                  # Invert colors
        image = image.resize((28, 28))                  # Resize to 28x28

        # Normalize and reshape
        img_array = np.array(image).astype("float32") / 255.0
        img_array = img_array.reshape(1, 28, 28, 1)

        # Predict using the model
        prediction = model.predict(img_array, verbose=0)
        predicted_digit = np.argmax(prediction)

        # Show the processed image and prediction
        st.image(image, caption="Processed Image (28x28)", width=150)
        st.subheader(f"Predicted Digit: {predicted_digit}")
    except Exception as e:
        st.error(f"Error processing the image: {e}")
