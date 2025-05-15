import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

# Load the trained model with caching
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("mnist_cnn_model.h5")

model = load_model()

# Streamlit page settings
st.set_page_config(page_title="Digit Recognizer", layout="centered")
st.title("Handwritten Digit Recognizer")
st.markdown("Upload a digit image (28x28 pixels, white on black background).")

# Upload image input
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Process the image
    image = Image.open(uploaded_file).convert("L")  # grayscale
    image = ImageOps.invert(image)                  # white digit on black
    image = image.resize((28, 28))                  # resize to 28x28

    # Preprocess for model
    img_array = np.array(image).astype("float32") / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    # Show output
    st.image(image, caption="Processed Image", width=150)
    st.subheader(f"Predicted Digit: {predicted_digit}")
