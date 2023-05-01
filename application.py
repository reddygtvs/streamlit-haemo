import streamlit as st
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import numpy as np
import io
import cv2

# Load the models
model = load_model("models/binaryClassifierV2.h5")
classify = load_model("models/multiClass3.h5")

# Define a function to preprocess the image
def preprocess_image(img):
    img = tf.image.resize(img, (256, 256))
    return img

def predict(image):
    # Make prediction with binary classifier
    img = preprocess_image(image)
    prediction = model.predict(np.expand_dims(img/255, 0))
    if prediction > 0.5:
        # If positive, use multi class classifier
        result = classify.predict(np.expand_dims(img/255, 0))
        prediction = np.argmax(result)
        if prediction == 0:
            prediction_label = "Blood sample is infected - **Anaplasmosis**"
        elif prediction == 1:
            prediction_label = "Blood sample is infected - **Babesiosis**"
        else:
            prediction_label = "Blood sample is infected - **Theileriosis**"
    else:
        prediction_label = "Blood sample is not infected"
    return prediction_label

# Set the page configuration and title
st.set_page_config(page_title="Blood Sample Classifier")
st.title("Blood Sample Classifier")

# Add a button to upload the image
uploaded_file = st.file_uploader("Upload a blood sample image", type="jpg")

camera = None
# Add a button to start the camera
if st.button("Take a picture"):
    camera = st.camera_input("Take a picture")

# Get the image from the uploaded file or the camera and predict
if uploaded_file is not None:
    image = Image.open(io.BytesIO(uploaded_file.read()))
    st.image(image, caption="Blood sample", use_column_width=True)

    if st.button("Predict from uploaded file"):
        prediction = predict(image)
        st.header(f"Prediction: {prediction}")
elif camera is not None:
    if st.button("Predict from camera"):
        image = Image.open(camera)
        prediction = predict(image)
        st.header(f"Prediction: {prediction}")
