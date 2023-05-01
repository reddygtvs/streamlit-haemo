import streamlit as st
from keras.models import load_model
from PIL import Image
import tensorflow as tf
import io
import numpy as np

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
            prediction_label = "Anaplasmosis"
        elif prediction == 1:
            prediction_label = "Babesiosis"
        else:
            prediction_label = "Theileriosis"
    else:
        prediction_label = "Blood sample is not infected"
    return prediction_label

st.set_page_config(page_title="Blood Sample Classifier")

st.title("Blood Sample Classifier")

uploaded_file = st.file_uploader("Choose a blood sample image", type="jpg")

if uploaded_file is not None:
    image = Image.open(io.BytesIO(uploaded_file.read()))
    st.image(image, caption="Blood sample", use_column_width=True)

    if st.button("Predict"):
        prediction = predict(image)
        st.write(f"Prediction: {prediction}")
