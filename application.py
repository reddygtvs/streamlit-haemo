import streamlit as st
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import numpy as np
import io

model = load_model("models/binaryClassifierV2.h5")
classify = load_model("models/multiClass3.h5")

def preprocess_image(img):
    img = tf.image.resize(img, (256, 256))
    return img

def predict(image):
    
    img = preprocess_image(image)
    prediction = model.predict(np.expand_dims(img/255, 0))
    if prediction > 0.5:
       
        result = classify.predict(np.expand_dims(img/255, 0))
        prediction = np.argmax(result)
        if prediction == 0:
            prediction_label = "Haemoprotozoan sample is infected - **Anaplasmosis**"
        elif prediction == 1:
            prediction_label = "Haemoprotozoan sample is infected - **Babesiosis**"
        else:
            prediction_label = "Haemoprotozoan sample is infected - **Theileriosis**"
    else:
        prediction_label = "Sample is not infected"
    return prediction_label


st.set_page_config(page_title="Haemoprotozoan Classifier")
st.title("Haemoprotozoan Classifier")

uploaded_file = st.file_uploader("Choose a bloodstream image", type=["jpg", "jpeg", "png"])
st.subheader("OR")

camera = st.camera_input("Take a picture")

if uploaded_file is not None:
    image = Image.open(io.BytesIO(uploaded_file.read()))
    st.image(image, caption="Bloodstream sample", use_column_width=True)

    if st.button("Predict from uploaded file"):
        prediction = predict(image)
        st.header(f"Prediction: {prediction}")
elif camera is not None:
    if st.button("Predict from camera"):
        image = Image.open(camera)
        prediction = predict(image)
        st.header(f"Prediction: {prediction}")