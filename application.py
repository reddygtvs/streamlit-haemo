import streamlit as st
import tensorflow as tf
from keras.models import load_model
from PIL import Image
import numpy as np
import io
import cv2
import av
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

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

# Define a video transformer to capture the image from the camera
class VideoTransformer(VideoTransformerBase):
    def transform(self, frame):
        # Convert the frame to a PIL Image
        img = frame.to_image()
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = Image.fromarray(img)

        return img

# Set the page configuration and title
st.set_page_config(page_title="Blood Sample Classifier")
st.title("Blood Sample Classifier")

# Add a button to upload the image
uploaded_file = st.file_uploader("Choose a blood sample image", type="jpg")

# Add a button to start the webcam
webrtc_ctx = webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

# Get the image from the uploaded file or the webcam and predict
if uploaded_file is not None:
    image = Image.open(io.BytesIO(uploaded_file.read()))
    st.image(image, caption="Blood sample", use_column_width=True)

    if st.button("Predict from uploaded file"):
        prediction = predict(image)
        st.header(f"Prediction: {prediction}")
elif webrtc_ctx.video_transformer:
    if st.button("Predict from webcam"):
        image = webrtc_ctx.video_transformer.get_image()
        prediction = predict(image)
        st.header(f"Prediction: {prediction}")