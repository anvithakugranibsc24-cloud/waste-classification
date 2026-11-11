
import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image

st.set_page_config(
    page_title="AI Waste Classification",
    page_icon="♻️",
    layout="centered"
)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "waste_classification_cnn.keras"
    )

model = load_model()

with open("class_names.json", "r") as f:
    class_names = json.load(f)

st.title("♻️ AI-Based Waste Classification")
st.write(
    "Upload a waste image to identify its category "
    "using deep learning."
)

uploaded_file = st.file_uploader(
    "Choose a waste image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Waste Image",
        use_container_width=True
    )

    if st.button("Classify Waste", type="primary"):

        resized_image = image.resize((224, 224))

        image_array = np.array(
            resized_image,
            dtype=np.float32
        )

        image_array = np.expand_dims(
            image_array,
            axis=0
        )

        prediction = model.predict(image_array)

        predicted_index = np.argmax(prediction[0])

        predicted_class = class_names[predicted_index]

        confidence = float(
            prediction[0][predicted_index] * 100
        )

        st.success(
            f"Predicted Category: {predicted_class.title()}"
        )

        st.metric(
            "Prediction Confidence",
            f"{confidence:.2f}%"
        )

        st.subheader("Category Probabilities")

        for name, score in zip(class_names, prediction[0]):
            st.write(f"{name.title()}: {score * 100:.2f}%")
            st.progress(float(score))
