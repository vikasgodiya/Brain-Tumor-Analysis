# ✅ Top section remains
import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

st.set_page_config(page_title="Brain Tumor MRI Classifier", layout="centered")

@st.cache_resource
def load_model_and_labels():
    model = load_model("C:\\Users\\vikas\\OneDrive\\Desktop\\custom_cnn_best_model.h5")
    with open("C:\\Users\\vikas\\OneDrive\\Desktop\\labels.txt", "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return model, labels

model, class_names = load_model_and_labels()

# ✅ Everything else here:
if __name__ == "__main__":
    # Streamlit UI layout
    st.markdown(
        """
        <style>
        .title {
            font-size:40px !important;
            text-align: center;
            color: #222222;
            font-weight: bold;
        }
        .footer {
            position: fixed;
            bottom: 10px;
            width: 100%;
            text-align: center;
            color: gray;
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown('<div class="title">🧠 Brain Tumor MRI Classifier</div>', unsafe_allow_html=True)
    st.markdown("## Upload a brain MRI image for tumor classification")

    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

        img_resized = img.resize((224, 224))
        img_array = image.img_to_array(img_resized) / 255.0
        img_expanded = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_expanded)
        pred_index = np.argmax(predictions)
        pred_label = class_names[pred_index]
        confidence = predictions[0][pred_index] * 100

        st.markdown("### 🧾 Prediction Result")
        st.success(f"**Predicted Tumor Type:** {pred_label}")
        st.info(f"**Confidence:** {confidence:.2f}%")

        st.markdown("### 📊 Confidence for All Classes")
        for i, label in enumerate(class_names):
            st.write(f"{label}: {predictions[0][i]*100:.2f}%")
    else:
        st.warning("Please upload an image.")

    st.markdown('<div class="footer">Made with ❤️ using Streamlit & TensorFlow</div>', unsafe_allow_html=True)
