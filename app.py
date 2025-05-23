import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

allowed_labels = [
    "India and South Asia (India, Pakistan, Sri Lanka)",
    "German English,Non native speaker",
    "Southern African (South Africa, Zimbabwe, Namibia)",
    "Filipino",
    "Hong Kong English",
    "Singaporean English"
]

@st.cache_resource
def load_your_model():
    model_path = "my_accent_model.keras"  # Replace with your actual model path
    model = load_model(model_path)
    return model

def preprocess_audio(file_path):
    # TODO: Replace this with actual preprocessing to produce input shape (1, 80)
    dummy_input = np.random.rand(1, 80).astype(np.float32)
    return dummy_input

def main():
    st.title("Accent Detection")

    uploaded_file = st.file_uploader("Upload an audio file (wav, mp3, etc.)")

    if uploaded_file is not None:
        input_data = preprocess_audio(uploaded_file)

        model = load_your_model()

        predictions = model.predict(input_data)[0]  # shape (num_classes,)

        predicted_index = np.argmax(predictions)
        predicted_label = allowed_labels[predicted_index]
        confidence = predictions[predicted_index] * 100

        st.write(f"Predicted Accent: **{predicted_label}**")
        st.write(f"Confidence: **{confidence:.2f}%**")

if __name__ == "__main__":
    main()
