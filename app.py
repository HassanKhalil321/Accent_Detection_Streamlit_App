import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import os
import tempfile

allowed_labels = [
    "India and South Asia (India, Pakistan, Sri Lanka)",
    "German English, Non native speaker",
    "Southern African (South Africa, Zimbabwe, Namibia)",
    "Filipino",
    "Hong Kong English",
    "Singaporean English"
]

AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".ogg")
VIDEO_EXTENSIONS = (".mp4", ".webm", ".mov")

@st.cache_resource
def load_your_model():
    model_path = "my_accent_model.keras"
    return load_model(model_path)

def preprocess_audio(uploaded_file):
    filename = uploaded_file.name.lower()

    # Check file extension
    if filename.endswith(AUDIO_EXTENSIONS) or filename.endswith(VIDEO_EXTENSIONS):
        # Simulate real preprocessing – replace this with your actual logic
        dummy_input = np.random.rand(1, 80).astype(np.float32)
        return dummy_input
    else:
        return None

def main():
    st.title("Accent Detection")

    uploaded_file = st.file_uploader("Upload an audio or video file (wav, mp3, mp4, etc.)")

    if uploaded_file is not None:
        input_data = preprocess_audio(uploaded_file)

        if input_data is None:
            st.error("Invalid file type. Please upload an audio or video file.")
        else:
            model = load_your_model()
            predictions = model.predict(input_data)[0]
            predicted_index = np.argmax(predictions)
            predicted_label = allowed_labels[predicted_index]
            confidence = predictions[predicted_index] * 100

            st.success(f"Predicted Accent: **{predicted_label}**")
            st.write(f"Confidence: **{confidence:.2f}%**")

if __name__ == "__main__":
    main()
