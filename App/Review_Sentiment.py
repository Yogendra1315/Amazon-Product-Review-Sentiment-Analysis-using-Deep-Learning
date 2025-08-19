import streamlit as st
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# -------------------------------
# Load model and tokenizer once
# -------------------------------
@st.cache_resource
def load_resources():
    model = load_model("Sentiment.h5", compile=False)
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_resources()

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="Sentiment Analysis App", page_icon="😊", layout="centered")
st.title("📝 Sentiment Analysis App")
st.markdown("Enter your text below to predict if it has a positive or negative sentiment.")

user_input = st.text_area("Type your review here:", height=150)

if st.button("Predict"):
    if not user_input.strip():
        st.warning("⚠️ Please enter some text!")
    else:
        # Sanitize input
        safe_input = re.sub(r"[<>]", "", user_input)

        # Convert input to sequence and pad
        seq = tokenizer.texts_to_sequences([safe_input])
        seq_padded = pad_sequences(seq, maxlen=100)
        pred = model.predict(seq_padded)

        # Corrected logic: pred < 0.5 = Positive, pred >= 0.5 = Negative
        sentiment = "Positive 😊" if pred[0][0] < 0.5 else "Negative 😞"
        confidence = (1 - pred[0][0]) if pred[0][0] < 0.5 else pred[0][0]

        # Set background color
        color = "#d4edda" if sentiment.startswith("Positive") else "#f8d7da"

        # Display output
        st.markdown(
            f"""
            <div style='background-color:{color};padding:15px;border-radius:10px'>
                <h4>Prediction Score: {confidence:.2%}</h4>
                <h3>Sentiment: {sentiment}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )