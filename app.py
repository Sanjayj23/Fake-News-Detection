import streamlit as st
import torch
import numpy as np
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ✅ Load Hugging Face token from environment
HF_TOKEN = os.getenv("HF_TOKEN")  # Don't hardcode the token!

# ✅ Hugging Face model name (your private or gated repo)
MODEL_NAME = "shi13u/fake_news_detection_bert"

# ✅ Load model and tokenizer securely
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, token=HF_TOKEN)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# ✅ Streamlit app config
st.set_page_config(page_title="📰 Fake News Detector", layout="centered")

# ✅ Title and description
st.markdown("<h1 style='text-align: center;'>🤖 Fake News Detection Chatbot</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center;'>Paste a news snippet below to check if it's <strong>Real ✅</strong> or <strong>Fake ❌</strong> using BERT!</p>",
    unsafe_allow_html=True
)

# ✅ Input field
user_input = st.text_area("🗞️ Enter News Article or Headline:", height=200)

# ✅ Button to trigger prediction
if st.button("🔍 Check Now"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        # Tokenize input
        inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1).numpy()[0]
            pred = np.argmax(probs)
            confidence = float(np.max(probs) * 100)

        # Label interpretation (0 = FAKE, 1 = REAL — adjust if needed)
        if pred == 1:
            label = "✅ **This looks like REAL news!**"
            emoji = "🟢"
        else:
            label = "❌ **This might be FAKE news!**"
            emoji = "🔴"

        # ✅ Show result
        st.markdown(f"### {emoji} {label}")
        st.markdown(f"📊 **Confidence:** `{confidence:.2f}%`")
