"""Streamlit interface for the BERT fake-news classifier."""

import os

import streamlit as st

from detector import FakeNewsDetector, InputValidationError


st.set_page_config(page_title="Fake News Detection with BERT", page_icon="📰", layout="centered")
MODEL_NAME = os.getenv("HF_MODEL_NAME", "shi13u/fake_news_detection_bert")


def _hugging_face_token():
    """Read an optional Hugging Face token without requiring Streamlit secrets."""
    try:
        return st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")
    except (FileNotFoundError, KeyError):
        return os.getenv("HF_TOKEN")


@st.cache_resource(show_spinner=False)
def load_detector(model_name, token):
    return FakeNewsDetector.from_pretrained(model_name=model_name, token=token)


st.title("Fake News Detection with BERT")
st.caption("Classify an English news passage as likely fake or real using a fine-tuned BERT model.")
st.info(
    "This is a machine-learning classification demo, not a fact-checking service. "
    "Verify important claims with trustworthy primary sources."
)

with st.form("classification_form"):
    article = st.text_area(
        "News text", height=220, max_chars=20_000,
        placeholder="Paste a headline and article text here...",
    )
    submitted = st.form_submit_button("Analyze article", type="primary", use_container_width=True)

if submitted:
    try:
        with st.spinner("Loading the model and analyzing the text..."):
            detector = load_detector(MODEL_NAME, _hugging_face_token())
            result = detector.predict(article)
    except InputValidationError as exc:
        st.warning(str(exc))
    except OSError:
        st.error("The model could not be downloaded. Check the network connection or model access settings.")
    except Exception as exc:
        st.error("Prediction failed. Please try again with a shorter passage.")
        with st.expander("Technical details"):
            st.code(str(exc))
    else:
        left, right = st.columns(2)
        left.metric("Prediction", result.label.title())
        right.metric("Confidence", f"{result.confidence:.1%}")
        st.progress(result.confidence)
        probability_left, probability_right = st.columns(2)
        probability_left.write(f"**Fake probability:** {result.fake_probability:.1%}")
        probability_right.write(f"**Real probability:** {result.real_probability:.1%}")
        if result.was_truncated:
            st.warning(
                f"The input contained about {result.token_count} tokens; only the first 512 "
                "were used because of the model limit."
            )
        if result.confidence_band == "low":
            st.warning("The model is uncertain about this passage. Treat the result cautiously.")

with st.expander("How to interpret the result"):
    st.write(
        "The score reflects patterns learned from training data. It does not inspect sources, "
        "confirm quotations, or search the web. Check the publisher, author, date, supporting "
        "evidence, and independent reporting before trusting a claim."
    )
