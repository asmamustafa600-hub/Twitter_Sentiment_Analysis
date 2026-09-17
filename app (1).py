import json
import time

import joblib
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from preprocess import clean_text

MODEL_DIR = "models"
RNN_MAX_LEN = 40

SENTIMENT_THEMES = {
    "Positive": {"color": "#059669", "bg": "#ECFDF5", "icon": "\U0001F60A"},
    "Negative": {"color": "#DC2626", "bg": "#FEF2F2", "icon": "\U0001F621"},
    "Neutral": {"color": "#2563EB", "bg": "#EFF6FF", "icon": "\u2696\uFE0F"},
    "Irrelevant": {"color": "#7C3AED", "bg": "#F5F3FF", "icon": "\U0001F3AF"},
}

CUSTOM_CSS = """
<style>
.app-header {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    border-radius: 14px;
    padding: 22px 28px;
    color: #FFFFFF;
    margin-bottom: 22px;
}
.app-title { font-size: 1.8rem; font-weight: 800; margin: 0; }
.app-subtitle { color: #CBD5E1; font-size: 0.92rem; margin: 6px 0 0 0; }
.result-card { border-radius: 14px; padding: 22px 26px; margin-top: 16px; }
.sentiment-heading { font-size: 1.9rem; font-weight: 800; margin: 6px 0; }
.confidence-val { font-size: 2rem; font-weight: 800; }
.prob-row { margin-bottom: 12px; }
.prob-labels { display: flex; justify-content: space-between; font-weight: 700; font-size: 0.9rem; margin-bottom: 4px; }
.prob-track { width: 100%; height: 10px; background: #F1F5F9; border-radius: 9999px; overflow: hidden; }
.prob-fill { height: 100%; border-radius: 9999px; }
</style>
"""


@st.cache_resource
def load_naive_bayes():
    model = joblib.load(f"{MODEL_DIR}/naive_bayes_model.joblib")
    vectorizer = joblib.load(f"{MODEL_DIR}/tfidf_vectorizer.joblib")
    with open(f"{MODEL_DIR}/nb_accuracy.json") as f:
        accuracy = json.load(f)["accuracy"]
    return model, vectorizer, accuracy


@st.cache_resource
def load_rnn():
    model = load_model(f"{MODEL_DIR}/rnn_model.keras")
    tokenizer = joblib.load(f"{MODEL_DIR}/rnn_tokenizer.joblib")
    with open(f"{MODEL_DIR}/rnn_accuracy.json") as f:
        accuracy = json.load(f)["accuracy"]
    return model, tokenizer, accuracy


@st.cache_resource
def load_label_encoder():
    return joblib.load(f"{MODEL_DIR}/label_encoder.joblib")


def predict_naive_bayes(text, label_encoder):
    start = time.time()
    model, vectorizer, accuracy = load_naive_bayes()
    features = vectorizer.transform([clean_text(text)])
    probs = model.predict_proba(features)[0]
    label = label_encoder.inverse_transform([np.argmax(probs)])[0]
    return label, probs, accuracy, (time.time() - start) * 1000


def predict_rnn(text, label_encoder):
    start = time.time()
    model, tokenizer, accuracy = load_rnn()
    sequence = tokenizer.texts_to_sequences([clean_text(text)])
    padded = pad_sequences(sequence, maxlen=RNN_MAX_LEN, padding="post", truncating="post")
    probs = model.predict(padded, verbose=0)[0]
    label = label_encoder.inverse_transform([np.argmax(probs)])[0]
    return label, probs, accuracy, (time.time() - start) * 1000


def render_result_card(label, confidence, accuracy, model_name, latency_ms):
    theme = SENTIMENT_THEMES[label]
    st.markdown(
        f"""
        <div class="result-card" style="background:{theme['bg']}; border:2px solid {theme['color']};">
            <div style="display:flex; justify-content:space-between; align-items:flex-start; flex-wrap:wrap;">
                <h2 class="sentiment-heading" style="color:{theme['color']};">{theme['icon']} {label}</h2>
                <div style="text-align:right;">
                    <div class="confidence-val" style="color:{theme['color']};">{confidence * 100:.1f}%</div>
                    <div style="font-size:0.78rem; color:#64748B;">CONFIDENCE</div>
                </div>
            </div>
            <div style="margin-top:10px; font-size:0.85rem; color:#475569;">
                <b>{model_name}</b> \u00b7 validation accuracy {accuracy * 100:.2f}% \u00b7 {latency_ms:.1f} ms
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_probability_bars(classes, probs):
    st.markdown("#### Probability breakdown")
    for cls, prob in zip(classes, probs):
        theme = SENTIMENT_THEMES[cls]
        pct = prob * 100
        st.markdown(
            f"""
            <div class="prob-row">
                <div class="prob-labels">
                    <span>{theme['icon']} {cls}</span>
                    <span style="color:{theme['color']};">{pct:.2f}%</span>
                </div>
                <div class="prob-track">
                    <div class="prob-fill" style="width:{max(pct, 2):.2f}%; background:{theme['color']};"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def main():
    st.set_page_config(page_title="Twitter Sentiment Classifier", page_icon="\U0001F426", layout="centered")
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="app-header">
            <h1 class="app-title">\U0001F426 Twitter Sentiment Classifier</h1>
            <p class="app-subtitle">Classify a tweet as Positive, Negative, Neutral, or Irrelevant.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    label_encoder = load_label_encoder()

    model_choice = st.selectbox("Choose a model:", ["Multinomial NB", "RNN"])
    tweet = st.text_area("Tweet text:", placeholder="Type or paste a tweet here...", height=110)

    if st.button("Analyze Sentiment", type="primary"):
        if not tweet.strip():
            st.warning("Please enter some text first.")
        else:
            if model_choice == "Multinomial NB":
                label, probs, accuracy, latency = predict_naive_bayes(tweet, label_encoder)
            else:
                label, probs, accuracy, latency = predict_rnn(tweet, label_encoder)

            render_result_card(label, np.max(probs), accuracy, model_choice, latency)
            render_probability_bars(label_encoder.classes_, probs)


if __name__ == "__main__":
    main()
