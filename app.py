import sys
import types
sys.modules['imghdr'] = types.ModuleType('imghdr')

import streamlit as st
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
    return " ".join(tokens)

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# ---------- PREMIUM CSS ----------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f172a, #020617);
}

/* Title */
.title {
    text-align: center;
    font-size: 38px;
    font-weight: 600;
    color: white;
}
.subtitle {
    text-align: center;
    font-size: 16px;
    color: #94a3b8;
    margin-bottom: 30px;
}

/* Card */
.card {
    background: rgba(255,255,255,0.05);
    padding: 25px;
    border-radius: 16px;
    backdrop-filter: blur(10px);
    box-shadow: 0px 4px 30px rgba(0,0,0,0.3);
}

/* Buttons */
.stButton button {
    border-radius: 10px;
    height: 45px;
    font-weight: 500;
    background: linear-gradient(90deg, #38bdf8, #6366f1);
    color: white;
    border: none;
}

/* Textbox */
textarea {
    border-radius: 10px !important;
}

/* Example buttons */
.example-btn button {
    background: #1e293b !important;
    color: #cbd5f5 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<div class="title">🎬 Movie Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyze reviews using NLP & Machine Learning</div>', unsafe_allow_html=True)

# ---------- CARD START ----------
st.markdown('<div class="card">', unsafe_allow_html=True)

# ---------- INPUT ----------
user_input = st.text_area(
    "Write your review",
    value=st.session_state.get("input", ""),
    height=120
)

# ---------- EXAMPLES ----------
st.markdown("**Quick examples:**")

col1, col2 = st.columns(2)

if col1.button("✨ Amazing movie"):
    st.session_state["input"] = "This movie was absolutely amazing and inspiring!"

if col2.button("💔 Waste of time"):
    st.session_state["input"] = "This was the worst movie ever and very boring."

# ---------- PREDICT ----------
if st.button("🚀 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Enter a review first")
    else:
        with st.spinner("Analyzing..."):
            cleaned = clean_text(user_input)
            vec = vectorizer.transform([cleaned])

            pred = model.predict(vec)[0]
            prob = model.predict_proba(vec)[0].max()

        st.markdown("---")

        confidence = round(prob * 100, 2)

        if pred == 1:
            st.success(f"🎉 Positive Sentiment")
            st.markdown(f"Confidence: **{confidence}%**")
            st.progress(int(confidence))
        else:
            st.error(f"😞 Negative Sentiment")
            st.markdown(f"Confidence: **{confidence}%**")
            st.progress(int(confidence))

# ---------- CARD END ----------
st.markdown('</div>', unsafe_allow_html=True)

# ---------- FOOTER ----------
st.markdown(
    "<center style='margin-top:20px;color:#64748b;'>Built by MADARAPU RAVICHANDANA 🚀</center>",
    unsafe_allow_html=True
)
