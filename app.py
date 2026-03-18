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

# ---------- PAGE ----------
st.set_page_config(page_title="AI Sentiment Analyzer", layout="centered")

# ---------- STYLE ----------
st.markdown("""
<style>
body {
    background: #020617;
}

/* Title */
.title {
    text-align: center;
    font-size: 34px;
    font-weight: 600;
    color: white;
    margin-bottom: 5px;
}
.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 25px;
}

/* Chat bubbles */
.user-msg {
    background: #1e293b;
    padding: 12px 16px;
    border-radius: 12px;
    margin: 10px 0;
    color: white;
}
.bot-msg {
    background: #0ea5e9;
    padding: 12px 16px;
    border-radius: 12px;
    margin: 10px 0;
    color: white;
}

/* Input */
textarea {
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<div class="title">🎬 AI Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Understand movie reviews instantly</div>', unsafe_allow_html=True)

# ---------- SESSION MEMORY ----------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------- DISPLAY CHAT ----------
for chat in st.session_state.history:
    if chat["role"] == "user":
        st.markdown(f"<div class='user-msg'>🧑 {chat['text']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-msg'>🤖 {chat['text']}</div>", unsafe_allow_html=True)

# ---------- INPUT ----------
user_input = st.text_area("Type your review", height=100)

# ---------- QUICK EXAMPLES ----------
st.markdown("**Try:**")

col1, col2 = st.columns(2)

if col1.button("✨ Amazing movie"):
    user_input = "This movie was absolutely amazing and inspiring!"

if col2.button("💔 Waste of time"):
    user_input = "This was the worst movie ever and very boring."

# ---------- PREDICT ----------
if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Enter a review")
    else:
        # Save user message
        st.session_state.history.append({"role": "user", "text": user_input})

        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])

        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0].max()

        confidence = round(prob * 100, 2)

        if pred == 1:
            response = f"🎉 Positive Sentiment\nConfidence: {confidence}%"
        else:
            response = f"😞 Negative Sentiment\nConfidence: {confidence}%"

        # Save bot response
        st.session_state.history.append({"role": "bot", "text": response})

        st.rerun()

# ---------- FOOTER ----------
st.markdown(
    "<center style='color:#64748b;margin-top:20px;'>Built by MADARAPU RAVICHANDANA 🚀</center>",
    unsafe_allow_html=True
)
