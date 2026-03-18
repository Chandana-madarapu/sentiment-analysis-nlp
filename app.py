import sys
import types
sys.modules['imghdr'] = types.ModuleType('imghdr')

import streamlit as st
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# 🔥 FIX: keep negation words
stop_words = set(stopwords.words('english')) - {"not", "no", "never"}

# 🔥 FIX: better preprocessing
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    tokens = word_tokenize(text)

    # handle negation (not good → not_good)
    new_tokens = []
    skip = False

    for i in range(len(tokens)):
        if skip:
            skip = False
            continue

        if tokens[i] in ["not", "no", "never"] and i+1 < len(tokens):
            new_tokens.append(tokens[i] + "_" + tokens[i+1])
            skip = True
        else:
            new_tokens.append(tokens[i])

    new_tokens = [w for w in new_tokens if w not in stop_words and len(w) > 2]

    return " ".join(new_tokens)

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="🎬 Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
body {
    background-color: #0f172a;
}
.main {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #38bdf8;
}
.subtitle {
    text-align: center;
    font-size: 18px;
    color: #cbd5f5;
    margin-bottom: 25px;
}
.stTextArea textarea {
    border-radius: 12px;
    padding: 12px;
    font-size: 16px;
}
.stButton button {
    background: linear-gradient(90deg, #38bdf8, #6366f1);
    color: white;
    border-radius: 12px;
    height: 50px;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<div class="title">🎬 Movie Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyze reviews using NLP & Machine Learning</div>', unsafe_allow_html=True)

# ---------- EXAMPLES ----------
st.markdown("### 📝 Enter or Choose a Review")

col1, col2 = st.columns(2)

if col1.button("👍 Use Positive Example"):
    st.session_state["input"] = "This movie was absolutely amazing and inspiring!"

if col2.button("👎 Use Negative Example"):
    st.session_state["input"] = "This was not good at all and very boring."

# ---------- INPUT ----------
user_input = st.text_area(
    "✍️ Write your review",
    value=st.session_state.get("input", ""),
    height=150
)

# ---------- SINGLE BUTTON (FIXED) ----------
if st.button("🎯 Predict Sentiment"):

    if user_input.strip() == "":
        st.warning("⚠️ Please enter a review first!")
    else:
        with st.spinner("Analyzing sentiment... ⏳"):
            cleaned = clean_text(user_input)
            vec = vectorizer.transform([cleaned])

            pred = model.predict(vec)[0]
            prob = model.predict_proba(vec)[0].max()

        st.markdown("---")

        confidence = round(prob * 100, 2)

        if pred == 1:
            st.success(f"🎉 Positive Review! 😊\nConfidence: {confidence}%")
            st.progress(int(confidence))
        else:
            st.error(f"😞 Negative Review!\nConfidence: {confidence}%")
            st.progress(int(confidence))

# ---------- FOOTER ----------
st.markdown("---")
st.markdown(
    """
    <center style='color:#94a3b8; font-size:14px;'>
    Built with ❤️ by <b style='color:#38bdf8;'>MADARAPU RAVICHANDANA</b> 🚀
    </center>
    """,
    unsafe_allow_html=True
)
