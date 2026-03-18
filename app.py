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
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🎬",
    layout="centered"
)

# ---------- MINIMAL CSS ----------
st.markdown("""
<style>
.title {
    text-align: center;
    font-size: 36px;
    font-weight: 600;
}
.subtitle {
    text-align: center;
    font-size: 16px;
    color: #888;
    margin-bottom: 25px;
}
.stButton button {
    border-radius: 8px;
    height: 45px;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<div class="title">🎬 Movie Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyze movie reviews using NLP</div>', unsafe_allow_html=True)

# ---------- EXAMPLES (SUBTLE) ----------
st.markdown("**Try quick examples:**")

col1, col2 = st.columns(2)

if col1.button("👍 Positive"):
    st.session_state["input"] = "I loved this movie, it was amazing!"

if col2.button("👎 Negative"):
    st.session_state["input"] = "This was the worst movie ever."

# ---------- INPUT ----------
user_input = st.text_area(
    "Write your review",
    value=st.session_state.get("input", ""),
    height=120
)

# ---------- PREDICT ----------
if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Enter a review first")
    else:
        with st.spinner("Analyzing..."):
            cleaned = clean_text(user_input)
            vec = vectorizer.transform([cleaned])

            pred = model.predict(vec)[0]
            prob = model.predict_proba(vec)[0].max()

        st.markdown("---")

        if pred == 1:
            st.success(f"Positive 😊 ({round(prob*100,1)}%)")
        else:
            st.error(f"Negative 😞 ({round(prob*100,1)}%)")

# ---------- FOOTER ----------
st.markdown("---")
st.markdown(
    "<center style='font-size:13px;color:gray;'>Built by MADARAPU RAVICHANDANA</center>",
    unsafe_allow_html=True
)
