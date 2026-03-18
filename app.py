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
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

# ---------- STYLE ----------
st.markdown("""
<style>
.title {
    text-align: center;
    font-size: 34px;
    font-weight: 600;
}
.subtitle {
    text-align: center;
    color: #94a3b8;
    margin-bottom: 20px;
}
textarea {
    border-radius: 10px !important;
}
.stButton button {
    border-radius: 8px;
    height: 45px;
}
.result {
    padding: 10px;
    border-radius: 8px;
    margin: 5px 0;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
st.markdown('<div class="title">🎬 Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyze one or multiple reviews</div>', unsafe_allow_html=True)

# ---------- INPUT ----------
user_input = st.text_area(
    "Write your reviews (one per line)",
    placeholder="Example:\nThis movie was amazing!\nWorst movie ever...",
    height=150
)

# ---------- PREDICT ----------
if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Enter at least one review")
    else:
        reviews = user_input.split("\n")

        st.markdown("### Results")

        for review in reviews:
            if review.strip() == "":
                continue

            cleaned = clean_text(review)
            vec = vectorizer.transform([cleaned])

            pred = model.predict(vec)[0]
            prob = model.predict_proba(vec)[0].max()
            confidence = round(prob * 100, 2)

            if pred == 1:
                st.markdown(
                    f"<div class='result' style='background:#dcfce7;'>"
                    f"👍 {review}<br><b>Positive ({confidence}%)</b>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='result' style='background:#fee2e2;'>"
                    f"👎 {review}<br><b>Negative ({confidence}%)</b>"
                    f"</div>",
                    unsafe_allow_html=True
                )

# ---------- FOOTER ----------
st.markdown("---")
st.markdown(
    "<center style='color:gray;font-size:13px;'>Built by MADARAPU RAVICHANDANA 🚀</center>",
    unsafe_allow_html=True
)
