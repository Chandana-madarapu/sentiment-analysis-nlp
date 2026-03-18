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

# Keep negation words
stop_words = set(stopwords.words('english')) - {"not", "no", "never"}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    tokens = word_tokenize(text)

    # handle negation
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

# ---------- PAGE ----------
st.set_page_config(page_title="🎬 Sentiment Analyzer", layout="centered")

st.title("🎬 Movie Sentiment Analyzer")
st.caption("Analyze one or multiple reviews")

# ---------- INPUT ----------
user_input = st.text_area(
    "Write review (or multiple reviews - one per line)",
    value=st.session_state.get("input", ""),
    height=150
)

# ---------- PREDICT ----------
if st.button("Analyze"):

    if user_input.strip() == "":
        st.warning("Enter a review first")
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
                st.success(f"👍 {review}\n\nPositive ({confidence}%)")
            else:
                st.error(f"👎 {review}\n\nNegative ({confidence}%)")

# ---------- EXAMPLES (MOVED BELOW RESULTS) ----------
st.markdown("---")
st.markdown("**Try examples:**")

col1, col2 = st.columns(2)

if col1.button("👍 Positive Example"):
    st.session_state["input"] = "This movie was amazing and inspiring!"

if col2.button("👎 Negative Example"):
    st.session_state["input"] = "This movie was not good and very boring."

# ---------- FOOTER ----------
st.markdown("---")
st.markdown(
    "<center style='color:gray;'>Built by MADARAPU RAVICHANDANA 🚀</center>",
    unsafe_allow_html=True
)
