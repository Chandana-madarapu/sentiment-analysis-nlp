import sys
import types

# Fix for Python 3.13+ (imghdr removed)
sys.modules['imghdr'] = types.ModuleType('imghdr')

import streamlit as st
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

# UI
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🎬")

st.title("🎬 Movie Sentiment Analyzer")
st.write("Enter a movie review and get sentiment prediction")

user_input = st.text_area("Enter review:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter text")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)

        if pred[0] == 1:
            st.success("Positive 😊")
        else:
            st.error("Negative 😞")
