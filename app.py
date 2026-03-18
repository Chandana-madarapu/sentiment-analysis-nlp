import streamlit as st
import pickle
import re
import nltk
import os

# ✅ Setup NLTK data directory

nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(nltk_data_path)

# Download required datasets
nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('punkt_tab', download_dir=nltk_data_path)  # 🔥 IMPORTANT FIX
nltk.download('stopwords', download_dir=nltk_data_path)

# ✅ Download ONLY if missing
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

stop_words = set(stopwords.words('english'))

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

# UI
st.set_page_config(page_title="Sentiment Analyzer")

st.title("🎬 Movie Sentiment Analyzer")

user_input = st.text_area("Enter your review:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Enter something!")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])

        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec)[0].max()

        if pred == 1:
            st.success(f"Positive 😊 ({round(prob*100,2)}%)")
        else:
            st.error(f"Negative 😞 ({round(prob*100,2)}%)")
