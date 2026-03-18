import streamlit as st
import pickle
import re
import nltk
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# NLTK setup
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
nltk.data.path.append(nltk_data_path)

nltk.download('punkt', download_dir=nltk_data_path)
nltk.download('punkt_tab', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

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
st.set_page_config(page_title="Sentiment Analyzer", layout="centered")

st.title("🎬 Advanced Sentiment Analyzer")
st.write("Analyze reviews with NLP + Visualization")

# Input
user_input = st.text_area("Enter one or multiple reviews (one per line):")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Enter some text!")
    else:
        reviews = user_input.split("\n")
        results = []
        all_text = ""

        for review in reviews:
            cleaned = clean_text(review)
            vec = vectorizer.transform([cleaned])
            pred = model.predict(vec)[0]
            results.append(pred)
            all_text += " " + cleaned

        # Display results
        for i, review in enumerate(reviews):
            if results[i] == 1:
                st.success(f"👍 {review}")
            else:
                st.error(f"👎 {review}")

        # 📊 Sentiment chart
        st.subheader("📊 Sentiment Distribution")
        pos = results.count(1)
        neg = results.count(0)

        st.bar_chart({"Positive": pos, "Negative": neg})

        # ☁️ WordCloud
        st.subheader("☁️ Word Cloud")
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

        fig, ax = plt.subplots()
        ax.imshow(wordcloud)
        ax.axis("off")

        st.pyplot(fig)
