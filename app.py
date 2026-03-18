import streamlit as st
import pickle
import re

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Manual stopwords (no nltk dependency)
stop_words = {
    "a","an","the","and","or","but","if","while","is","am","are","was","were",
    "be","been","being","have","has","had","do","does","did","of","to","in",
    "for","on","with","as","by","at","from","this","that","it","i","you","he",
    "she","they","we","me","him","her","them","my","your","his","their"
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

# UI
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
