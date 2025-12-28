import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# Load model & tfidf
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# NLP tools
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = text.split()
    text = [stemmer.stem(w) for w in text if w not in stop_words]
    return " ".join(text)

# UI Streamlit
st.set_page_config(page_title="Analisis Sentimen", page_icon="📝")
st.title("📝 Analisis Sentimen Review Produk")

review = st.text_area("Masukkan Review Produk:")

if st.button("Prediksi Sentimen"):
    if review.strip() == "":
        st.warning("Teks tidak boleh kosong")
    else:
        clean = clean_text(review)
        vector = tfidf.transform([clean])
        result = model.predict(vector)[0]

        if result == "positive":
            st.success("✅ Sentimen POSITIF")
        else:
            st.error("❌ Sentimen NEGATIF")


