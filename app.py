import streamlit as st
import pickle
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# =====================
# Load model & tfidf
# =====================
model = pickle.load(open("model.pkl", "rb"))
tfidf = pickle.load(open("tfidf.pkl", "rb"))

# =====================
# NLP tools (Indonesia)
# =====================
stem_factory = StemmerFactory()
stemmer = stem_factory.create_stemmer()

stop_factory = StopWordRemoverFactory()
stop_words = set(stop_factory.get_stop_words())

# =====================
# Text preprocessing
# =====================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

# =====================
# UI Streamlit
# =====================
st.set_page_config(
    page_title="Analisis Sentimen",
    page_icon="📝",
    layout="centered"
)

st.title("📝 Analisis Sentimen Review Produk")
st.write("Masukkan teks review, lalu sistem akan memprediksi sentimennya.")

review = st.text_area("Masukkan Review Produk:")

if st.button("Prediksi Sentimen"):
    if review.strip() == "":
        st.warning("⚠️ Teks tidak boleh kosong")
    else:
        clean = clean_text(review)
        vector = tfidf.transform([clean])
        result = model.predict(vector)[0]

        if result == "positive":
            st.success("✅ Sentimen POSITIF")
        else:
            st.error("❌ Sentimen NEGATIF")
