import streamlit as st
import pickle
import re

# ----------------------------
# Load model
# ----------------------------
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# ----------------------------
# Text cleaning
# ----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="🧠",
    layout="centered"
)

# ----------------------------
# Clean Styling
# ----------------------------
st.markdown("""
<style>
body {
    background-color: #0f172a;
}

.main {
    background-color: #0f172a;
}

h1 {
    font-size: 34px;
    font-weight: 600;
    color: #e5e7eb;
}

.subtext {
    color: #9ca3af;
    margin-bottom: 25px;
}

.stTextArea textarea {
    background-color: #111827;
    color: #e5e7eb;
    border-radius: 10px;
    border: 1px solid #374151;
}

.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 500;
}

.result {
    padding: 14px;
    border-radius: 8px;
    margin-top: 15px;
    font-size: 16px;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Header
# ----------------------------
st.markdown("## 🧠 Sentiment Analysis")
st.markdown('<div class="subtext">Analyze customer reviews using machine learning</div>', unsafe_allow_html=True)

# ----------------------------
# Input
# ----------------------------
review = st.text_area("Enter review")

# ----------------------------
# Prediction
# ----------------------------
if st.button("Analyze"):
    if review.strip() == "":
        st.warning("Please enter a review")
    else:
        cleaned = clean_text(review)
        vector = tfidf.transform([cleaned])
        result = model.predict(vector)[0]

        if result == 1:
            st.markdown('<div class="result" style="background-color:#052e16; color:#22c55e;">✔ Positive Review</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result" style="background-color:#3f0d0d; color:#ef4444;">✖ Negative Review</div>', unsafe_allow_html=True)

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.caption("Model: LinearSVC | TF-IDF Features")