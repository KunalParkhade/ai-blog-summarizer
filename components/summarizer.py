from google.genai import Client
from google.genai.types import GenerateContentConfig
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import streamlit as st

# Ensure NLTK models
for pkg in ["punkt", "punkt_tab", "stopwords"]:
    try:
        nltk.data.find(pkg)
    except LookupError:
        nltk.download(pkg)

def extractive_summary(text: str, max_sentences: int = 5) -> str:
    sentences = sent_tokenize(text)
    if len(sentences) <= max_sentences:
        return text
    words = [w.lower() for w in word_tokenize(text) if w.isalpha()]
    sw = set(stopwords.words('english'))
    words = [w for w in words if w not in sw]
    freq = {w: words.count(w) for w in set(words)}
    scored = [(sum(freq.get(w.lower(), 0) for w in word_tokenize(s)), s) for s in sentences]
    top = [s for _, s in sorted(scored, reverse=True)[:max_sentences]]
    return " ".join([s for s in sentences if s in top])


def generate_summary(text: str, api_key: str, max_sentences: int):
    if not api_key:
        return extractive_summary(text, max_sentences)

    try:
        client = Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=text,
            config=GenerateContentConfig(temperature=0.2)
        )
        return response.text
    except Exception as e:
        st.warning(f"Gemini failed, fallback used. ({e})")
        return extractive_summary(text, max_sentences)