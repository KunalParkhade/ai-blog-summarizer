import google.generativeai as genai
import streamlit as st
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

def extractive_summary(text: str, max_sentences: int=5) -> str:
    sentences = sent_tokenize(text)
    if len(sentences) <= max_sentences:
        return text
    
    words = [w.lower() for w in word_tokenize(text) if w.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if w not in stop_words]
    freq = {w: words.count(w) for w in set(words)}
    scores = [(sum(freq.get(w.lower(), 0) for w in word_tokenize(s)), s) for s in sentences]
    top = [s for _, s in sorted(scores, reverse=True)[:max_sentences]]
    return " ".join([s for s in sentences if s in top])

def generate_summary(text: str, api_key: str, max_sentences: int):
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"Summarize clearly with title, key points and tone:\n{text}"
        resp = model.generate_content(prompt)
        return resp.text.strip()
    except Exception as e:
        st.warning(f"Gemini failed, using fallback. ({e})")
        return extractive_summary(text, max_sentences)