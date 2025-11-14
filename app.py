import streamlit as st
from components import summarizer, sentiment, keywords, vector_search, visualizer
import json
from pathlib import Path
import time

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
HISTORY_PATH = DATA_DIR / "history.json"

# Load / Save

def load_history():
    if HISTORY_PATH.exists():
        try:
            return json.loads(HISTORY_PATH.read_text(encoding='utf-8'))
        except Exception:
            return []
    return []

def save_history(history):
    HISTORY_PATH.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding='utf-8')

st.set_page_config(page_title="AI Blog Summarizer Dashboard", layout="wide")
st.title("🧠 AI Blog Summarizer & Insight Dashboard (Gemini NEW API)")


with st.sidebar:
    api_key = st.text_input("Gemini API Key", type="password")
    max_sent = st.slider("Max sentences (fallback)", 1, 10, 5)
    st.caption("Using google-genai new API.")

article = st.text_area("Paste your blog/article here", height=300)


if st.button("Analyze Article"):
    if not article.strip():
        st.warning("Please enter text.")
    else:
        with st.spinner("Analyzing with Gemini..."):
            summary = summarizer.generate_summary(article, api_key, max_sent)
            sentiment_scores = sentiment.analyze(article)
            top_keywords = keywords.extract(article)
            wc = keywords.make_wordcloud(article)

            hist = load_history()
            hist.append({
                "summary": summary,
                "sentiment": sentiment_scores,
                "keywords": top_keywords,
                "text": article,
                "date": time.strftime('%Y-%m-%d %H:%M:%S')
            })
            save_history(hist)

        st.subheader("Summary")
        st.write(summary)
        visualizer.plot_sentiment(sentiment_scores)
        st.image(wc.to_image())


# ========== Vector Search + Cluster Visualization ==========
st.write("---")
st.subheader("🔍 Semantic Search & Cluster Visualization (Gemini Embeddings)")

search_query = st.text_input("Search similar articles:")

if search_query and api_key:
    with st.spinner("Searching and clustering..."):
        hist = load_history()

        if len(hist) > 2:
            vector_search.semantic_search(hist, search_query, api_key)
            fig = vector_search.cluster_visualization(hist, api_key)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Add at least 3 articles for clustering.")

st.write("---")
st.caption("Built with ❤️ using the NEW Gemini API (google-genai)")