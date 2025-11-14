import faiss
import numpy as np
from google.genai import Client
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd
import streamlit as st


def embed_texts(texts: list[str], api_key: str) -> np.ndarray:
    client = Client(api_key=api_key)
    result = client.models.embed_content(
        model="gemini-embedding-1",
        contents=texts
    )

    embeddings = result.get("embeddings") or result.get("embedding")
    return np.array(embeddings, dtype="float32")

def semantic_search(history, query, api_key):
    summaries = [h['summary'] for h in history]
    emb = embed_texts(summaries, api_key)
    q_emb = embed_texts([query], api_key)

    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb)
    D, I = index.search(q_emb, k=min(3, len(summaries)))

    st.markdown("### 🔎 Top Similar Summaries")
    for idx in I[0]:
        st.write(f"- {history[idx]['summary'][:250]}...")


def cluster_visualization(history, api_key):
    summaries = [h['summary'] for h in history]
    emb = embed_texts(summaries, api_key)

    coords = PCA(n_components=2).fit_transform(emb)
    df = pd.DataFrame(coords, columns=["x", "y"])
    df['label'] = [h['summary'][:60] for h in history]

    fig = px.scatter(df, x="x", y="y", text="label", title="Clusters of Summaries (FAISS + PCA)")
    return fig