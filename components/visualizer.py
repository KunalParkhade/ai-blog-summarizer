import plotly.graph_objects as go
import streamlit as st

def plot_sentiment(scores):
    labels, values = list(scores.keys()), [scores[k] for k in scores]
    fig = go.Figure(go.Bar(x=labels, y=values))
    fig.update_layout(title_text="Sentiment", yaxis=dict(range=[-1, 1]))
    st.plotly_chart(fig, use_container_width=True)