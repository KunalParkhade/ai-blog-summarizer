from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud

def extract(text: str, topk: int = 15):
    words = [w.lower() for w in word_tokenize(text) if w.isalpha()]
    sw = set(stopwords.words('english'))
    filtered = [w for w in words if w not in sw and len(w) > 2]
    freq = {w: filtered.count(w) for w in set(filtered)}
    return [w for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:topk]]

def make_wordcloud(text: str):
    return WordCloud(width=800, height=400, collocations=False).generate(text)