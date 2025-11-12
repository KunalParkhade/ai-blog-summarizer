from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def analyze(text: str):
    analyzer = SentimentIntensityAnalyzer()
    return analyzer.polarity_scores(text)