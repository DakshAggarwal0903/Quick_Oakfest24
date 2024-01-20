import pandas as pd
import newsapi
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import requests

def get_latest_news(ticker_input):
    # Initialize the NewsAPI client
    api = newsapi.NewsApiClient(api_key='fcecf50fcd844427b98a50db33f5ed42')

    
    query = ticker_input
    language = 'en'
    sort_by = 'publishedAt'
    page_size = 10

    # Get the latest news articles related to the ticker
    articles = api.get_everything(q=query, language=language, sort_by=sort_by, page_size=page_size)

    # Extract the first article and return it as a Pandas Series
    article = articles['articles'][0]
    return article['title']

headline = get_latest_news(ticker_input="GOOGL")

nltk_stopwords = set(stopwords.words('english'))
sid = SentimentIntensityAnalyzer()
words = word_tokenize(headline.lower())
sentiment_score = sid.polarity_scores(headline)['compound']

print(sentiment_score)
