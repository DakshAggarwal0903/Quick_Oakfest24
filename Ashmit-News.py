import openpyxl
import newsapi
import yfinance as yf
import pandas as pd
import datetime as datetime, timedelta
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Set up NLTK for NLP. Sentiment used for values later on
nltk_stopwords = set(stopwords.words('english'))
sid = SentimentIntensityAnalyzer()

# Set up newsapi
api_key = 'fcecf50fcd844427b98a50db33f5ed42'
newsapi.NewsApiClient(api_key)

# Set up yfinance to download stock data
ticker = str(input("give ticker to trace - "))
stock_data = yf.download(ticker, start='2023-12-20', end=datetime.now(), interval='1h')

#price setup
last_price = stock_data['Close'].iloc[-1]

#i sleep here is the headline data array
headline_data = []

# Get news headlines also newsapi cringe as hell man why only december 20
news_api = newsapi.NewsApiClient(api_key)
query = ticker
articles = news_api.get_everything(q=query, language='en', sort_by='relevancy', from_param='2023-12-20')
headlines = [article['title'] for article in articles['articles']]
