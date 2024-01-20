import openpyxl
import newsapi
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Set up NLTK
nltk_stopwords = set(stopwords.words('english'))
sid = SentimentIntensityAnalyzer()

# Set up newsapi client
api_key = 'fcecf50fcd844427b98a50db33f5ed42'
newsapi.NewsApiClient(api_key)

# Set up yfinance to download stock data
ticker = str(input("give ticker to trace - "))
stock_data = yf.download(ticker, start='2023-12-20', end=datetime.now(), interval='1h')

# Get the current price of the stock
last_price = stock_data['Close'].iloc[-1]

# Set up data structures to store stock and headline data
headline_data = []

# Get news headlines for the past year
news_api = newsapi.NewsApiClient(api_key)
query = ticker
articles = news_api.get_everything(q=query, language='en', sort_by='relevancy', from_param='2023-12-20')
headlines = [article['title'] for article in articles['articles']]

# Iterate through the headlines and find the hourly change in the stock price after each headline
for headline in headlines:
    for i in range(len(stock_data)-1, -1, -1):
        if (stock_data.index[i].replace(tzinfo=None) > ((datetime.now() - timedelta(hours=24))).replace(tzinfo=None)):
            # If the headline was published in the last 24 hours, find the hourly change
            last_headline_price = stock_data.iloc[i]['Close']
            hourly_change = (stock_data.iloc[i]['Close'] - last_price) / last_price
            headline_data.append({'Headline': headline, 'Hourly Change': hourly_change})
            last_price = stock_data.iloc[i]['Close']

            words = word_tokenize(headline.lower())
            filtered_words = [word for word in words if word.isalnum() and word not in nltk_stopwords]
            freq_dist = FreqDist(filtered_words)
            important_words = [word for word in filtered_words if freq_dist[word] > 1]
            importance_score = len(important_words) / len(filtered_words)
            sentiment_score = sid.polarity_scores(headline)['compound']

            # Add the scores to the headline data
            headline_data[-1]['Importance Score'] = importance_score
            headline_data[-1]['Sentiment Score'] = sentiment_score

# Create dataset
data = [{'Date': stock_data.index[i], 'Open': stock_data.iloc[i]['Open'], 'High': stock_data.iloc[i]['High'],
         'Low': stock_data.iloc[i]['Low'], 'Close': stock_data.iloc[i]['Close'], 'Adj Close': stock_data.iloc[i]['Adj Close'], 'Diff': (stock_data.iloc[i]['Close']-stock_data.iloc[i]['Open']),
         'Volume': stock_data.iloc[i]['Volume']} for i in range(len(stock_data))]
df = pd.DataFrame(data)

# Add the headline data to the dataframe
headline_df = pd.DataFrame(headline_data)
df = pd.concat([df, headline_df], axis=1)

print(df.head(20))

# Set the filename
filename = ('output_'+ticker+'_stock.json')

# Use the to_json function to export the dataframe
###df.to_json(filename, orient='records')
###temporarily down for testing

### getting latest news

import requests

def get_latest_news(ticker_input):
    query_v2 = ticker_input
    language_v2 = 'en'
    sort_by_v2 = 'publishedAt'
    page_size_v2 = 10

    # Get the latest news articles related to the ticker
    articles_v2 = news_api.get_everything(q=query_v2, language=language_v2, sort_by=sort_by_v2, page_size=page_size_v2)

    # Extract the first article and return it as a Pandas Series
    article_v2 = articles_v2['articles'][0]
    return article_v2['title']

headline_v2 = get_latest_news(ticker_input=ticker)

words_v2 = word_tokenize(headline_v2.lower())
sentiment_score_v2 = sid.polarity_scores(headline_v2)['compound']

print(headline_v2 + str(sentiment_score_v2))