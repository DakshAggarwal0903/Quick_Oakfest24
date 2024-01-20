#setup
import openpyxl
import newsapi
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Set up NLTK for NLP
nltk_stopwords = set(stopwords.words('english'))
sid = SentimentIntensityAnalyzer()

#i hate commenting
api_key = 'fcecf50fcd844427b98a50db33f5ed42'
newsapi.NewsApiClient(api_key)

#yfinance up
ticker = str(input("give ticker to trace - "))
stock_data = yf.download(ticker, start='2023-12-20', end=datetime.now(), interval='1h')

#stock price
last_price = stock_data['Close'].iloc[-1]

#data
headline_data = []

#give me ur headlines
news_api = newsapi.NewsApiClient(api_key)
query = ticker
articles = news_api.get_everything(q=query, language='en', sort_by='relevancy', from_param='2023-12-20')
headlines = [article['title'] for article in articles['articles']]

#iteration my goat
for headline in headlines:
    for i in range(len(stock_data)-1, -1, -1):
        if (stock_data.index[i].replace(tzinfo=None) > ((datetime.now() - timedelta(hours=24))).replace(tzinfo=None)):
            last_headline_price = stock_data.iloc[i]['Close']
            hourly_change = (stock_data.iloc[i]['Close'] - last_price) / last_price
            headline_data.append({'Headline': headline, 'Hourly Change': hourly_change})
            last_price = stock_data.iloc[i]['Close']
            #nlp scoring code uwuww aughagahhaga
            words = word_tokenize(headline.lower())
            filtered_words = [word for word in words if word.isalnum() and word not in nltk_stopwords]
            freq_dist = FreqDist(filtered_words)
            important_words = [word for word in filtered_words if freq_dist[word] > 1]
            importance_score = len(important_words) / len(filtered_words)
            sentiment_score = sid.polarity_scores(headline)['compound']

            #scores for headlines
            headline_data[-1]['Importance Score'] = importance_score #doesnt work ever
            headline_data[-1]['Sentiment Score'] = sentiment_score

#dataframesetthingy
data = [{'Date': stock_data.index[i], 'Open': stock_data.iloc[i]['Open'], 'High': stock_data.iloc[i]['High'],
         'Low': stock_data.iloc[i]['Low'], 'Close': stock_data.iloc[i]['Close'], 'Adj Close': stock_data.iloc[i]['Adj Close'],
         'Volume': stock_data.iloc[i]['Volume']} for i in range(len(stock_data))]
df = pd.DataFrame(data)

#become one
headline_df = pd.DataFrame(headline_data)
df = pd.concat([df, headline_df], axis=1)

print(df.head(20))

#foolname
filename = ('output_'+ticker+'_stock.json')

#excport
df.to_json(filename, orient='records')
