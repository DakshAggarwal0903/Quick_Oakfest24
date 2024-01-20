import openpyxl
import newsapi
import yfinance as yf
import pandas as pd
import datetime as datetime, timedelta
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Set up NLTK
nltk_stopwords = set(stopwords.words('english'))
sid = SentimentIntensityAnalyzer()
