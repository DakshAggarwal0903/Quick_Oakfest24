lame_i=input("Input ticker please - ")
from colorama import Fore

def stockPredict(ticker2):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.ensemble import GradientBoostingRegressor
    import yfinance as yf
    from datetime import date, timedelta

    # Fetch historical stock data using yfinance
    ticker = ticker2
    start_date = date.today() - timedelta(days=6, hours=23, minutes=30)
    end_date = date.today()
    stock_data = yf.download(ticker, start=start_date, end=end_date, interval='1m')

    # Extracting the 'Open', 'High', 'Low', 'Close' prices and volume
    open_prices = stock_data['Open'].values.reshape(-1, 1)
    high_prices = stock_data['High'].values.reshape(-1, 1)
    low_prices = stock_data['Low'].values.reshape(-1, 1)
    volume = stock_data['Volume'].values.reshape(-1, 1)
    adj_close = stock_data['Adj Close'].values.reshape(-1, 1)

    # Normalize the data
    scaler = MinMaxScaler()
    open_prices_scaled = scaler.fit_transform(open_prices)
    high_prices_scaled = scaler.fit_transform(high_prices)
    low_prices_scaled = scaler.fit_transform(low_prices)
    volume_scaled = scaler.fit_transform(volume)
    adj_prices_scaled = scaler.fit_transform(adj_close)
    close_prices_scaled = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))

    # Create input features and labels
    X = np.hstack((open_prices_scaled[:-1], high_prices_scaled[:-1], low_prices_scaled[:-1], close_prices_scaled[:-1], volume_scaled[:-1], close_prices_scaled[:-1]))
    y = close_prices_scaled[1:]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

    # Build a Gradient Boosting Regressor model with L2 regularization
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.25, max_depth=10, loss='lad', random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    score = model.score(X_test, y_test)
    print(f'R-squared score on test data: {score}')

    # Make predictions on the test data
    y_pred = model.predict(X_test)

    # Inverse transform the scaled predictions and actual values to get actual stock prices
    y_pred_actual = scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    plt.figure(figsize=(12,6))
    plt.plot(y_pred_actual, label='Predicted Prices')
    plt.plot(y_test_actual, label='Actual Prices')
    plt.xlabel('Time')
    plt.ylabel('Stock Price')
    plt.title('Predicted and Actual Stock Prices: '+f'R-squared score on test data: {score}')
    plt.legend()
    plt.grid(True)
    plt.show()



def newsPredict(ticker_i):
    import newsapi
    import yfinance as yf
    import pandas as pd
    from datetime import datetime, timedelta
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.probability import FreqDist
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    # Set up NLTK for NLP. Sentiment used for values later on
    # Set up NLTK for NLP
    nltk_stopwords = set(stopwords.words('english'))
    sid = SentimentIntensityAnalyzer()

    # Set up newsapi
    #i hate commenting
    api_key = 'fcecf50fcd844427b98a50db33f5ed42'
    newsapi.NewsApiClient(api_key)

    # Set up yfinance to download stock data
    #yfinance up
    ticker = ticker_i
    stock_data = yf.download(ticker, start='2023-12-20', end=datetime.now(), interval='1h')

    #price setup
    #stock price
    last_price = stock_data['Close'].iloc[-1]

    #i sleep here is the headline data array
    #data
    headline_data = []

    # Get news headlines also newsapi cringe as hell man why only december 20
    #give me ur headlines
    dupes=0
    news_api = newsapi.NewsApiClient(api_key)
    query = ticker
    articles = news_api.get_everything(q=query, language='en', sort_by='relevancy', from_param='2024-01-10')
    headlines = [article['title'] for article in articles['articles']]
    unique_headlines = set()
    for headline in headlines:
        for i in range(len(stock_data)-1, -1, -1):
            if (stock_data.index[i].replace(tzinfo=None) > ((datetime.now() - timedelta(hours=48))).replace(tzinfo=None)):
                last_headline_price = stock_data.iloc[i]['Close']
                hourly_change = (stock_data.iloc[i]['Close'] - last_price) / last_price
                if headline not in unique_headlines:
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
                    unique_headlines.add(headline)
                else:
                    dupes+=1
                    
    #print(dupes)

    #dataframesetthingy
    data = [{'Date': stock_data.index[i], 'Open': stock_data.iloc[i]['Open'], 'High': stock_data.iloc[i]['High'],
            'Low': stock_data.iloc[i]['Low'], 'Close': stock_data.iloc[i]['Close'], 'Adj Close': stock_data.iloc[i]['Adj Close'],
            'Volume': stock_data.iloc[i]['Volume'], 'Diff': (stock_data.iloc[i]['Close']-stock_data.iloc[i]['Open'])} for i in range(len(stock_data))]
    df = pd.DataFrame(data)



    #become one
    headline_df = pd.DataFrame(headline_data)
    df = pd.concat([df, headline_df], axis=1)

    df= df.dropna()

    #foolname
    filename = ('output_'+ticker+'_stock.json')
    ###
    ###i hate this
    ###
    ###ai
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

    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score


    X = df['Sentiment Score'].values.reshape(-1, 1)
    y = df['Diff'].values.reshape(-1, 1)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    print("R-squared value: {:.2f}".format(r2))
    
    two_d_array = np.array([[sentiment_score_v2]])

    true_pred = model.predict(two_d_array)

    returnable_array = [headline_v2, true_pred, ]
    return true_pred

predicted = newsPredict(ticker_i=lame_i)
print(predicted)

stockPredict(ticker2=lame_i)



