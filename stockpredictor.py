import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingRegressor
import yfinance as yf
from datetime import date

# Fetch historical stock data using yfinance
ticker = input("Enter ticker name in caps: ")  # Example: Apple Inc.
start_date = input("enter start date in yyyy-mm-dd format: ")
end_date = date.today()
stock_data = yf.download(ticker, start=start_date, end=end_date, interval='1m')

# Extracting the 'Open', 'High', 'Low', 'Close' prices and volume
open_prices = stock_data['Open'].values.reshape(-1, 1)
high_prices = stock_data['High'].values.reshape(-1, 1)
low_prices = stock_data['Low'].values.reshape(-1, 1)
volume = stock_data['Volume'].values.reshape(-1, 1)

# Normalize the data
scaler = MinMaxScaler()
open_prices_scaled = scaler.fit_transform(open_prices)
high_prices_scaled = scaler.fit_transform(high_prices)
low_prices_scaled = scaler.fit_transform(low_prices)
volume_scaled = scaler.fit_transform(volume)
close_prices_scaled = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))

# Create input features and labels
X = np.hstack((open_prices_scaled[:-1], high_prices_scaled[:-1], low_prices_scaled[:-1], close_prices_scaled[:-1], volume_scaled[:-1]))
y = close_prices_scaled[1:]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=42)

# Build a Gradient Boosting Regressor model with L2 regularization
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.2, max_depth=6, loss='ls', random_state=42)

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

# Plot the results
plt.figure(figsize=(12,6))
plt.plot(y_test_actual, label='Actual Prices')
plt.plot(y_pred_actual, label='Predicted Prices')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.title('Actual vs Predicted Stock Prices: '+f'R-squared score on test data: {score}')
plt.legend()
plt.grid(True)
plt.show()
