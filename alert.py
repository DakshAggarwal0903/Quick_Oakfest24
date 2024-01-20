import time
import yfinance as yf

class FinancialAssistant:
    def __init__(self, user_preferences):
        self.user_preferences = user_preferences

    def monitor_stock_prices(self, symbols):
        while True:
            try:
                for symbol in symbols:
                    stock_data = yf.Ticker(symbol)
                    
                   
                    if len(stock_data.history(period='1d')['Close']) >= 2:
                        current_price = stock_data.history(period='1d')['Close'].iloc[-1]
                        previous_price = stock_data.history(period='1d')['Close'].iloc[-2]
                        percentage_change = ((current_price - previous_price) / previous_price) * 100

                        print(f"Current price of {symbol}: {current_price}")
                        print(f"Percentage change for {symbol}: {percentage_change:.2f}%")

                        if percentage_change > self.user_preferences.get('threshold_increase'):
                            self.send_alert("Stock Price Alert", f"{symbol} has experienced a substantial increase of {percentage_change:.2f}%!")

                        elif percentage_change < self.user_preferences.get('threshold_decrease'):
                            self.send_alert("Stock Price Alert", f"{symbol} has experienced a substantial decrease of {percentage_change:.2f}%!")

            except Exception as e:
                print(f"Error: {e}")

            time.sleep(10)

    def send_alert(self, alert_type, message):
        print(f"{alert_type}: {message}")


user_preferences = {
    'threshold_increase': 2,
    'threshold_decrease': -2,
}


assistant = FinancialAssistant(user_preferences)

print("Starting to monitor the stocks")
assistant.monitor_stock_prices(['AAPL', 'GOOGL', 'LMT'])