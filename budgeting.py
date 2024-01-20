import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


data = {
    'Month': list(range(1, 13)),
    'Income': [5000, 5200, 5500, 4800, 5100, 5300, 5600, 4900, 5200, 5500, 5800, 5100],
    'Groceries': [300, 350, 320, 310, 300, 330, 340, 310, 300, 320, 330, 310],
    'Utilities': [150, 160, 140, 130, 150, 170, 160, 140, 150, 160, 170, 150],
    'Rent': [1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200, 1200],
    'Entertainment': [200, 180, 220, 190, 210, 200, 220, 210, 230, 240, 220, 200],
    'Savings': [1000, 1100, 1200, 1000, 1050, 1100, 1200, 1000, 1050, 1100, 1200, 1000]
}

df = pd.DataFrame(data)


X = df[['Month', 'Income', 'Groceries', 'Utilities', 'Rent']]
y = df['Savings']

model = LinearRegression()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)


predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')


plt.scatter(X_test['Month'], y_test, color='black', label='Actual Savings')
plt.plot(X_test['Month'], predictions, color='blue', linewidth=3, label='Predicted Savings')
plt.xlabel('Month')
plt.ylabel('Savings')
plt.legend()
plt.show()