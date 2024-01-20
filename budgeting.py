import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from nltk import download

# Download necessary NLTK resources
download('stopwords')
download('wordnet')

# Load and preprocess data
data = pd.read_json(r"C:\Users\daksh\OneDrive\Desktop\Folders\Hackathons\Codefest'24\training_data.json")

# Data Preprocessing
data['user_name'] = data['user'].apply(lambda x: x['name'].lower())
data['monthly_income'] = data['user'].apply(lambda x: x['monthly_income'])
data['savings_account_balance'] = data['user'].apply(lambda x: x['savings_account_balance'])

# Flatten 'transactions' field and update the original dataframe
transactions_df = pd.json_normalize(data['transactions']).add_prefix('transaction_')
data = pd.concat([data, transactions_df], axis=1)

# Feature Engineering
data['transaction_month'] = pd.to_datetime(data['transaction_time']).dt.month
data['day_of_week'] = pd.to_datetime(data['transaction_time']).dt.dayofweek

# Text Preprocessing using NLTK
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text).lower())
    words = word_tokenize(text)
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return ' '.join(words)

data['preprocessed_description'] = data['description'].apply(preprocess_text)
print("Preprocessing done.")
# Labeling
data['overspending'] = data['amount'] < 0  # Binary label: True if overspending, False otherwise

# Split data
X = data[['transaction_month', 'day_of_week', 'preprocessed_description']]
y = data['overspending']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a pipeline with a TF-IDF Vectorizer and RandomForestClassifier
model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, input_=X.columns[2])),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train model
model.fit(X_train, y_train)
print("Training Done.")
# Evaluate model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Done.")