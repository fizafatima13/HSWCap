# ======================================
# Hate Speech Detection using Decision Tree
# ======================================

# Import Libraries
import numpy as np
import pandas as pd
import re
import string
import nltk

# Google Drive (Only for Google Colab)
from google.colab import drive
drive.mount('/content/drive')

# Download NLTK data
nltk.download('stopwords')

from nltk.corpus import stopwords

# Machine Learning Libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# --------------------------------------
# Load Dataset
# --------------------------------------

# Change this path according to your file location
df = pd.read_csv('twitter_data.csv')

print("First 5 Rows:")
print(df.head())

# --------------------------------------
# Select Required Columns
# --------------------------------------

df = df[['tweet', 'class']]

# --------------------------------------
# Convert Numbers to Labels
# --------------------------------------

df['labels'] = df['class'].map({
    0: "Hate Speech",
    1: "Offensive Language",
    2: "Neither"
})

# --------------------------------------
# Text Cleaning
# --------------------------------------

stop_words = set(stopwords.words("english"))

def clean(text):
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove punctuation
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)

    # Remove new lines
    text = re.sub(r'\n', '', text)

    # Remove numbers
    text = re.sub(r'\w*\d\w*', '', text)

    # Remove stopwords
    words = [word for word in text.split() if word not in stop_words]

    return " ".join(words)

df["tweet"] = df["tweet"].apply(clean)

print("\nCleaned Data:")
print(df.head())

# --------------------------------------
# Convert Labels to Numbers
# --------------------------------------

encoder = LabelEncoder()

df['labels'] = encoder.fit_transform(df['labels'])

# --------------------------------------
# Features and Labels
# --------------------------------------

X = df['tweet']
Y = df['labels']

# --------------------------------------
# Convert Text into Numbers
# --------------------------------------

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(X)

# --------------------------------------
# Split Dataset
# --------------------------------------

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.33,
    random_state=42
)

# --------------------------------------
# Train Decision Tree Model
# --------------------------------------

model = DecisionTreeClassifier()

model.fit(X_train, Y_train)

# --------------------------------------
# Test Model
# --------------------------------------

predictions = model.predict(X_test)

print("\nAccuracy:")
print(accuracy_score(Y_test, predictions))

print("\nClassification Report:")
print(classification_report(Y_test, predictions))

# --------------------------------------
# Predict Your Own Tweet
# --------------------------------------

sample = input("\nEnter a tweet: ")

sample = clean(sample)

sample_vector = vectorizer.transform([sample])

prediction = model.predict(sample_vector)

print("\nPrediction:", encoder.inverse_transform(prediction)[0])
import pickle

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))
pickle.dump(encoder, open("encoder.pkl", "wb"))

print("Model saved successfully")