from flask import Flask, render_template, request, jsonify
import pickle
import re
import string
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords

app = Flask(__name__)

# --------------------------------------
# Load saved model files (once, when server starts)
# --------------------------------------
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('encoder.pkl', 'rb') as f:
    encoder = pickle.load(f)

stop_words = set(stopwords.words("english"))

def clean(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    words = [word for word in text.split() if word not in stop_words]
    return " ".join(words)

# --------------------------------------
# Route 1: Serves the HTML page itself
# --------------------------------------
@app.route('/')
def home():
    return render_template('index.html')

# --------------------------------------
# Route 2: Receives text from the page, returns a prediction
# --------------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')

    if text.strip() == '':
        return jsonify({'error': 'No text provided'}), 400

    cleaned = clean(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)
    label = encoder.inverse_transform(prediction)[0]

    return jsonify({'label': label})

# --------------------------------------
# Run the server
# --------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
