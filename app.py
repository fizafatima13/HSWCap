from flask import Flask, render_template, request
import pickle


app = Flask(__name__)


# Load saved AI files
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))



@app.route("/")
def home():
    return render_template("index.html")



@app.route("/predict", methods=["POST"])
def predict():

    tweet = request.form["tweet"]

    # Convert text into numbers
    tweet_vector = vectorizer.transform([tweet])

    # AI prediction
    prediction = model.predict(tweet_vector)

    # Convert number back to label
    result = encoder.inverse_transform(prediction)[0]


    return render_template(
        "index.html",
        prediction=result
    )



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
