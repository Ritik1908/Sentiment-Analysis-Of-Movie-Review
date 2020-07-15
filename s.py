from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

word_list = pickle.load(open('reviews.pkl', 'rb'));

clf = pickle.load(open('model.pkl', 'rb'));


@app.route('/')
def home():
    return render_template("index.html");


@app.route("/predict", methods=['POST'])
def predict():
    review = request.form.get("review");
    return review;


if __name__ == "__main__":
    app.run()
