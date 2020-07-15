from flask import Flask, redirect, request, render_template
import pickle
import nltk
from nltk.corpus import stopwords # For Stopwords
from nltk.stem.porter import PorterStemmer # For stemming words
import re
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer # For Vectorizing words


words = pickle.load(open('models/words', 'rb'));
models = pickle.load(open('models/model', 'rb'))

app = Flask(__name__)

def join_back(list_input):
    return " ".join(list_input)


def process_text(text):
    clean = re.compile('<.*?>')
    text = re.sub(clean, '', text)
    text = text.lower()


    x = ""
    for i in text:
        if i.isalnum():
            x = x + i
        else:
            x = x + " "
    text = x

    nltk.download('stopwords')
    x = []
    for i in text.split():
        if i not in stopwords.words('english'):
            x.append(i)
    text = x[:]

    ps = PorterStemmer()
    y = []
    for i in text:
        y.append(ps.stem(i))
    text = join_back(y[:])
    return text

@app.route('/')
def home():
   return render_template("index.html")

@app.route("/predict", methods=['POST'])
def predict():
    review_original = request.form.get("review")
    review_processed = [process_text(review_original)]
    # review_processed = [review_original]
    review_processed = words.transform(review_processed)
    ans = str(models.predict(review_processed)[0])
    # return ans
    return render_template("predict.html", original_review = review_original, result = ans)

if __name__ == '__main__':
   app.run(debug=True)
