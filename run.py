from flask import Flask, jsonify, request, render_template
import json
import numpy as np
import pandas as pd
import pickle

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class Text_Transformer(object):

    def __init__(self):
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('stopwords')

    def fit(self, X, y=None):
        df = pd.DataFrame()
        df['Abstract'] = X.Abstract.map(self.preprocess_text)
        return df.Abstract        
        
    def transform(self, X, y=None):
        return X.map(self.preprocess_text)
    
    def fit_transform(self, X, y=None):
        return self.fit(X)

    def preprocess_text(self, text):
        # Convert to lower case
        text = text.lower()

        # Remove punctuation and special characters/digits
        # \W matches any non-word character, \d matches digits
        text = re.sub(r'[\W\d]', ' ', text)

        # Remove single letter words (looking for spaces before and after the letter)
        text = re.sub(r'\b[a-z]\b', '', text)

        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
    
        # Tokenization
        tokens = nltk.word_tokenize(text)
    
        # Stopwords Removal and Lemmatization
        lemmatizer = WordNetLemmatizer()
        cleaned_text = ' '.join([lemmatizer.lemmatize(word) for word in tokens if word not in stopwords.words('english')])
    
        return cleaned_text
    

with open("literature_screening_model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def index():
    pred = ""
    if request.method == "POST":
        abstract = request.form["abstract_text"]
        abstract = pd.Series([abstract])
        pred = model.predict_proba(abstract)[0][1]

        print("output: ", model.predict(abstract))
        
        # age = 78 # request.form["age"]
        # year = 94 # request.form["year"]
        # num_Axi = 12 # request.form["num_Axi"]
        # X = np.array([[float(age), float(year), float(num_Axi)]])
        # pred = model.predict_proba(X)[1][0] 

    return render_template("index.html", pred=pred)


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)

# https://www.w3schools.com/tags/tryit.asp?filename=tryhtml_input_size
