from fastapi import FastAPI
import numpy

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

from nltk.corpus import stopwords
from pydantic import BaseModel
import string

import pickle
import uvicorn

app = FastAPI()

class Response(BaseModel):
    text: str
    result: bool


def process_text(text: str) -> list[str]:

    # Check string to see if they are a punctuation
    nopunc = [char for char in text if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Convert string to lowercase and remove stopwords
    clean_string = [word for word in nopunc.split() if word.lower() not in stopwords.words('indonesian')]
    return clean_string

def bow_transtrorm(text_clean: str):
    vectorizer = CountVectorizer().fit(process_text(text_clean))
    text_bow = vectorizer.transform([text_clean])
    return text_bow

def tfidf_transformer(text_bow):
    tfidf_transformer = TfidfTransformer().fit(text_bow)
    text_tfidf = tfidf_transformer.transform(text_bow)
    return text_tfidf

model = pickle.load(open('model.pkl', 'rb'))
vec = pickle.load(open('vec.pkl', 'rb'))

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.post("/predict", response_model=Response)
def predict(text: str):
    x_features = vec.transform([text])
    y_pred: numpy.ndarray = model.predict(x_features)
    print(y_pred)
    response = Response(text=text, result=y_pred.tolist()[0])
    return response


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
