import streamlit as st
import pickle
import string
import nltk
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords.words("english")


def transform_text(text):
    text = text.lower()  # lower case
    text = nltk.word_tokenize(text)  # tokenize means every is a token

    y = []
    for i in text:
        if i.isalnum():  # removing alphanumeric
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if (i not in stopwords.words("english")
                and i not in string.punctuation):  # removing stopwords and applying punctuation
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))  # stemming

    return " ".join(y)


tfidf = pickle.load(open("vectoriizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button("Predict"):

    # 1. Preprocess
    transformed_sms = transform_text(input_sms)

    # 2. Vectorize
    vector_input = tfidf.transform([transformed_sms])

    # 3. Predict
    result = model.predict(vector_input)[0]

    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")