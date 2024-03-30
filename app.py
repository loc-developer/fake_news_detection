import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import time
import requests
import joblib
import pickle
from transformer import PadTransformer

# Download necessary resources
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("News Data Collection")

# Collect news link
news_link = st.text_input("Enter the news link:")

# Collect news title
news_title = st.text_input("Enter the news title:")

# Select algorithm
algorithm = st.selectbox("Select Algorithm", ["Base GNN", "Wide GNN", "Deep GNN"])

# Preprocess the news title
stop_words = set(stopwords.words("english"))
tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()
tokenized_title = tokenizer.tokenize(news_title.lower())
filtered_title = [lemmatizer.lemmatize(word) for word in tokenized_title if word not in stop_words]
preprocessed_title = ' '.join(filtered_title)

if st.button("Submit"):
    st.write("Predicting...")
    time.sleep(3)  # Simulate prediction time
    res = requests.get(news_link)
    if res.status_code != 200:
        st.write("Site Cannot be accessed")
    else:
        # Vectorization
        vectorizer = PadTransformer(length=6965)
        news_vector = vectorizer.fit_transform([preprocessed_title])
        pred = model.predict(news_vector)[0]

        st.write("Predicted Result for", news_title)
        st.write("Algorithm Selected:", algorithm)
        if pred == 1:
            st.write('\nDetected News Type: Real News')
        else:
            st.write('\nDetected News Type: Fake News')

    