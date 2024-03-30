from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin
import numpy as np

class PadTransformer(TransformerMixin):
    def __init__(self, length):
        self.length = length

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        padded_X = []
        for x in X:
            # Get TF-IDF vector
            vectorizer = TfidfVectorizer(max_features=self.length)
            tfidf_vector = vectorizer.fit_transform([x]).toarray()[0]

            # Pad the vector if its length is less than self.length
            if len(tfidf_vector) < self.length:
                padded_vector = np.pad(tfidf_vector, (0, self.length - len(tfidf_vector)), 'constant')
                padded_X.append(padded_vector)
            else:
                padded_X.append(tfidf_vector[:self.length])

        return np.array(padded_X)