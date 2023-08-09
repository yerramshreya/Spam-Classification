from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk

import pandas as pd
import re

cp = pd.read_csv('Corpus')
nltk.download('stopwords')



lemm = WordNetLemmatizer()
sw = stopwords.words('english')
corpus = []
for i in range(len(cp.Corpus)):
    corpus.append(cp.Corpus.loc[i])

cv = CountVectorizer(max_features=40000)
corpus = cv.fit_transform(corpus).toarray()
tfidf = TfidfTransformer()
corpus = tfidf.fit_transform(corpus).toarray()
