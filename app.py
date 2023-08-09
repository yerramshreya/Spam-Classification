import pickle
import pandas as pd
import streamlit as st
from collections import Counter
from itertools import chain
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

import pandas as pd
import re

cp = pd.read_csv('Corpus')


lemm = WordNetLemmatizer()
sw = stopwords.words('english')
corpus = []
for i in range(len(cp.Corpus)):
    corpus.append(cp.Corpus.loc[i])

cv = CountVectorizer(max_features=40000)
corpus = cv.fit_transform(corpus).toarray()
tfidf = TfidfTransformer()
corpus = tfidf.fit_transform(corpus).toarray()




def preprocess(input):
    scaler = StandardScaler()
    cp = []
    if len(cp) > 0:
        cp  = []
    cp.append(input)
    tmp_vec = cv.transform(cp)
    tf_vec = tfidf.transform(tmp_vec)
    return scaler.fit_transform(tf_vec.toarray())


# Naive Bayes Model
model_1 = pickle.load(open("Naive-B",'rb'))
# Neural Network Model
model_2 = load_model("Neural-nets.h5")
# SVM Model
model_3 = pickle.load(open("SVM-model.txt",'rb'))

st.set_page_config(layout='wide',page_title='Spam Classification📩 by Areddy Sathvik')

st.image('titleimg.jpg')
st.title("Lets Classify The Spam and Ham Messages by One-Click")
st.write('---')

st.write('## Enter Email / Text you want to check')
inp = st.text_input(label='',max_chars=200,)


if st.button('Check'):
    w = preprocess(inp)
    pr_1 = model_1.predict(w)
    pr_2 = model_2.predict(w)
    pr_3 = model_3.predict(w)
    tmp = []
    tmp.append(pr_1)
    tmp.append(pr_3)
    fp = list(chain.from_iterable(tmp))
    pr_2_ = 1 if pr_2 > 0.5 else 0
    fp.append(pr_2_)
    pr = Counter(fp).most_common()[0][0]
    op_val = ((pr_2 + pr_3 + pr_1)/3) * 100
    data_plot = pd.DataFrame()
    data_plot['Spam'] = [op_val]
    data_plot['Not Spam'] = [100-op_val]

    if op_val >= 75:
        st.error('SPAM')
    else:
        st.success('HAM')
    st.write('---')
    st.write("### Exact Probability of the message being spam is" )
    st.write(op_val)

st.write('---')
st.write("### would you like to have a chat with me✍️👇")
st.write('areddysathvik@gmail.com')

    
