
from TideneVectorizers import *
from TideneReadCorpus import *
import multiprocessing
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
import numpy as np
import re

from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score,precision_score,recall_score
from gensim.models import KeyedVectors
import os
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from TfidfEmbeddingVectorizer import *
from MeanEmbeddingVectorizer import *
from nltk.corpus import stopwords
import nltk
import gensim.models.word2vec as w2v
from Word2VecVectorizer import *



csv.field_size_limit(10**9)
PATH = "../../base-wipo/base-total-300"
TRAIN_SET_PATH = PATH +  "/preprocess_stop/treinamento.csv"
TEST_SET_PATH = PATH + "/preprocess_stop/teste.csv"


encoding="utf-8"
stopwords = set(stopwords.words('english'))

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')



def main():

    ########################################### PART 1 #################################


    #Load csv files


    X_train = TideneIterCSVClass(TRAIN_SET_PATH)
    X_test = TideneIterCSVClass(TEST_SET_PATH)
    Y_train = pd.read_csv(os.path.join(os.path.dirname(__file__),TRAIN_SET_PATH),
                        header=0,delimiter=";",usecols=["section"], quoting=3)

    print( " ------------------------------------------------ ")
    import gensim.downloader as api

    model = api.load("glove-wiki-gigaword-100")

    #print(model.most_similar("mobile"))

    w2v = dict(zip(model.wv.index2word, model.wv.syn0))

    w2v_vectorizer =  Word2VecVectorizer(model)

    X_train_w2v = w2v_vectorizer.fit_transform(X_train,len(Y_train))

    print(X_train_w2v.shape)








if __name__ == "__main__":
    main()
