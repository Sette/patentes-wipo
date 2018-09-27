
from TideneVectorizers import *
from TideneReadCorpus import *
import multiprocessing
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
import numpy as np
import re
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
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
import gensim.models.word2vec as w2v



csv.field_size_limit(10**9)
PATH = "../../base-wipo/"
TRAIN_SET_PATH = "../../base-wipo/preprocess-AB-min/treinamento.csv"
TEST_SET_PATH = "../../base-wipo/preprocess-AB-min/teste.csv"


encoding="utf-8"
stopwords = set(stopwords.words('english'))

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')



def main():

    ########################################### PART 1 #################################


    #Load csv files
    PATH = PATH = "/home/bruno/base-wipo/preprocess-AB-min/preprocess_token/treinamento.csv"

    #itercsvW2V = TideneIterCSVW2V([TRAIN_SET_PATH,TEST_SET_PATH])
    X_train = pd.read_csv(os.path.join(os.path.dirname(__file__),PATH),
                        header=0,delimiter=";",usecols=["data"], quoting=3)["data"].tolist()
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

    X_train = [[w for w in tokenizer.tokenize(x) \
    if w not in stopwords and len(w) > 3] \
    for x in X_train]
    print(len(X_train[2]))




    # Configura valores para o word2vec
    num_features = 100  # Word vector dimensionality
    min_word_count = 40  # Minimum word count
    num_workers = 4  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words
    model_name = "100features_40minwords_10context"

    #Verifica se o modelo existe
    try:
        model = gensim.models.Word2Vec.load(model_name)
    except:
        print("Gerou o modelo")
        model = w2v.Word2Vec(X_train,workers=num_workers, \
                    size=num_features, min_count=min_word_count, \
                    window=context, sample=downsampling, seed=1)
        #model.build_vocab(X_train)
        #model.train(X_train)
        model.save(model_name)






if __name__ == "__main__":
    main()
