
from TideneVectorizers import *
from TideneReadCorpus import *
from Patente import Patente
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



csv.field_size_limit(10**9)
PATH = "../../base-wipo/"
TRAIN_SET_PATH = "../../base-wipo/treinamento.csv"
TEST_SET_PATH = "../../base-wipo/treinamento.csv"


encoding="utf-8"
stopwords = set(stopwords.words('english'))

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')





def preprocessDataset(reader):
    X, y = [], []
    #For objects in patentes list
    for index,row in reader.iterrows():

        print("Progress:", (index+1), "/", len(list(reader.iterrows())))

        review_text = re.sub("[^a-zA-Z]", " ", row["data"].lower())
        review_text = [w for w in tokenizer.tokenize(review_text) if w not in stopwords]
        X.append(review_text)
        y.append(row["section"])

    return np.array(X), np.array(y)




def main():




    ########################################### PART 2 #################################

    train = pd.read_csv(os.path.join(os.path.dirname(__file__),TRAIN_SET_PATH),
                        delimiter=",", quoting=csv.QUOTE_MINIMAL,header=0)

    test = pd.read_csv(os.path.join(os.path.dirname(__file__),TEST_SET_PATH),
                        delimiter=",", quoting=csv.QUOTE_MINIMAL,header=0)
    
    X,y = preprocessDataset(train)

    X_test,y_test = preprocessDataset(test)
    


if __name__=="__main__":
    main()