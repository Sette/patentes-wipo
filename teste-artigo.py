import pandas as pd
import statistics
import numpy as np
import random
import gensim
from TideneReadCorpus import *
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import os
from sklearn.svm import LinearSVC
from MeanEmbeddingVectorizer import *
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import re
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from gensim.models.word2vec import Word2Vec
import struct
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from gensim.models import KeyedVectors
import os
from sklearn.model_selection import train_test_split
from TfidfEmbeddingVectorizer import *
from MeanEmbeddingVectorizer import *
from nltk.corpus import stopwords
import nltk
from nltk.metrics import *
import gensim.models.word2vec as w2v
from TideneReadCorpus import *
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC

PATH = "/home/sette/base-wipo/preprocess-min/preprocess_stop/"
teste = "teste.csv"
treinamento = "treinamento.csv"

def train(elem):
    classe = elem[0]
    c = elem[1]
    sections = ["A","B","C","D","E","F","G","H"]
    #X_test = TideneIterCSVClass(PATH+teste)
    #X_train = TideneIterCSVClass(PATH+treinamento)

    Y_test = pd.read_csv(os.path.join(os.path.dirname(__file__),PATH+teste),
                        header=0,delimiter=";",usecols=["section"], quoting=3)
    Y_train = pd.read_csv(os.path.join(os.path.dirname(__file__),PATH+treinamento),
                        header=0,delimiter=";",usecols=["section"], quoting=3)


    '''
    #Estatistica
    sections = ["A","B","C","D","E","F","G","H"]
    print("Conjunto de treinamento ...")
    result = [(x, Y_train['section'].tolist().count(x)) for x in sections]
    print(result)

    print("Conjunto de teste ...")
    result = [(x, Y_test['section'].tolist().count(x)) for x in sections]
    print(result)


    model_name = "100features_40minwords_10context"
    try:
        model = gensim.models.Word2Vec.load(model_name)
    except:
        print("Modelo não encontrado, por favor gere o modelo w2v")
        return

    w2v = dict(zip(model.wv.index2word, model.wv.syn0))
    '''

    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import confusion_matrix

    tfidf_transformer = TfidfVectorizer()

    #------------ LinearSVC test ---------------------
    X_train = TideneIterCSVClass(PATH+treinamento)
    X_test = TideneIterCSVClass(PATH+teste)
    clf = LinearSVC(C=c).fit(tfidf_transformer.fit_transform(X_train), Y_train['section'].tolist())

    predict = clf.predict(tfidf_transformer.transform(X_test))

    print(classe + " com acurácia: "+ str(accuracy(Y_test['section'].tolist(),predict)))

    cm = confusion_matrix(Y_test['section'].tolist(), predict, labels = sections)
    print(cm)
    return




def main():
    data = pd.read_csv("saida-artigo.csv",delimiter=";")


    train(['Geral',0.03162277660168379])

    '''
    for elem in data.values:
        train(list(elem))
    return
    '''


    teste = list(data.c.values)

    teste = [float(elemento) for elemento in teste]

    print("Moda -------------------------")
    print(statistics.mode(teste))

    print("Mediana -----------------------------")
    print(statistics.median(teste))
    print("Variancia ------------------------------")

    print(statistics.variance(teste))
    print("Desvio padrão ---------------------------")
    print(statistics.stdev(teste))

    print("Média ---------------------------")
    print(statistics.mean(teste))

    print(data)




if __name__ == "__main__":
    main()
