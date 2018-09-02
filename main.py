
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
import gensim
import numpy as np
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


    
#PATH = "../../base-wipo/preprocess/"
PATH = "../../base-wipo/preprocess_lemm_stemm/"
teste = "teste.csv"
treinamento = "treinamento.csv"



def main():


    X_test = TideneIterCSVClass(PATH+teste)
    X_train = TideneIterCSVClass(PATH+treinamento)

    Y_test = pd.read_csv(os.path.join(os.path.dirname(__file__),PATH+teste),
                        header=0,delimiter=";",usecols=["section"], quoting=3)
    Y_train = pd.read_csv(os.path.join(os.path.dirname(__file__),PATH+treinamento),
                        header=0,delimiter=";",usecols=["section"], quoting=3)
    
    #Estatistica
    sections = ["A","B","C","D","E","F","G","H"]
    print("Conjunto de treinamento ...")
    result = [(x, Y_train['section'].tolist().count(x)) for x in sections]
    print(result)
    
    print("Conjunto de teste ...")
    result = [(x, Y_test['section'].tolist().count(x)) for x in sections]
    print(result)

    '''
    train = pd.read_csv(os.path.join(os.path.dirname(__file__),'labeledTrainData.tsv'),
                        header=0,delimiter="\t", quoting=3)

    X_w2v, y_w2v = loadDataset_Review(train)

    train, test = train_test_split(train, train_size=0.7, random_state=42)

    X,y = loadDataset_Review(train)

    X_test,y_test = loadDataset_Review(test)

    '''
    # train word2vec on all the texts - both training and test set
    # we're not using test labels, just texts so this is fine
    #model = Word2Vec(X, size=100, window=5, min_count=5, workers=2)

    # Configura valores para o word2vec
    num_features = 100  # Word vector dimensionality
    min_word_count = 40  # Minimum word count
    num_workers = 4  # Number of threads to run in parallel
    context = 10  # Context window size
    downsampling = 1e-3  # Downsample setting for frequent words
    model_name = "100features_40minwords_10context"
    try:
        model = gensim.models.Word2Vec.load(model_name)
    except:
        print("Modelo não encontrado, por favor gere o modelo w2v")
        return
        

    w2v = dict(zip(model.wv.index2word, model.wv.syn0))

    
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.metrics import confusion_matrix
    
    tfidf_transformer = TfidfVectorizer()

    #clf = MultinomialNB().fit(tfidf_transformer.fit_transform(X_train), Y_train)

    #predict = clf.predict(tfidf_transformer.transform(X_test))

    #print(accuracy(Y_test['section'].tolist(),predict))  

    #m = confusion_matrix(Y_test['section'].tolist(), predict, labels = sections)
    #print(cm)

    # start with the classics - naive bayes of the multinomial and bernoulli varieties
    # with either pure counts or tfidf features
    mult_nb = Pipeline(
        [("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
    bern_nb = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
    mult_nb_tfidf = Pipeline(
        [("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("multinomial nb", MultinomialNB())])
    bern_nb_tfidf = Pipeline(
        [("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("bernoulli nb", BernoulliNB())])
    # SVM - which is supposed to be more or less state of the art
    # http://www.cs.cornell.edu/people/tj/publications/joachims_98a.pdf
    svc = Pipeline([("count_vectorizer", CountVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])
    svc_tfidf = Pipeline(
        [("tfidf_vectorizer", TfidfVectorizer(analyzer=lambda x: x)), ("linear svc", SVC(kernel="linear"))])

    # Extra Trees classifier is almost universally great, let's stack it with our embeddings

    etree_w2v = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
                          ("extra trees", ExtraTreesClassifier(n_estimators=200))])
    etree_w2v_tfidf = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
                                ("extra trees", ExtraTreesClassifier(n_estimators=200))])


    #Random forest classifier with our embeddings
    random_w2v_tfidf = Pipeline([("word2vec vectorizer", TfidfEmbeddingVectorizer(w2v)),
                                ("extra trees", RandomForestClassifier(n_estimators=200))])
    random_w2v = Pipeline([("word2vec vectorizer", MeanEmbeddingVectorizer(w2v)),
                          ("extra trees", RandomForestClassifier(n_estimators=200))])


    all_models = [
        ("mult_nb_tfidf", mult_nb_tfidf),
        ("bern_nb_tfidf", bern_nb_tfidf),
        ("svc_tfidf", svc_tfidf),
        ("w2v_tfidf", etree_w2v_tfidf),
        ("random_w2v_tfidf", random_w2v_tfidf),
    ]


    '''


    all_models = [
        ("mult_nb", mult_nb),
        ("mult_nb_tfidf", mult_nb_tfidf),
        ("bern_nb", bern_nb),
        ("bern_nb_tfidf", bern_nb_tfidf),
        ("svc", svc),
        ("svc_tfidf", svc_tfidf),
        ("w2v", etree_w2v),
        ("w2v_tfidf", etree_w2v_tfidf),
        ("random_w2v", random_w2v),
        ("random_w2v_tfidf", random_w2v_tfidf),

    ]
    
    '''
    
    unsorted_scores = []
    for name, model in all_models:
        print("Training with ", name)
        clf = model.fit(X_train,Y_train['section'].tolist())
        predict = clf.predict(X_test)
        unsorted_scores.append((name,\
            accuracy(Y_test['section'].tolist(),predict),\
            f_measure(set(Y_test['section'].tolist()),set(predict)),\
            precision(set(Y_test['section'].tolist()),set(predict)),\
            recall(set(Y_test['section'].tolist()),set(predict))))



    scores = sorted(unsorted_scores, key=lambda x: -x[1])

    print(tabulate(scores, floatfmt=".4f", headers=("model", 'Accuracy','F1','Precision','Recall')))

if __name__ == "__main__":
    main()
