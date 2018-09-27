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

rand_st = 42

PATH = "/home/bruno/base-wipo/preprocess/preprocess_token/"
treinamento = "treinamento.csv"
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold


y = pd.read_csv(os.path.join(os.path.dirname(__file__),PATH+treinamento),
                    header=0,delimiter=";",usecols=["section"], quoting=3)

'''
X = pd.read_csv(os.path.join(os.path.dirname(__file__),PATH+treinamento),
                    header=0,delimiter=";",usecols=["data"], quoting=3)

X = X["data"].tolist()
'''

X = TideneIterCSVClass(PATH+treinamento)

tfidf_transformer = TfidfVectorizer()

n = len(y)


random.seed(1)
from evolutionary_search import EvolutionaryAlgorithmSearchCV
'''
#--------------------------------- GA-RF ---------------------------------
from evolutionary_search import EvolutionaryAlgorithmSearchCV

clf_RF_gs = RandomForestClassifier(random_state=rand_st, n_jobs=-1)
clf_RF_pg = [{
    'max_depth': np.logspace(0.3,4,num = 10 ,base=10,dtype='int'), #[1, 5, 13, 34, 87, 226, 584, 1505, 3880, 10000]
    'n_estimators' : np.logspace(0.1,3,num = 10 ,base=10,dtype='int'), #[1, 2, 5, 11, 24, 51, 107, 226, 476, 1000]
    'min_samples_split' : np.logspace(0.4, 1, num=5, base=10, dtype='int'), #[2, 3, 5, 7, 10]
    'min_samples_leaf' : np.logspace(0.1,1,num = 4 ,base=9,dtype='int'), #[1, 2, 4, 9]
    'max_features' : ['auto', None]
              }]


model_name = "100features_40minwords_10context"


model = gensim.models.Word2Vec.load(model_name)


w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}



cv = EvolutionaryAlgorithmSearchCV(estimator=clf_RF_gs,
                                   params=clf_RF_pg,
                                   scoring="accuracy",
                                   cv=StratifiedKFold(n_splits=4),
                                   verbose=1,
                                   population_size=10,
                                   gene_mutation_prob=0.05,
                                   gene_crossover_prob=0.25,
                                   tournament_size=2,
                                   generations_number=3,
                                   n_jobs=2)


#print(model.wv.most_similar('sensitive'))
cv.fit(tfidf_transformer.fit_transform(X), y['section'].tolist())
#cv.fit(MeanEmbeddingVectorizer(w2v).transform(X), y['section'].tolist())


'''


#--------------------------------- GA-SVC ---------------------------------

paramgrid = {"C"  : np.logspace(-9, 9, num=25, base=10)}

cv = EvolutionaryAlgorithmSearchCV(estimator=LinearSVC(),
                                   params=paramgrid,
                                   scoring="accuracy",
                                   cv=StratifiedKFold(n_splits=4),
                                   verbose=1,
                                   population_size=50,
                                   gene_mutation_prob=0.10,
                                   gene_crossover_prob=0.5,
                                   tournament_size=3,
                                   generations_number=5,
                                   n_jobs=4)

cv.fit(tfidf_transformer.fit_transform(X), y['section'].tolist())
