import numpy as np
import random
from TideneReadCorpus import *
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import os


PATH = "/home/bruno/base-wipo/preprocess-AB-min/preprocess_token/"
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

paramgrid = {"kernel": ["rbf"],
             "C"     : np.logspace(-9, 9, num=5, base=10),
             "gamma" : np.logspace(-9, 9, num=5, base=10)}

random.seed(1)

from evolutionary_search import EvolutionaryAlgorithmSearchCV
cv = EvolutionaryAlgorithmSearchCV(estimator=SVC(),
                                   params=paramgrid,
                                   scoring="accuracy",
                                   cv=StratifiedKFold(n_splits=4),
                                   verbose=1,
                                   population_size=10,
                                   gene_mutation_prob=0.05,
                                   gene_crossover_prob=0.25,
                                   tournament_size=2,
                                   generations_number=3,
                                   n_jobs=2)

cv.fit(tfidf_transformer.fit_transform(X), y['section'].tolist())
