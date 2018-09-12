import numpy as np
import random
from TideneReadCorpus import *
import pandas as pd
import os


PATH = "../../base-wipo/preprocess_lemm_stemm/"
treinamento = "treinamento.csv"
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold


y = pd.read_csv(os.path.join(os.path.dirname(__file__),PATH+treinamento),
                    header=0,delimiter=";",usecols=["section"], quoting=3)
X = pd.read_csv(os.path.join(os.path.dirname(__file__),PATH+treinamento),
                    header=0,delimiter=";", quoting=3)

n = len(y)

paramgrid = {"kernel": ["rbf"],
             "C"     : np.logspace(-9, 9, num=25, base=10),
             "gamma" : np.logspace(-9, 9, num=25, base=10)}

random.seed(1)

from evolutionary_search import EvolutionaryAlgorithmSearchCV
cv = EvolutionaryAlgorithmSearchCV(estimator=SVC(),
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
cv.fit(X, y)
