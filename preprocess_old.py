
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
import csv



csv.field_size_limit(10**9)
PATH = "../../base-wipo/"
TRAIN_SET_PATH = "../../base-wipo/treinamento.csv"
TEST_SET_PATH = "../../base-wipo/teste.csv"


encoding="utf-8"




def main():


	for train in [TEST_SET_PATH,TRAIN_SET_PATH]:
		X  = TideneIterCSVCorpus([train])
		out = "preprocess/"
		if "teste" in train:
			out+= "teste.csv"
		else:
			out+= "treinamento.csv"

		with open(out,'a') as data:
			writer = csv.writer(data)
			for x in X:
				writer.writerow(x)

if __name__=="__main__":
    main()