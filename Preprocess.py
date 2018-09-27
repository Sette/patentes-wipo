

import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
import numpy as np
import re
from TideneReadCorpus import *

PATH = "../../base-wipo/preprocess-AB-min/"
PREPROCESS_PATH = "../../base-wipo/preprocess-AB-min/preprocess_token/"


encoding="utf-8"
stopwords = set(stopwords.words('english'))

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')


def load_CSV(train_or_test):
	if train_or_test == "treinamento":
		train_or_test = "treinamento.csv"
	else:
		train_or_test = "teste.csv"

	data = TideneIterCSVCorpus(PATH+train_or_test)

	#Write csv file
	with open(PREPROCESS_PATH+train_or_test, 'w') as csvfile_out:
		spamwriter = csv.writer(csvfile_out, delimiter=';',quotechar='|', quoting=csv.QUOTE_MINIMAL)
		spamwriter.writerow(["subgroup","maingroup","subclas","clas","section","othersipcs","data"])
		for patente in data:
	   		spamwriter.writerow(patente)


	return True



def main():

    #Load csv files
    load_CSV("teste")
    load_CSV("treinamento")




if __name__ == "__main__":
    main()
