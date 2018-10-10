import csv
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
import numpy as np


class TideneIterCSVW2V(object):
	def __init__(self,csvfiles):

		for csvfile in csvfiles:
			csv.field_size_limit(10**9)
			self.reader = csv.reader(open(csvfile,"r"),delimiter=";", quoting=csv.QUOTE_MINIMAL)
			self.reader.__next__()

			apaga = csv.reader(open(csvfile,"r"),delimiter=";", quoting=csv.QUOTE_MINIMAL)
			apaga.__next__()
			self.totalsents = (len(list(apaga)))

	def __iter__(self):
		for index,row in enumerate(self.reader):
			#print("Progress:", (index+1), "/", self.totalsents)
			yield row[6].split() #['data']





class TideneIterCSVCorpus(object):
	def __init__(self,csvfile):
		self.stopwords = set(stopwords.words('english'))
		self.tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
		self.porter_stemmer = PorterStemmer()
		self.wordnet_lemmatizer = WordNetLemmatizer()

		csv.field_size_limit(10**9)
		self.reader = csv.reader(open(csvfile,"r"),delimiter=",", quoting=csv.QUOTE_MINIMAL)
		self.reader.__next__()

		apaga = csv.reader(open(csvfile,"r"),delimiter=",", quoting=csv.QUOTE_MINIMAL)
		apaga.__next__()
		self.totalsents = (len(list(apaga)))

	def __iter__(self):
		index = 0
		for index,row in enumerate(self.reader):
			#print("Progress:", (index+1), "/", self.totalsents)
			row[6] = re.sub("[^a-zA-Z]", " ", row[6].lower())
			#row[6] = [self.wordnet_lemmatizer.lemmatize(self.porter_stemmer.stem(w)) \
			#for w in self.tokenizer.tokenize(row[6]) if w not in self.stopwords \
			row[6] = [w for w in self.tokenizer.tokenize(row[6]) if w not in self.stopwords and len(w) > 3]
			row[6] = ' '.join(row[6])

			index += 1

			yield row #['data']




class TideneIterCSVClass(object):
	def __init__(self,csvfile):
		self.tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
		csv.field_size_limit(10**9)
		self.reader = csv.reader(open(csvfile,"r"),delimiter=";", quoting=csv.QUOTE_MINIMAL)
		self.reader.__next__()

		apaga = csv.reader(open(csvfile,"r"),delimiter=";", quoting=csv.QUOTE_MINIMAL)
		apaga.__next__()
		self.totalsents = (len(list(apaga)))

	def __iter__(self):
		index = 0
		for index,row in enumerate(self.reader):
			#print("Progress:", (index+1), "/", self.totalsents)
			index += 1
			yield row[6]  #['data']




















class TideneIterCSVGA(object):
	def __init__(self,csvfile):
		self.tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
		csv.field_size_limit(10**9)
		self.reader = csv.reader(open(csvfile,"r"),delimiter=",", quoting=csv.QUOTE_MINIMAL)
		self.reader.__next__()

		apaga = csv.reader(open(csvfile,"r"),delimiter=",", quoting=csv.QUOTE_MINIMAL)
		apaga.__next__()
		self.totalsents = (len(list(apaga)))

	def __iter__(self):
		index = 0
		for index,row in enumerate(self.reader):
			print("Progress:", (index+1), "/", self.totalsents)
			index += 1
			yield  row[6] #['data']
