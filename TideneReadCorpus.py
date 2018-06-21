import csv
import re
import nltk
from nltk.corpus import stopwords




class TideneIterCSVW2V(object):
	def __init__(self,csvfiles):

		for csvfile in csvfiles:
			csv.field_size_limit(10**9)
			self.reader = csv.reader(open(csvfile,"r"),delimiter=",", quoting=csv.QUOTE_MINIMAL)
			self.reader.__next__()

			apaga = csv.reader(open(csvfile,"r"),delimiter=",", quoting=csv.QUOTE_MINIMAL)
			apaga.__next__()
			self.totalsents = (len(list(apaga)))

	def __iter__(self):
		for index,row in enumerate(self.reader):
			print("Progress:", (index+1), "/", self.totalsents)
			yield re.sub("[^a-zA-Z]", " ", row[6].lower()) #['data']


class TideneIterCSVCorpus(object):
	def __init__(self,csvfiles):
		self.stopwords = set(stopwords.words('english'))

		self.tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

		
		for csvfile in csvfiles:
			csv.field_size_limit(10**9)
			self.reader = csv.reader(open(csvfile,"r"),delimiter=",", quoting=csv.QUOTE_MINIMAL)
			self.reader.__next__()

			apaga = csv.reader(open(csvfile,"r"),delimiter=",", quoting=csv.QUOTE_MINIMAL)
			apaga.__next__()
			self.totalsents = (len(list(apaga)))

	def __iter__(self):
		index = 1
		for index,row in enumerate(self.reader):
			print("Progress:", (index+1), "/", self.totalsents)
			review_text = re.sub("[^a-zA-Z]", " ", row[6].lower())
			review_text = [w for w in self.tokenizer.tokenize(review_text) if w not in self.stopwords]
			row[6] = str1 = ' '.join(review_text)
			index += 1

			

			yield row #['data']

