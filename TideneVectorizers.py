'''
====== Tidene Extract Features ==========
=== operations: tokenizing,stopword removal,stemming
Version: 30-03-2018 - Grupo Pesquisa
Anaconda Python 3
'''
import nltk
import gensim
import pandas as pd
import numpy as np
from math import *
from sklearn.decomposition import PCA
from matplotlib import pyplot
from sklearn.feature_extraction.text import *
from TideneSimilarity import *

class TidenePrints(object):
	
	def __init__(self):
		super(TidenePrints, self).__init__()

	def print_matrix(self,matrix): #matrix,vectorizer=None,target=None,w2v=None):
			print(" Shape da matriz - docs x palavras/termos")
			(num_docs,num_atributos) = matrix.shape
			print(" numero de documentos da base ", num_docs)
			print(" numero de atributos ", num_atributos)
			print("============= parte do vocabulario obtido pelo vetorizador ============================= ")
			print("  O DICIONÁRIO TODO ")
			print(self.vocabulary_)   # o dicionario todo
			print("   SÓ ALGUMAS CHAVES ")
			print(list(self.vocabulary_.keys())[40:60]) #so chaves
			print("============= parte dos nomes dos atributos (termos encontrados) ============================= ")
			print("     ALGUNS ATRIBUTOS ")
			print(self.get_feature_names()[40:60])
			
			print("============= parte da matriz com classes / vectorizador atributo ============================= ")
			linha_inicio = num_docs-2
			linha_fim = num_docs
			col_atributos_inicio = num_atributos-50
			col_atributos_fim = num_atributos-45
			print(pd.DataFrame(matrix.toarray(),columns=self.get_feature_names()).iloc[linha_inicio:linha_fim,col_atributos_inicio:col_atributos_fim])
		

	def print_matrix_w2v(self,matrix): #matrix,vectorizer=None,target=None,w2v=None):
			# para visualizar o modelo gerado com o word2vec
			#w2v = {w: vec for w, vec in zip(matrix.wv.index2word, matrix.wv.syn0)}
			print("============= Matriz Word2Vec ============================= ")
			print(matrix.shape)
			
			
			X = self.w2v_model[self.w2v_model.wv.vocab]
			pca = PCA(n_components=2)
			result = pca.fit_transform(X)
			pyplot.scatter(result[:, 0], result[:, 1])
			words = list(self.w2v_model.wv.vocab)
			for i, word in enumerate(words):
				pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
			pyplot.show()
			
			#print(self.w2v_model[self.w2v_model.wv.vocab][8])
			
			#print(self.w2v_model.wv.vocab['accompani'])
			
			#print(self.vocabulary_)   # o dicionario todo
			#print(list(self.vocabulary_.keys())[40:60]) #so chaves
			
			#print(self.get_feature_names()[40:60])
			
			#print(matrix)
				
			#print(self.w2v_model.wv.index2word)
			#print(self.w2v_model.wv.vocab['accompani'].index)
			#print(self.w2v_model.wv.index)
			#print(self.w2v_model.wv.vocab)
			#print(self.w2v_model.wv.syn0)
			#w2v = {w: vec for w, vec in matrix}
			#w2v = {w: vec for w, vec in zip(self.w2v_model.wv.index2word, self.w2v_model.wv.syn0)}
			#print(w2v)
			# (8, 129)	0.0426920786811

class TideneW2VVectorizer(TidenePrints):
    # word2vec reading http://adventuresinmachinelearning.com/word2vec-tutorial-tensorflow/
    #https://towardsdatascience.com/word-to-vectors-natural-language-processing-b253dd0b0817
    
	def __init__(self, min_df = None,tokenizer=nltk.tokenize.RegexpTokenizer(r'\w+')):
		self.tokenizer = tokenizer
		super(TideneW2VVectorizer,self).__init__()
	
	def get_feature_names(self):
		return(list(self.w2v_model.wv.vocab.keys()))
		
	def get_most_common_words(self,top_n=3):
		lst = []
		for i in range(top_n):
			lst.append(self.w2v_model.wv.index2word[i])
		return(lst)
		

	def fit(self, X, y=None):
		return self

	
	def fit_transform(self,data):
		self.index2word_set = set(self.w2v_model.index2word)
		return(self.transform(data))
	
	def get_avg_feature_vector(self,doc,w2v_model,num_features,index2word_set): # transform
		words = self.tokenizer(doc)
		feature_vec = np.zeros((num_features, ), dtype='float32')
		n_words = 0
		for word in words:
			if word in index2word_set:
				n_words += 1
				feature_vec = np.add(feature_vec, w2v_model[word])
		if (n_words > 0):
			feature_vec = np.divide(feature_vec, n_words)
		return feature_vec

	def transform(self,data):
		vecs=[]
		for d in data:
			vecs.append(self.get_avg_feature_vector(d, self.w2v_model, 100, self.index2word_set))
		return(vecs)
		
	def convert_in_a_matrix(self): # to use in Keras and tensor flow
		embedding_matrix = np.zeros((len(self.w2v_model.wv.vocab), self.vector_dim))
		for i in range(len(self.w2v_model.wv.vocab)):
			embedding_vector = self.w2v_model.wv[self.w2v_model.wv.index2word[i]]
			if embedding_vector is not None:
				embedding_matrix[i] = embedding_vector
		return(embedding_matrix)
	
	def build_w2v_model(self,data,num_features = 100,min_word_count = 40\
		,num_workers = 4,context = 10,downsampling = 1e-3):

		
	    model_name = "100features_40minwords_10context"

	    #Verifica se o modelo existe
	    try:
	        model = gensim.models.Word2Vec.load(model_name)
	    except:
	        print("Gerou o modelo")
	        model = gensim.models.Word2Vec(data, workers=num_workers, \
	                     size=num_features, min_count=min_word_count, \
	                     window=context, sample=downsampling, seed=1)
	        model.save(model_name)

	    w2v = dict(zip(model.wv.index2word, model.wv.syn0))


	def set_w2v_model_file(self,name):
		self.model_name = name



	def load_w2v_model(self,vector_dim=100,load_from_bin_file="w2v_model.bin"):
		self.vector_dim = vector_dim
		model = gensim.models.KeyedVectors.load_word2vec_format(load_from_bin_file, binary=True)
		self.w2v_model = model

	def print_matrix(self,matrix):
		self.print_matrix_w2v(matrix)



class TideneCountVectorizer(CountVectorizer,TidenePrints):
	def __init__(self,min_df = 1,tokenizer=nltk.tokenize.RegexpTokenizer(r'\w+')):
		super(TideneCountVectorizer,self).__init__(min_df,tokenizer)
		

class TideneTfidfVectorizer(TfidfVectorizer,TidenePrints):
	def __init__(self,min_df = 1,tokenizer=nltk.tokenize.RegexpTokenizer(r'\w+')):
		super(TideneTfidfVectorizer,self).__init__(min_df,tokenizer)
