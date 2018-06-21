'''
====== Tidene Extract Features ==========
Version: 30-03-2018 - Grupo Pesquisa
Anaconda Python 3
'''
import operator
import nltk
import gensim
from sklearn.metrics.pairwise import cosine_similarity
from scipy import spatial
from math import *

class TideneSimilarity(object):
	
	def __init__(self,traindata,traintarget,testdata,testtarget):
		self.traindata = traindata
		self.traintarget = traintarget
		self.testdata = testdata
		self.testtarget = testtarget
	
	def square_rooted(self,x):
		return round(sqrt(sum([a*a for a in x])),3)
		
	def spatial_cosine(self,vec1,vec2):
		sim = 1 - spatial.distance.cosine(vec1,vec2)
		return(sim)
	
	def top_sim(self,top=1,matrix=None,vectorizer=None):  # dftrain = lista
		''' 
		calcula os top similares para cada doc do teste -- 
		lista [doctestId,listaDocsSimilares[docid,similaridade]]
		'''
		mat_sim = []
		for l,text1 in enumerate(self.testdata):
			aux = []
			for c,text2 in enumerate(self.traindata):
				aux.append([c,cosine_similarity(matrix[c],vectorizer.transform([text1]))[0][0]])
			mat_sim.append([l,sorted(aux,key=operator.itemgetter(1),reverse=True)[:top]])
		return(mat_sim)
	
	def w2v_top_sim(self,vecs,top=1):
		''' 
		calcula os top similares para cada doc do teste -- 
		lista [doctestId,listaDocsSimilares[docid,similaridade]]
		'''
		mat_sim = []

		vecs1 = vecs
        		
		for indexi,itemi in enumerate(vecs1):
			aux = []
			for indexj,itemj in enumerate(vecs):
				if not(indexi==indexj):
					aux.append([indexj,self.spatial_cosine(itemi,itemj)])
			mat_sim.append([indexi,sorted(aux,key=operator.itemgetter(1),reverse=True)[:top]])
		return(mat_sim)

	
	
	
	def print(self,mat_sim):
		print(" ============ sim results ============== ")
		print(" Quantidade de documentos teste ", len(self.testdata))
		for docid,simresults in mat_sim:
			print("===============================")
			print(" novo documento (teste) ",self.testtarget[docid]," ",self.testdata[docid][:100])
			aux = []
			print(" ------------ docs mais semelhantes da base ------------")
			for resid,r in simresults:
				print(r," ",self.traintarget[resid]," ", self.traindata[resid][:100])
	

		
	

	
# literatura
#https://janav.wordpress.com/2013/10/27/tf-idf-and-cosine-similarity/
	
