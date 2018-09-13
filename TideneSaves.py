'''
====== Tidene Saves ==========
Version: 22-08-2017 - Grupo Pesquisa
Anaconda Python 3
'''

import os
import csv
import unicodedata
import xml.etree.ElementTree as ET

class TideneSaves(object):
# save csv files

	def norm(text):
		try:
			textNorm = unicodedata.normalize('NFKD', text).encode('ASCII','ignore').decode('UTF-8')
		except ValueError:
			textNorm = unicodedata.normalize('NFKD',text).encode('UTF-8','ignore').decode()
		return(textNorm)


	def df2CsvFile(df,corpuscsvfile):
		print("=============== saving dataframe on a csv file ========= ")
		## transform data field on a text stream
		joinlst = []
		for d in df.data:
			if (type(d) is list):
				temp = ' '.join(d)
			else:
				temp = d
			joinlst.append(temp)
		df.data = joinlst
		df.to_csv(corpuscsvfile, index=False)

	def patentsDir2CsvFile(corpusdir,csvfile,tokenizer):
		print("=========== transforming corpus files to a csv - on disk =========")
		fd = open(csvfile,'w')
		writer = csv.DictWriter(fd, fieldnames=['target','data','ipcl1','ipcl2','ipcl3'])
		writer.writeheader()
		for dirName, subdirList, fileList in os.walk(corpusdir):
			for fname in fileList:
				cat = dirName.split("/")[-1]
				if (cat == ""):
					p = dirName + fname
					cat = fname.split(".")[0]
				else:
					p = dirName+"/"+fname
				try: # Reading utf-8 file
					stream = open(p, encoding="UTF-8").read().replace("\n"," ").lower()
				except ValueError:	# if error Read as ISO-8859-15 file
					stream = open(p, encoding="ISO-8859-15").read().replace("\n"," ").lower()
				tokens = tokenizer.tokenize(Saves.__norm(stream).strip())
				if (len(tokens) < 2):
					print (" bad text formation ", tokens)
				else:
					newstream = ' '.join(tokens)
					ipcl1 = cat[0]
					try:
						ipcl2 = cat[1:3]
					except:
						ipcl2 = ""
					try:
						ipcl3 = cat[3]
					except:
						ipcl3 = ""
					writer.writerow({'target': cat, 'IPCl1':ipcl1, 'IPCl2':ipcl2, 'IPCl3':ipcl3, 'data': newstream})
		fd.close

	def patentsDirXML2CsvFile(corpusdir,csvfile,tokenizer):
		print("=========== transforming xml corpus files to a csv - on disk =========")
		fd = open(csvfile,'w')
		writer = csv.DictWriter(fd, fieldnames=['target','data','ipcl1','ipcl2','ipcl3','ipcl4'])
		writer.writeheader()
		for dirName, subdirList, fileList in os.walk(corpusdir):
			for fname in fileList:
				cat = dirName.split("/")[-4:]
				if (cat == ""):
					p = dirName + fname
					cat = fname.split(".")[0]
				else:
					p = dirName+"/"+fname

				try:
					tree = ET.parse(p)
					root = tree.getroot()
					print(" Getting data from ", p)
					# title + abst + claim
					title = root.find('tis').find('ti').text
					t = root.find('abs').find('ab').text
					if t:
						abstract = t
					else:
						abstract = ' '

					claim = root.find('cls').find('cl').text
					stream =  title + abstract + claim
					tokens = tokenizer.tokenize(Saves.__norm(stream.lower()).strip())
					temp = ''.join(cat) #cat[0]+cat[1]+cat[2]+cat[3]
					newstream = ' '.join(tokens)
					writer.writerow({'target': temp, 'ipcl1':cat[0], 'ipcl2':cat[1], 'ipcl3':cat[2], 'ipcl4':cat[3],  'data': newstream})
				except:
					print("parse error file ",p)
		fd.close
