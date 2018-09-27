

import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
import numpy as np
import re
from TideneReadCorpus import *
import xml.etree.ElementTree as ET
from TideneSaves import TideneSaves
PATH = "../../base-wipo/zipAB-min/"
PREPROCESS_PATH = "../../base-wipo/preprocess-AB-min/"
TRAIN_SET_PATH = PREPROCESS_PATH + "treinamento.csv"
TEST_SET_PATH = PREPROCESS_PATH + "teste.csv"


encoding="utf-8"
stopwords = set(stopwords.words('english'))

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

def patentsDirXML2CsvFileMultiplesIPCS_300wordsText(corpusdir,csvfile,tokenizer):
        # only text field - 300 first words
        print("=========== transforming xml corpus files to a csv - on disk =========")
        fd = open(csvfile,'w')
        # Group number = A01B 1/00 = A01B00100 = subgroup
        # Section = A  Clas = A01 SubClas = A01B main group = A01B001
        #section = subgroup[0]     clas = subgroup[0:3]     subclas = subgroup[0:4] maingroup = subgroup[0:7]
        writer = csv.DictWriter(fd, fieldnames=['subgroup','maingroup','subclas','clas','section','othersipcs','data'])
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

                    mainIPC = root.find('ipcs').get('mc')
                    #print("MAIN IPC-CODE ", mainIPC)
                    othersIPC = []
                    for ipc in root.iter('ipc'):
                        othersIPC.append(ipc.get('ic'))
                    #print("OTHERS IPC ",othersIPC)
                    if ( len(othersIPC) > 0):
                        othersIPC = '-'.join(othersIPC)
                    else:
                        othersIPC = ' '

                    # title + abst + claim
                    title = root.find('tis').find('ti').text
                    t = root.find('abs').find('ab').text
                    if t:
                        abstract = t
                    else:
                        abstract = ' '
                    #claim = root.find('cls').find('cl').text

                    text = root.find('txts').find('txt').text

                    stream = text # title + abstract + claim

                    tokens = tokenizer.tokenize(TideneSaves.norm(stream.lower()).strip())
                    newstream = ' '.join(tokens[0:300])   # only 300 first words of field text
                    section = mainIPC[0]
                    clas = mainIPC[0:3]
                    subclas = mainIPC[0:4]
                    maingroup = mainIPC[0:7]
                    writer.writerow({'subgroup': mainIPC, 'maingroup':maingroup,'subclas':subclas,'clas':clas,'section':section,'othersipcs':othersIPC, 'data': newstream})
                except:
                    print("parse error file ",p)
        fd.close




def main():
    patentsDirXML2CsvFileMultiplesIPCS_300wordsText(PATH+"Testzip",TEST_SET_PATH,tokenizer)
    patentsDirXML2CsvFileMultiplesIPCS_300wordsText(PATH+"Trainzip",TRAIN_SET_PATH,tokenizer)


if __name__ == "__main__":
    main()
