

import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
import numpy as np
import re
from TideneReadCorpus import *
import xml.etree.ElementTree as ET
from TideneSaves import TideneSaves

PATH = "../../base-wipo/zip_all/"
PREPROCESS_PATH = "../../base-wipo/preprocess-artigo/"


classes = ["A","B","C","D","E","F","G","H"]

from itertools import combinations

subsets = []

for subset in combinations(classes, 2):
    subsets.append(subset)
    try:
        os.makedirs(PREPROCESS_PATH+str(subset[0]+subset[1]))
    except:
        pass


TRAIN_SET_PATH = PREPROCESS_PATH + "treinamento.csv"
TEST_SET_PATH = PREPROCESS_PATH + "teste.csv"


encoding="utf-8"
stopwords = set(stopwords.words('english'))

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')

validation = [ [sub[0],sub[1]] for sub in subsets]
def func_validation(classe):
    global validation
    resu = [ sub for sub in validation if classe in sub ]
    for sub in validation:
        if classe == sub[0]:
            sub[0] = 0
        elif classe == sub[1]:
            sub[1] = 0
    print(resu)
    return resu

def patentesDirX2Csv_300_words(corpusdir,csvfile):
    global subsets, classes, validation


    for subset in subsets:
        if "Test" in corpusdir:
            csv_name = "/teste.csv"
        else:
            csv_name = "/treinamento.csv"
        csvfile_new = csvfile + str(subset[0]+subset[1]) + csv_name

        # only text field - 300 first words
        print("=========== transforming xml corpus files to a csv - on disk =========")
        fd_class = open(csvfile_new,'w')
        # Group number = A01B 1/00 = A01B00100 = subgroup
        # Section = A  Clas = A01 SubClas = A01B main group = A01B001
        #section = subgroup[0]     clas = subgroup[0:3]     subclas = subgroup[0:4] maingroup = subgroup[0:7]
        writer = csv.DictWriter(fd_class, fieldnames=['subgroup','maingroup','subclas','clas','section','othersipcs','data'])
        writer.writeheader()
        fd_class.close

    for sub in classes:
        csvfile_new = csvfile + str(sub) + csv_name
        print(csvfile_new)
        # only text field - 300 first words
        print("=========== transforming xml corpus files to a csv - on disk =========")
        fd = open(csvfile_new,'w')
        # Group number = A01B 1/00 = A01B00100 = subgroup
        # Section = A  Clas = A01 SubClas = A01B main group = A01B001
        #section = subgroup[0]     clas = subgroup[0:3]     subclas = subgroup[0:4] maingroup = subgroup[0:7]
        writer = csv.DictWriter(fd, fieldnames=['subgroup','maingroup','subclas','clas','section','othersipcs','data'])
        writer.writeheader()
        for dirName, subdirList, fileList in os.walk(corpusdir+sub):
            print(dirName)
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
                    tokens = [words for words in tokens[0:300] if words not in stopwords]

                    newstream = ' '.join(tokens)
                    section = mainIPC[0]
                    clas = mainIPC[0:3]
                    subclas = mainIPC[0:4]
                    maingroup = mainIPC[0:7]
                    writer.writerow({'subgroup': mainIPC, 'maingroup':maingroup,'subclas':subclas,'clas':clas,'section':section,'othersipcs':othersIPC, 'data': newstream})
                except:
                    print("erro ao carregar")

            #copia para subsets
            fd.close



            subsets = func_validation(sub)




            for subset in subsets:
                print(subset)
                fd_in = open(csvfile_new,'r')
                reader = csv.DictReader(fd_in, delimiter=',')
                fout = open(csvfile + str(subset[0]) + str(subset[1]) + csv_name)

                writer = csv.DictWriter(fout, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL,fieldnames=reader.fieldnames)
                for row in reader:
                    writer.writerows(row)
                fout.close
                fd_in.close











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
                    newstream = [[w for w in words if w not in stopwords] for words in tokens[0:300]]
                    newstream = ' '.join(newstream)
                    section = mainIPC[0]
                    clas = mainIPC[0:3]
                    subclas = mainIPC[0:4]
                    maingroup = mainIPC[0:7]
                    writer.writerow({'subgroup': mainIPC, 'maingroup':maingroup,'subclas':subclas,'clas':clas,'section':section,'othersipcs':othersIPC, 'data': newstream})
                except:
                    print("parse error file ",p)
        fd.close




def main():

    patentesDirX2Csv_300_words(PATH+"Testzip/",PREPROCESS_PATH)
    '''
    patentsDirXML2CsvFileMultiplesIPCS_300wordsText(PATH+"Testzip",TEST_SET_PATH,tokenizer)
    patentsDirXML2CsvFileMultiplesIPCS_300wordsText(PATH+"Trainzip",TRAIN_SET_PATH,tokenizer)
    '''

if __name__ == "__main__":
    main()
