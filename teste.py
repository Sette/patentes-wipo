
from Patente import Patente

import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
import numpy as np
import re


PATH = "../../base-wipo/"
TRAIN_SET_PATH = "../../base-wipo/treinamento.csv"
TEST_SET_PATH = "../../base-wipo/treinamento.csv"


encoding="utf-8"
stopwords = set(stopwords.words('english'))

tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')



def loadDataset(patentes):
    X, y = [], []
    #For objects in patentes list
    for patente in patentes:
        print(patente.data)
        break
        review_text = re.sub("[^a-zA-Z]", " ", patente.data.lower())
        review_text = [w for w in tokenizer.tokenize(review_text) if w not in stopwords]
        X.append(review_text)
        y.append(patente.clas)

    return np.array(X), np.array(y)


def load():

    #Open csv files with pandas dataframe
    test = pd.read_csv(os.path.join(os.path.dirname(__file__), PATH+'teste.csv'),
                        header=0, delimiter=",", quoting=3)
    train = pd.read_csv(os.path.join(os.path.dirname(__file__), PATH+'treinamento.csv'),
                        header=0, delimiter=",", quoting=3)



    #Get elements using iterrows method for dataframe
    elementos_test = [item for index,row in test.iterrows() for item in row]
    elementos_train = [item for index,row in train.iterrows() for item in row]

    

    #Convert to Patente object
    patentes_test = [Patente(elemento) for elemento in elementos_test]
    patentes_train = [Patente(elemento) for elemento in elementos_train]

    #Clear memmory
    del(test)
    del(train)
    del(elementos_test)
    del(elementos_train)


    return patentes_train,patentes_test

def main():
    
    #Load csv files
    patentes_train,patentes_test = load()

    #Convert to np array
    X_train,y_train = loadDataset(patentes_train)
    del(patentes_train)

    print(X_train)



if __name__ == "__main__":
    main()