import numpy as np


class Word2VecVectorizer():
    def __init__(self,word_vectors):
        self.word_vectors = word_vectors

    def fit(self, lst_tokens):
      pass

    #para cada sentenca tokenizada ele representa cada palavra segundo a
    #representacao w2v e depois tira a media de todas as palavras da sentenca
    def transform(self, lst_tokens, len_token): #pega uma lista de tokens

        self.D = self.word_vectors.get_vector(self.word_vectors.index2word[0]).shape[0]
        X = np.zeros(len_token, self.D)
        n = 0
        emptycount = 0
        for tokens in lst_tokens:
            vecs = []
            m = 0
            for word in tokens:
                try:
                    vec = self.word_vectors.get_vector(word)
                    vecs.append(vec)
                    m += 1
                except KeyError:
                    print('Palavra ',word,' nao pode ser representada')
                    pass
            if len(vecs) > 0:
                vecs = np.array(vecs)
                X[n] = vecs.mean(axis=0)
            else:
                emptycount += 1
                n += 1
        return X


    def fit_transform(self, lst_token, len_tokens):
        self.fit(lst_tokens)
        return self.transform(lst_tokens, len_tokens)
