import math
import pickle
from scipy import linalg
import numpy
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD

'''

'''
class VectorModel:

    def __init__(self, **kwargs):
        return super().__init__(**kwargs)

    '''
    ritorna l'indice doc-terms in versione tabellare (lista di liste)
    '''
    def get_doc_index_table(self, doc_index):
        return list(doc_index.values())

    '''
    @:author Davide, Domenico
    @:param tfidf the TF-IDF weights dictionary
    @:return a dictionary where the key is an arbitrary value that indicate the document, and the value is a row
                containing the weights of each word within the document.
    '''
    def get_doc_index(self, tfidf):
        index = {}

        words = tfidf.keys()
        count=0
        words_pos = {}
        for word in words:
            words_pos[word] = count
            count+=1

        for word in tfidf:
            for tuple in tfidf[word]:
                id_doc = tuple[0]
                weight = tuple[1]
                if id_doc in index:
                    pos = words_pos[word]
                    index[id_doc][pos] = weight
                else:
                    index[id_doc] =  [0] * len(words)

        return index

    '''
    @:author Davide, Domenico
    @:param sigma_values valori indici delle dimensioni DA ELIMINARE
    @:param tfidf in versione tabellare!
    '''
    def LSA(self, tfidf, ncomponents):
        svd = TruncatedSVD(n_components=ncomponents, random_state=42)
        reduced = svd.fit_transform(tfidf)
        return reduced

    '''
    @:author Davide
    @:param pmi the PMI dictionary

    Plot the PMI values.
    '''
    def plot_pmi(self, pmi):
        pmi_values = [p[1] for p in pmi]
        plt.plot(pmi_values)
        plt.ylabel('PMI values')
        plt.show()

    '''
    @:author Davide
    Calcola i vettori dei pesi per il dataset. Struttura del dizionario:

    {
        ...
        'word': [(ID_Doc,Weight), ...],
        ...
    }
    '''
    def get_tfidf(self, phrases, frequencies=None):
        dict = {}

        # calcola le frequenze delle parole nel dataset
        if not frequencies:
            frequencies = self.get_words_frequencies(phrases, None, True)

        # calcola il numero di documenti
        D = len(phrases)

        for word in frequencies:
            # prendo la lista dei documenti in cui la parola è presente
            docs_containing_word = frequencies[word]
            idf_denom = len(docs_containing_word.keys())
            dict[word] = []
            for doc in docs_containing_word:
                # frase contenente la parola

                try:
                    the_phrase = [ph[1] for ph in phrases if ph[0] == doc][0]
                except:
                    print('doc: ' + doc + ' word: ' + word)
                    print(phrases)
                # numero di termini nel documento corrente
                doc_size = len(set(the_phrase.split()))
                # numero di occorrenze del termine i nel documento j
                nij = frequencies[word][doc]
                tfidf = (nij / doc_size) * math.log(D / idf_denom)
                dict[word].append((doc, tfidf))
        return dict

    '''
    @:author Davide
    Calcola il dizionario delle frequenze delle singole parole nelle frasi date in ingresso. Se frequencies è fornito,
    vengono aggiornate sue frequenze
    '''
    def get_words_frequencies(self, phrases, frequencies = None, lowerKeys = False):
        if frequencies:
            dict = frequencies.copy()
        else:
            dict = {}

        for phrase in phrases:
            id = phrase[0]
            the_phrase = phrase[1]
            tokens = [t for t in the_phrase.split() if len(t) > 1]
            for word in tokens:
                if lowerKeys:
                    word = word.lower()
                if word in dict:
                    if not id in dict[word]:
                        dict[word][id] = 1
                    else:
                        dict[word][id] += 1
                else:
                    dict[word] = {id: 1}
        return dict

    '''
    @:author Davide, Domenico
    '''
    def deserialize_tfidf(self, fname):
        with open(fname, 'rb') as pkl_file:
            data = pickle.load(pkl_file)
        return data

    '''
    @:author Davide, Domenico
    '''
    def persist_tfidf(self, tfidf, fname):
        with open(fname, 'wb') as output:
            pickle.dump(tfidf, output)