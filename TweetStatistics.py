import math
import pickle

class Statistics:

    def __init__(self, **kwargs):
        return super().__init__(**kwargs)

    '''
    @:author Davide
    Calculate the PMI for each phrase in the input list
    '''
    def get_pmi_for_list(self, phrases, sort=True):
        dicts = self.get_words_and_bigrams_count_from_list(phrases)
        wcount = dicts[0]
        bcount = dicts[1]
        return self.__calculate_PMI(wcount, bcount, sort)

    '''
    @:author Davide
    Calculate the PMI for the input phrase
    '''
    def get_pmi_for_phrase(self, phrase, sort=True):
        dicts = self.get_words_and_bigrams_count_from_list([phrase])
        wcount = dicts[0]
        bcount = dicts[1]
        return self.__calculate_PMI(wcount, bcount, sort)

    '''
    @:author Davide
    Calculate the PMI for all the phrases in the specified file
    '''
    def get_pmi_for_file(self, fname, sort=True):
        dicts = self.get_words_and_bigrams_count_from_file(fname)
        wcount = dicts[0]
        bcount = dicts[1]
        return self.__calculate_PMI(wcount, bcount, sort)

    '''
    @:author davide
    @:return a list of elements of the following format:

        [(word_i,word_k), PMI_ik]

    That is, each element within the list contains two words and their PMI value.
    @:param sort if True, sort the output list by the PMI values in descending order.
    '''
    def __calculate_PMI(self, words_dict, bigrams_dict, sort=False):
        pmi = []
        for bigram in bigrams_dict.keys():
            # get the bigram frequency (the numerator)
            bigram_frequency = bigrams_dict[bigram]
            try:
                # calculate the pmi
                the_pmi = math.log(bigram_frequency / (words_dict[bigram[0]] + words_dict[bigram[1]]))
            except:
                # if something goes wrong in previous calculation, proceed to next bigram
                continue
            # append
            pmi.append([bigram, the_pmi])

        if sort:
            pmi.sort(key=lambda x: x[1], reverse=True)
        return pmi

    '''
    @:author davide
    @:return a list [W,B] where W is the words frequencies dictionary, while B is the bigrams frequencies dictionary.

    Read the given file line by line and calculate the words and bigrams frequencies
    '''
    def get_words_and_bigrams_count_from_file(self, text_fname):
        words = {}
        bigrams = {}
        with open(text_fname) as f:
            for line in f:
                dicts = self.__get_words_and_bigrams_count(line)
                words = {**words, **dicts[0]}
                bigrams = {**bigrams, **dicts[1]}
        return [words, bigrams]

    '''
    @:author davide
    @:return a list [W,B] where W is the words frequencies dictionary, while B is the bigrams frequencies dictionary.

    Calculate the words/bigrams frequencies for all phrases in the input list
    '''
    def get_words_and_bigrams_count_from_list(self, phrases_list):
        words = {}
        bigrams = {}
        for phrase in phrases_list:
            dicts = self.__get_words_and_bigrams_count(phrase)
            words = {**words, **dicts[0]}
            bigrams = {**bigrams, **dicts[1]}
        return [words, bigrams]

    '''
    @:author davide
    Calculate words and bigrams frequencies for the input phrase
    '''
    def __get_words_and_bigrams_count(self, line):
        words = {}
        bigrams_dict = {}

        tokens = [t for t in line.split() if len(t) > 1]
        bigrams = list(zip(tokens, tokens[1:]))
        for word in tokens:
            if word in words:
                words[word] = words[word] + 1
            else:
                words[word] = 1
        for bg in bigrams:
            if bg in bigrams_dict:
                bigrams_dict[bg] += 1
            else:
                bigrams_dict[bg] = 1

        return [words, bigrams_dict]

    '''
    @:author davide
    Return the first k words coupled with the specified word. the_dict is ordered in descending order, so the first k
    value are those with the highest PMI.
    '''
    def get_higher_PMI(k, word, bigrams_dict):
        keys = [i for i in bigrams_dict.keys() if word in i]
        keys = keys[:5]
        the_words = []
        for w in keys:
            if (w[0] == word):
                the_words.append((w[1], bigrams_dict[w]))
            else:
                the_words.append((w[0], bigrams_dict[w]))
        return the_words

    '''
    @:author Davide
    @:return a list [W,B] where W is the words frequencies dictionary, while B is the bigrams frequencies dictionary.
    Deserialize words/bigrams frequencies
    '''
    def deserialize_dicts(self, fname):
        with open(fname, 'rb') as pkl_file:
            data = pickle.load(pkl_file)
        return data

    '''
    @:author Davide

    Persist the words/bigrams frequencies to file
    '''
    def persist_dicts(self, words_dict, bigrams_dict, fname):
        with open(fname, 'wb') as output:
            pickle.dump([words_dict, bigrams_dict], output)

    '''
    @:author Davide
    Deserialize the PMI data structure
    '''
    def deserialize_pmi(self, fname):
        with open(fname, 'rb') as pkl_file:
            data = pickle.load(pkl_file)
        return data

    '''
    @:author Davide
    '''
    def persist_pmi(self, pmi, fname):
        with open(fname, 'wb') as output:
            pickle.dump(pmi, output)