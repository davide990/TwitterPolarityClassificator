'''

@:author Domenico

'''

class BayesanClassificator:

    classificator = None

    """

    @:author Domenico
    @:param trainingSet a dictionary with inverted index of training set, classSet a dictionary with class label
    @:param classNum number of class in training set
    @:return classificator model

    """
    def training(self, trainingSet, classSet, classNum, smooth = 0):

        apriori_prob= {}
        prob = {}
        element_in_class = [0]*classNum
        element_total = 0

        #Calcolo le frequenze di ogni termine nelle classi
        for word in trainingSet:
            apriori_prob[word] = [smooth]*4
            prob[word] = 0
            for doc in trainingSet[word]:
                prob[word] += 1
                element_total += 1
                apriori_prob[word][classSet[doc[0]]-1] = + 1
                element_in_class[classSet[doc[0]]-1] += 1

        #A partire dalle frequenze calcolo le probabilità
        for word in trainingSet:
            cont = 0
            prob[word] = prob[word]/element_total
            for c in apriori_prob[word]:
                apriori_prob[word][cont] = c/(element_in_class[cont]+element_total+smooth)
                cont+=1

        self.classificator = {'apriori': apriori_prob, 'prob': prob}
        print(self.classificator)

        return self.classificator


    def test(self, testSet):
        pass