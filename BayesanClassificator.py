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
    def training(self, trainingSet, classSet, classNum):

        apriori_prob= {}
        element_in_class = [0]*classNum
        for word in trainingSet:
            apriori_prob[word] = [0]*4
            for doc in trainingSet[word]:
                apriori_prob[word][classSet[doc[0]]-1] = + 1
                element_in_class[classSet[doc[0]]-1] += 1

        
        for word in trainingSet:
            cont = 0
            for c in apriori_prob[word]:
                apriori_prob[word][cont] = c/element_in_class[cont]
                cont+=1
        print(apriori_prob)

    def test(self, testSet):
        pass