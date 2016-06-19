'''

@:author Domenico

'''

class BayesanClassificator:

    classificator = None

    """

    @:author Domenico
    @:param trainingSet a dictionary with inverted index of training set,
    @:param classSet a dictionary with class label
    @:param classNum number of class in training set
    @:param smooth value for additive smooth
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

        """
            Aggiungo le probabilità per le parole non presenti nel training set
        """
        apriori_prob['UNKNOW'] = [
                                  smooth/(element_in_class[cont]+element_total+smooth)
                                  for cont in range(0,4)
                                  ]
        prob['UNKNOW'] = smooth/(element_total+smooth)

        self.classificator = {
                              'apriori': apriori_prob,
                              'term_prob': prob,
                              'class_prob': [element_in_class[i]/element_total for i in range(0,4)]
                              }

        print(self.classificator)

        return self.classificator


    def test(self, testSet):

        test_dict = {}
        result_dict = {}
        for word in testSet:
            if word in self.classificator['apriori']:
                docList = testSet[word]
                for doc in docList:
                    if doc in test_dict:
                        test_dict[doc[0]] = [
                                            test_dict[doc[0]][i]*self.classificator['apriori'][word][i]/self.classificator['term_prob'][word]
                                            for i in range(0,4)
                                         ]
                    else:
                        test_dict[doc[0]]=[self.classificator['apriori'][word][i]*self.classificator['class_prob'][i]/self.classificator['term_prob'][word]
                                        for i in range(0,4)
                                        ]
            else:
                docList = testSet[word]
                for doc in docList:
                    if doc in test_dict:
                        test_dict[doc[0]] = [
                            test_dict[doc[0]][i] * self.classificator['apriori']['UNKNOW'][i] / self.classificator['term_prob']['UNKNOW']
                            for i in range(0, 4)
                            ]
                    else:
                        test_dict[doc[0]] = [self.classificator['apriori']['UNKNOW'][i] * self.classificator['class_prob'][i] /
                                          self.classificator['term_prob']['UNKNOW']
                                          for i in range(0, 4)
                                       ]
        for doc in test_dict:
            result_dict[doc] = test_dict[doc].index(max(test_dict[doc]))

        return result_dict

