import matplotlib.pyplot as plt

class ClassifierEvaluation:

    def __init__(self, **kwargs):
        return super().__init__(**kwargs)

    '''
        @:author Davide
        @:param predicted the solution predicted by a classifier. It is a dictionary where the key is the ID of the
            document while the value is an integer in range 1-4 1 indicates no polarity, 2 indicates positive polarity,
            3 indicates negative polarity and 4 indicates mixed polarity
        @:param gold the gold solution. It has the same format of the predicted solution
    '''
    def Precision(self, predicted, gold):
        TP = 0
        FP = 0
        for doc in predicted:
            if predicted[doc] == gold[doc]:
                TP += 1
            if gold[doc] == 2 and predicted[doc] != gold[doc]:
                FP += 1
        return TP / (TP+FP)

    '''
        @:author Davide
        @:param predicted the solution predicted by a classifier. It is a dictionary where the key is the ID of the
            document while the value is an integer in range 1-4 1 indicates no polarity, 2 indicates positive polarity,
            3 indicates negative polarity and 4 indicates mixed polarity
        @:param gold the gold solution. It has the same format of the predicted solution

    '''
    def Recall(self, predicted, gold):
        TP = 0
        FN = 0
        for doc in predicted:
            if predicted[doc] == gold[doc]:
                TP += 1
            if gold[doc] == 3 and predicted[doc] != gold[doc]:
                FN += 1
        return TP / (TP + FN)

    '''
        @:author Davide
        @:param predicted the solution predicted by a classifier. It is a dictionary where the key is the ID of the
            document while the value is an integer in range 1-4 1 indicates no polarity, 2 indicates positive polarity,
            3 indicates negative polarity and 4 indicates mixed polarity
        @:param gold the gold solution. It has the same format of the predicted solution
    '''
    def F1score(self, predicted, gold):
        p = self.Precision(predicted, gold)
        r = self.Recall(predicted, gold)
        return 2*((p*r)/(p+r))


    '''
        @:author Davide
        @:param predicted the solution predicted by a classifier. It is a dictionary where the key is the ID of the
            document while the value is an integer in range 1-4 1 indicates no polarity, 2 indicates positive polarity,
            3 indicates negative polarity and 4 indicates mixed polarity
        @:param gold the gold solution. It has the same format of the predicted solution
    '''
    def Accuracy(self, predicted, gold):
        correct = 0
        for doc in predicted:
            if predicted[doc] == gold[doc]:
                correct+=1
        return correct/len(predicted.keys())

    '''
        @:author Davide
        @:param predicted the solution predicted by a classifier. It is a dictionary where the key is the ID of the
            document while the value is an integer in range 1-4 1 indicates no polarity, 2 indicates positive polarity,
            3 indicates negative polarity and 4 indicates mixed polarity
        @:param gold the gold solution. It has the same format of the predicted solution
    '''
    def plotPrecisionRecall(self, precision, recall):
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.show()