from sklearn import svm

'''
@:author Davide

SVM for tweets classification
'''
class SVMClassifier:

    classificator = None

    '''
    @:author Davide
    Train an SVM with a RBF kernel
    '''
    def trainLinearSVM(self, features, labels):
        self.classificator = svm.SVC(kernel='linear')
        self.classificator.fit(features, labels)

    '''
    @:author Davide
    Train an SVM with a RBF kernel
    '''
    def trainRbfSVM(self, features, labels):
        self.classificator = svm.SVC(kernel='rbf')
        self.classificator.fit(features, labels)

    '''
    @:author Davide
    '''
    def Predict(self, features):
        prediction = self.classificator.predict(features)
        pred_dict = {}
        for line in range(0, len(prediction)):
            pred_dict[line] = prediction[line]

        return pred_dict