import os.path
import DatasetLoader
import TweetsCleaner
import VectorModel
import BayesanClassificator
import Utils
import ClassifierEvaluation
import numpy
import SVMClassifier
import json

def BayesTest(features, labels):
    fold_list = Utils.kfold2(features.shape[0], 10)
    #precision = []
    #recall = []
    #fscore = []
    type_eval = {}    #Dizionario per i risultati
    classifierType = ['multinomial', 'bernoulli', 'gaussian']
    classificator = BayesanClassificator.BayesanClassificator()
    evaluator = ClassifierEvaluation.ClassifierEvaluation()
    for type in classifierType:
        type_eval[type] = {} #Per ogni tipo di classificatore inizializzo un dizionario
        nfold = 1
        for fold in fold_list:
            fold_label = "fold"+str(nfold)
            type_eval[type][fold_label] = {} #Per ogni fold inizializzo un dizionario
            train = features[fold[0], :]
            test = features[fold[1], :]
            if type=='multinomial':
                classificator.TrainMultinomialBayes(train, labels[fold[0]])
            elif type=='bernoulli':
                classificator.TrainBernoulliBayes(train, labels[fold[0]])
            elif type=='gaussian':
                classificator.TrainGaussianBayes(train, labels[fold[0]])

            prediction = classificator.Predict(test)

            type_eval[type][fold_label]["accuracy"] = evaluator.Accuracy(prediction, classes_dataset)
            type_eval[type][fold_label]["precision"] = evaluator.Precision(prediction, classes_dataset)
            type_eval[type][fold_label]["recall"] = evaluator.Recall(prediction, classes_dataset)
            type_eval[type][fold_label]["f1score"] = evaluator.F1score(prediction, classes_dataset)
            nfold+=1
            """
            accuracy = evaluator.Accuracy(prediction, classes_dataset)
            precision.append(evaluator.Precision(prediction, classes_dataset))
            recall.append(evaluator.Recall(prediction, classes_dataset))
            fscore.append(evaluator.F1score(prediction, classes_dataset))
            print('Accuracy: ' + str(accuracy))
            """
    jsonFile = open("Dati/BayesResult.json", "w")
    json.dump(type_eval, jsonFile)

    """
    print('DONE')
    print('Precision: ' + str(precision))
    print('Recall: ' + str(recall))
    print('FScore: ' + str(fscore))
    """


def SVMtest(features, labels):
    fold_list = Utils.kfold2(features.shape[0], 10)
    precision = []
    recall = []
    fscore = []
    kernels = ['linear', 'rbf']
    svm = SVMClassifier.SVMClassifier()
    evaluator = ClassifierEvaluation.ClassifierEvaluation()
    for kernel in kernels:
        for fold in fold_list:
            train = features[fold[0], :]
            test = features[fold[1], :]
            if kernel=='linear':
                svm.trainLinearSVM(train, labels[fold[0]])
            elif kernel=='rbf':
                svm.trainRbfSVM(train, labels[fold[0]])

            prediction = svm.Predict(test)

            accuracy = evaluator.Accuracy(prediction, classes_dataset)
            precision.append(evaluator.Precision(prediction, classes_dataset))
            recall.append(evaluator.Recall(prediction, classes_dataset))
            fscore.append(evaluator.F1score(prediction, classes_dataset))
            print('Accuracy: ' + str(accuracy))

    print('DONE')
    print('Precision: ' + str(precision))
    print('Recall: ' + str(recall))
    print('FScore: ' + str(fscore))


if __name__ == "__main__":

    DEBUGMODE = 0
    numFeatures = 100

    path_dataset_dav_windows = 'Dati/training_set_text.csv'
    path_class_csv = 'Dati/training_set_features.csv'
    path_model_file = 'Dati/model.dat'

    cleaner = TweetsCleaner.TweetsCleaner()
    loader = DatasetLoader.DatasetLoader()
    model = VectorModel.VectorModel()
    classificator = BayesanClassificator.BayesanClassificator()
    evaluator = ClassifierEvaluation.ClassifierEvaluation()

    tweets_dataset = loader.LoadTweets(path_dataset_dav_windows)
    tweets_cleaned = cleaner.ProcessDatasetDict(tweets_dataset)
    features_dataset = loader.LoadFeatures(path_class_csv, 400)

    """
        Trasforma il vettore delle features in un dizionario con chiave IdDoc e valore la classe corrispondente
        (1 : neutra, 2: positiva, 3: negativa, 4: mista
    """
    classes_dataset = loader.createClasses(features_dataset)

    """
        Genero il Modello TF-IDF
    """
    all_phrases = list(tweets_cleaned.values())[:400]

    count = 0
    phrases_tuples = []
    for phrase in all_phrases:
        phrases_tuples.append((count,phrase))
        count += 1

    if not DEBUGMODE or not os.path.exists(path_model_file):
        tfidf = model.get_tfidf(phrases_tuples)
        model.persist_tfidf(tfidf,path_model_file)
    else:
        tfidf = model.deserialize_tfidf(path_model_file)

    doc_index = model.get_doc_index(tfidf)

    # prendo le etichette delle classi per la gold solution
    labels = numpy.array(list(classes_dataset.values()))

    #applico LSA
    reduced = model.LSA(model.get_doc_index_table(doc_index), numFeatures)
    #scalo in [0,1]
    reduced = loader.NormalizeDataset(reduced)

    BayesTest(reduced,labels)
    #SVMtest(reduced, labels)