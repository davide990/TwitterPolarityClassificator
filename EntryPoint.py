import pickle
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
from os import listdir
from os.path import isfile, join

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

            type_eval[type][fold_label]["accuracy"] = evaluator.Accuracy(prediction, labels)
            type_eval[type][fold_label]["precision"] = evaluator.Precision(prediction, labels)
            type_eval[type][fold_label]["recall"] = evaluator.Recall(prediction, labels)
            type_eval[type][fold_label]["f1score"] = evaluator.F1score(prediction, labels)
            nfold+=1
            """
            accuracy = evaluator.Accuracy(prediction, classes_dataset)
            precision.append(evaluator.Precision(prediction, classes_dataset))
            recall.append(evaluator.Recall(prediction, classes_dataset))
            fscore.append(evaluator.F1score(prediction, classes_dataset))
            print('Accuracy: ' + str(accuracy))
            """
    '''
    numfeatures = features.shape[1]
    fname = 'Dati/BayesResult_numfeatures_' + str(numfeatures) + '.json'
    jsonFile = open(fname, "w")
    json.dump(type_eval, jsonFile)
    '''
    return type_eval

def SVMtest(features, labels):

    fold_list = Utils.kfold2(features.shape[0], 10)
    #precision = []
    #recall = []
    #fscore = []
    type_eval = {}
    kernels = ['linear', 'rbf']
    svm = SVMClassifier.SVMClassifier()
    evaluator = ClassifierEvaluation.ClassifierEvaluation()


    for kernel in kernels:
        type_eval[kernel] = {}  # Per ogni tipo di classificatore inizializzo un dizionario
        nfold = 1
        for fold in fold_list:
            fold_label = "fold" + str(nfold)
            type_eval[kernel][fold_label] = {}
            train = features[fold[0], :]
            test = features[fold[1], :]
            if kernel=='linear':
                svm.trainLinearSVM(train, labels[fold[0]])
            elif kernel=='rbf':
                svm.trainRbfSVM(train, labels[fold[0]])

            prediction = svm.Predict(test)

            type_eval[kernel][fold_label]["accuracy"] = evaluator.Accuracy(prediction, labels)
            type_eval[kernel][fold_label]["precision"] = evaluator.Precision(prediction, labels)
            type_eval[kernel][fold_label]["recall"] = evaluator.Recall(prediction, labels)
            type_eval[kernel][fold_label]["f1score"] = evaluator.F1score(prediction, labels)
            nfold += 1

    #numfeatures = features.shape[1]
    #fname = 'Dati/SVMResult_numfeatures_'+str(numfeatures)+'.json'
    #jsonFile = open(fname, "w")
    #json.dump(type_eval, jsonFile)
    return type_eval

def svm_test(filenames):
    path_class_csv = 'Dati/training_set_features.csv'
    loader = DatasetLoader.DatasetLoader()
    features_dataset = loader.LoadFeatures(path_class_csv)
    classes_dataset = loader.createClasses(features_dataset)
    labels = numpy.array(list(classes_dataset.values()))
    output_file = 'Dati/outSVM.json'
    result = {}
    for path in filenames:
        base = os.path.basename(path)
        fname = os.path.splitext(base)[0]   #NO extension

        with open(path, 'rb') as pkl_file:
            print('processing file \''+fname+'\'')
            features = pickle.load(pkl_file)
            out = SVMtest(features, labels)
            result[fname] = out
            del out

    jsonFile = open(output_file, "w")
    json.dump(result, jsonFile)


def do_svm_test(folder_path):
    files = [entry.path for entry in os.scandir(folder_path) if entry.is_file()]
    svm_test(files)

def previous_main():
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
        phrases_tuples.append((count, phrase))
        count += 1

    if not DEBUGMODE or not os.path.exists(path_model_file):
        tfidf = model.get_tfidf(phrases_tuples)
        model.persist_tfidf(tfidf, path_model_file)
    else:
        tfidf = model.deserialize_tfidf(path_model_file)

    doc_index = model.get_doc_index(tfidf)

    # prendo le etichette delle classi per la gold solution
    labels = numpy.array(list(classes_dataset.values()))

    # applico LSA
    reduced = model.LSA(model.get_doc_index_table(doc_index), numFeatures)
    # scalo in [0,1]
    reduced = loader.NormalizeDataset(reduced)

    BayesTest(reduced, labels)
    # SVMtest(reduced, labels)

if __name__ == "__main__":
    do_svm_test('/home/davide/PycharmProjects/features_dataset/test1/')
