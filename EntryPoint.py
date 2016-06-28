import pickle
import os.path
import DatasetLoader
import BayesanClassificator
import Utils
import ClassifierEvaluation
import numpy
import json

def BayesTest(features, labels):
    fold_list = Utils.kfold2(features.shape[0], 10)
    type_eval = {}    #Dizionario per i risultati
    classifierType = ['multinomial', 'bernoulli', 'gaussian']
    classificator = BayesanClassificator.BayesanClassificator()
    evaluator = ClassifierEvaluation.ClassifierEvaluation()
    for type in classifierType:
        print("Processing Type: "+str(type))
        type_eval[type] = {} #Per ogni tipo di classificatore inizializzo un dizionario
        nfold = 1
        for fold in fold_list:
            print("Fold "+str(nfold))
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

    return type_eval

def SaveBayesResults(filenames):
    print("Carico il Dataset")
    path_class_csv = 'Dati/training_set_features.csv'
    loader = DatasetLoader.DatasetLoader()
    features_dataset = loader.LoadFeatures(path_class_csv)
    classes_dataset = loader.createClasses(features_dataset)
    labels = numpy.array(list(classes_dataset.values()))
    output_file = 'Dati/outBayes.json'
    result = {}
    for path in filenames:
        base = os.path.basename(path)
        fname = os.path.splitext(base)[0]   #NO extension

        with open(path, 'rb') as pkl_file:
            print('processing file \''+fname+'\'')
            features = pickle.load(pkl_file)
            features = loader.NormalizeDataset(features)
            out = BayesTest(features, labels)
            result[fname] = out
            del out

    jsonFile = open(output_file, "w")
    json.dump(result, jsonFile)

########################################################################################################################
if __name__ == "__main__":

    lsaFolder = 'Dati/'

    files = [entry.path for entry in os.scandir(lsaFolder) if entry.is_file()]
    SaveBayesResults(files)
