import os.path
import DatasetLoader
import TweetsCleaner
import VectorModel
import BayesanClassificator
import Utils
import ClassifierEvaluation
import numpy
import SVMClassifier

if __name__ == "__main__":

    DEBUGMODE = 0

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

    #applico LSA
    reduced = model.LSA(model.get_doc_index_table(doc_index), 100)
    #scalo in [0,1]
    reduced = loader.NormalizeDataset(reduced)

    #prendo le etichette delle classi per la gold solution
    labels = numpy.array(list(classes_dataset.values()))
    """
        Genero i kfold
    """
    fold_list = Utils.kfold2(reduced.shape[0], 10)

    precision = []
    recall = []
    fscore = []

    svm = SVMClassifier.SVMClassifier()

    for fold in fold_list:
        train = reduced[fold[0], :]
        test = reduced[fold[1], :]
        #classificator.TrainMultinomialBayes(train, labels[fold[0]])
        #svm.trainRbfSVM(train, labels[fold[0]])
        svm.trainLinearSVM(train, labels[fold[0]])
        prediction = svm.Predict(test)

        accuracy = evaluator.Accuracy(prediction, classes_dataset)
        precision.append(evaluator.Precision(prediction, classes_dataset))
        recall.append(evaluator.Recall(prediction, classes_dataset))
        fscore.append(evaluator.F1score(prediction, classes_dataset))
        print('Accuracy: '+str(accuracy))

    print('DONE')
    print('Precision: ' + str(precision))
    print('Recall: ' + str(recall))
    print('FScore: ' + str(fscore))

    '''
    precision = []
    recall = []
    fscore = []
    for fold in fold_list:
        """
            Addestro il classificatore
        """
        classificator.training(fold[0], classes_dataset, 4, 0.10)
        """
            Eseguo il test
        """
        result = classificator.test(fold[1])

        """
            Valuto il classificatore
        """
        precision.append(evaluator.Precision(result, classes_dataset))
        recall.append(evaluator.Recall(result, classes_dataset))
        fscore.append(evaluator.F1score(result, classes_dataset))

    print(precision)
    '''



    #print(tfidf)