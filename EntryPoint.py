import os.path
import DatasetLoader
import TweetsCleaner
import VectorModel
import BayesanClassificator
import Utils
import ClassifierEvaluation

if __name__ == "__main__":

    DEBUGMODE = 1;

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
    features_dataset = loader.LoadFeatures(path_class_csv)

    """
        Trasforma il vettore delle features in un dizionario con chiave IdDoc e valore la classe corrispondente
        (1 : neutra, 2: positiva, 3: negativa, 4: mista
    """
    classes_dataset = loader.createClasses(features_dataset)



    """
        Genero il Modello TF-IDF
    """
    all_phrases = list(tweets_cleaned.values())

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

    """
        Genero i kfold
    """
    fold_list = Utils.kfold(tfidf, count, 10)


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
    #print(tfidf)
