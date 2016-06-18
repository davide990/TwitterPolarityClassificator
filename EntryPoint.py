import os.path
import DatasetLoader
import TweetsCleaner
import VectorModel
import BayesanClassificator

if __name__ == "__main__":

    DEBUGMODE = 1;

    path_dataset_dav_windows = 'Dati/training_set_text.csv'
    path_class_csv = 'Dati/training_set_features.csv'
    path_model_file = 'Dati/model.dat'

    cleaner = TweetsCleaner.TweetsCleaner()
    loader = DatasetLoader.DatasetLoader()

    tweets_dataset = loader.LoadTweets(path_dataset_dav_windows)
    tweets_cleaned = cleaner.ProcessDatasetDict(tweets_dataset)
    features_dataset = loader.LoadFeatures(path_class_csv)

    """
        Trasforma il vettore delle features in un dizionario con chiave IdDoc e valore la classe corrispondente
        (1 : neutra, 2: positiva, 3: negativa, 4: mista
    """
    classes_dataset = loader.createClasses(features_dataset)



    all_phrases = list(tweets_cleaned.values())

    count = 0
    phrases_tuples = []
    for phrase in all_phrases:
        phrases_tuples.append((count,phrase))
        count += 1

    model = VectorModel.VectorModel()
    if not DEBUGMODE or not os.path.exists(path_model_file):
        tfidf = model.get_tfidf(phrases_tuples)
        model.persist_tfidf(tfidf,path_model_file)
    else:
        tfidf = model.deserialize_tfidf(path_model_file)

    doc_index = model.get_doc_index(tfidf)

    classificator = BayesanClassificator.BayesanClassificator()
    classificator.training(tfidf, classes_dataset, 4, 0.10)
    result = classificator.test(tfidf)

    print(result)
    #print(tfidf)
