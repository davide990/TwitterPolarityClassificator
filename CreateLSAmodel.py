import os.path
import DatasetLoader
import TweetsCleaner
import VectorModel
import ClassifierEvaluation
import pickle

if __name__ == "__main__":

    DEBUGMODE = 1

    path_dataset_dav_windows = 'Dati/training_set_text.csv'
    path_class_csv = 'Dati/training_set_features.csv'
    path_model_file = 'Dati/model.dat'

    cleaner = TweetsCleaner.TweetsCleaner()
    loader = DatasetLoader.DatasetLoader()
    model = VectorModel.VectorModel()
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

    #applico LSA
    nfeatures = len(tfidf)

    for i in range(int(nfeatures*0.10), int(nfeatures), int(nfeatures*0.10)):
        print("Calcolo LSA per "+str(i)+" componenti")
        reduced = model.LSA(model.get_doc_index_table(doc_index), i)
        print("Salvo LSA")
        lsa = open("Dati/LSA-"+str(i),"wb")
        pickle.dump(reduced, lsa)

pickle.dump(reduced, lsa)