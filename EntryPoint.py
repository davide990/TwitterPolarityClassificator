import os.path
import DatasetLoader
import TweetsCleaner
import VectorModel

if __name__ == "__main__":

    DEBUGMODE = 1;

    path_dataset_dav_windows = 'Dati/training_set_text.csv'
    path_model_file = 'Dati/model.dat'

    cleaner = TweetsCleaner.TweetsCleaner()
    loader = DatasetLoader.DatasetLoader()

    tweets_dataset = loader.LoadTweets(path_dataset_dav_windows)
    tweets_cleaned = cleaner.ProcessDatasetDict(tweets_dataset)

   #print(tweets_cleaned)


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
    print(tfidf)
