import DatasetLoader
import TweetsCleaner
import VectorModel

if __name__ == "__main__":

    path_dataset_dav_windows = 'C:/Users/davide/Dropbox/Università/IRSW/esame/dataset/training_set_text.csv'

    cleaner = TweetsCleaner.TweetsCleaner()
    loader = DatasetLoader.DatasetLoader()

    tweets_dataset = loader.LoadTweets(path_dataset_dav_windows)
    tweets_cleaned = cleaner.ProcessDatasetDict(tweets_dataset)

   #print(tweets_cleaned)


    all_phrases = list(tweets_cleaned.values())[:100]

    count = 0
    phrases_tuples = []
    for phrase in all_phrases:
        phrases_tuples.append((count,phrase))
        count += 1

    model = VectorModel.VectorModel()
    tfidf = model.get_tfidf(phrases_tuples)

    model.persist_tfidf(tfidf,'C:/Users/davide/Dropbox/Università/IRSW/esame/dataset/boh.csv')
    dd = model.deserialize_tfidf('C:/Users/davide/Dropbox/Università/IRSW/esame/dataset/boh.csv')

    doc_index = model.get_doc_index(tfidf)
    print(doc_index)
