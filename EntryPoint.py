import DatasetLoader
import TweetsCleaner

if __name__ == "__main__":
    cleaner = TweetsCleaner.TweetsCleaner()
    loader = DatasetLoader.DatasetLoader()

    tweets_dataset = loader.LoadTweets('/home/davide/Dropbox/Università/IRSW/esame/dataset/training_set_text.csv')
    tweets_cleaned = cleaner.ProcessDatasetDict(tweets_dataset)

    print(tweets_cleaned)
