import re

'''
@:author Davide
'''
class DatasetLoader:

    def __init__(self, **kwargs):
        return super().__init__(**kwargs)

    '''
        @:author Davide
        @:return a tuple containing the feature informations
        Parse a feature from the twitter dataset. A feature string must have the following format to be corrected parsed:

        -> '"idtwitter",subj,opos,oneg,iro,lpos,lneg,top'
    '''
    def _parseFeature(self, feature_str):
        p = re.compile(r'^\"(?P<ID>.*)\",(?P<subj>\d+),(?P<opos>\d+),(?P<oneg>\d+),(?P<iro>\d+),(?P<lpos>\d+),(?P<lneg>\d+),(?P<top>\d+)$')
        return re.findall(p, feature_str)[0]

    '''
        @:author Davide
        @:return a tuple containing the tweets informations (ID and tweet's text)
        Parse a tweet from the twitter dataset. A tweet string must have the following format to be corrected parsed:

        -> '"idtwitter","tweet"'
    '''
    def _parseText(self, text):
        p = re.compile(r'\"(?P<ID>.*)\",\"(?P<text>.*)\"')
        return re.findall(p, text)[0]

    '''
        @:author Davide
        @:return a dictionary containing the ID of the tweets as keys, and the tweets text as values
        @:param fname the full path to the file to be processed. Each line in file must have the following format:

        -> '"idtwitter","tweet"'
    '''
    def LoadTweets(self, fname):
        tweets = {}
        line_counter = 1
        lines_not_parsed = []

        with open(fname) as f:
            for line in f:
                try:
                    tweet = self._parseText(line)
                    tweets[tweet[0]] = tweet[1]
                    line_counter += 1
                except:
                    lines_not_parsed.append((line_counter,line))
                    line_counter += 1
                    continue
        if lines_not_parsed:
            raise Exception('Error parsing file. Following lines contains errors: \n' + str(lines_not_parsed))
        return tweets

    '''
        @:author Davide
        @:return a dictionary containing the ID of the tweets as keys, and the features as values
        @:param fname the full path to the file to be processed. Each line in file must have the following format:

        -> "idtwitter",subj,opos,oneg,iro,lpos,lneg,top
    '''
    def LoadFeatures(self, fname):
        features = {}
        line_counter = 1
        lines_not_parsed = []

        with open(fname) as f:
            for line in f:
                try:
                    feature = self._parseFeature(line)
                    features[feature[0]] = feature[1:]
                    line_counter += 1
                except:
                    lines_not_parsed.append((line_counter,line))
                    line_counter += 1
                    continue
        if lines_not_parsed:
            raise Exception('Error parsing file. Following lines contains errors: \n' + str(lines_not_parsed))
        return features


#------------------------------------------------------------------------
#------------------------------------------------------------------------
if __name__ == "__main__":

    loader = DatasetLoader()
    features = loader.LoadFeatures('/home/davide/Dropbox/Università/IRSW/esame/dataset/training_set_features.csv')
    tweets = loader.LoadTweets('/home/davide/Dropbox/Università/IRSW/esame/dataset/training_set_text.csv')