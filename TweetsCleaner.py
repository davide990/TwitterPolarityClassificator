import re

'''
@:author Davide
'''
class TweetsCleaner:

    def __init__(self, **kwargs):
        return super().__init__(**kwargs)

    '''
    @:author Davide
    @:param phrase the string to be processed
    '''
    def __removePunctuation(self, phrase):
        return re.sub(r'[^\w\s]',' ', phrase)

    '''
    @:author Davide
    @:param phrase the string to be processed
    '''
    def __removeURLandEmails(self, phrase):
        return re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', phrase, flags=re.IGNORECASE)

    '''
    @:author Davide
    @:param phrase the string to be processed
    Stopwords from  http://www.ranks.nl/stopwords/italian
    Added stopwords: 'dell','dall'
    '''
    def __removeStopWords(self, phrase):
        stopwords=['dall','dell', 'a', 'adesso', 'ai', 'al', 'alla', 'allo', 'allora', 'altre', 'altri', 'altro', 'anche', 'ancora', 'avere',
         'aveva', 'avevano', 'ben', 'buono', 'che', 'chi', 'cinque', 'comprare', 'con', 'consecutivi', 'consecutivo',
         'cosa', 'cui', 'da', 'del', 'della', 'dello', 'dentro', 'deve', 'devo', 'di', 'doppio', 'due', 'e', 'ecco', 'fare',
         'fine', 'fino', 'fra', 'gente', 'giu', 'ha', 'hai', 'hanno', 'ho', 'il', 'indietro', '	', 'invece', 'io', 'la',
         'lavoro', 'le', 'lei', 'lo', 'loro', 'lui', 'lungo', 'ma', 'me', 'meglio', 'molta', 'molti', 'molto', 'nei',
         'nella', 'no', 'noi', 'nome', 'nostro', 'nove', 'nuovi', 'nuovo', 'o', 'oltre', 'ora', 'otto', 'peggio', 'pero',
         'persone', 'piu', 'poco', 'primo', 'promesso', 'qua', 'quarto', 'quasi', 'quattro', 'quello', 'questo', 'qui',
         'quindi', 'quinto', 'rispetto', 'sara', 'secondo', 'sei', 'sembra', '	', 'sembrava', 'senza', 'sette', 'sia',
         'siamo', 'siete', 'solo', 'sono', 'sopra', 'soprattutto', 'sotto', 'stati', 'stato', 'stesso', 'su', 'subito',
         'sul', 'sulla', 'tanto', 'te', 'tempo', 'terzo', 'tra', 'tre', 'triplo', 'ultimo', 'un', 'una', 'uno', 'va', 'vai',
         'voi', 'volte', 'vostro']
        new_phrase = [word for word in phrase.split() if word not in stopwords and len(word)>1]
        return ' '.join(new_phrase)

    '''
    @:author Davide
    @:param phrase the string to be processed
    '''
    def __removeTwitterUsernames(self, phrase):
        return re.sub(r'(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9_]+)', '', phrase, flags=re.IGNORECASE)

    '''
    @:author Davide
    @:param phrase the string to be processed
    Simple algorithm for removing excessive letters repetitions in phrase's words
    '''
    def __fixMultipleLetters(self, phrase):
        counter = 0
        previousLetter = ''
        output_phrase = []
        for word in phrase.split():
            if len(word) < 3:
                continue
            output_word = []
            for c in word:
                if c == previousLetter:
                    counter += 1
                else:
                    counter = 1

                if counter < 3:
                    output_word.append(c)
                previousLetter = c
            output_phrase.append(''.join(output_word))
        return ' '.join(output_phrase)

    '''
    @:author Davide
    @:param phrase the string to be processed
    '''
    def ProcessPhrase(self, phrase, debug=False):
        lower_phrase = phrase.lower()
        if debug:
            print("lower phrase: '"+lower_phrase+"'")

        no_url = self.__removeURLandEmails(lower_phrase)
        if debug:
            print("without urls: '" + no_url + "'")

        no_username = self.__removeTwitterUsernames(no_url)
        if debug:
            print("without usernames: '" + no_username + "'")

        no_punctuation = self.__removePunctuation(no_username)
        if debug:
            print("without punctuation: '" + no_punctuation + "'")

        no_stopwords = self.__removeStopWords(no_punctuation)
        if debug:
            print("without stop words: '" + no_stopwords + "'")

        no_letters_repetitions = self.__fixMultipleLetters(no_stopwords)
        if debug:
            print("without excessice letters repetitions: '" + no_stopwords + "'")

        return no_letters_repetitions

    '''
        @:author Davide
        @:param phrasesList a list of strings to be processed
    '''
    def ProcessMultiplePhrases(self, phrasesList):
        outputList = []
        for phrase in phrasesList:
            outputList.append(self.ProcessPhrase(phrase))
        return outputList

    '''
        @:author Davide
        @:param fname the full path to the file to be processed line by line.
    '''
    def ProcessFile(self, fname):
        outputList = []
        with open(fname) as f:
            for line in f:
                outputList.append(self.ProcessPhrase(line))
        return outputList

# ---------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------
if __name__ == "__main__":
    cleaner = TweetsCleaner()

    #test_str2 = u"Mario Monti: False illusioni, sgradevoli realtà http://t.co/UrlhqFsd Dall'Italia la possibile disintegrazione dell'Unione Europea"
    test_str2 = u"@mauryred82 l'ho letto quell'articolo in treno sul CORRIERE DELLA SERA... parole durissime..ed e' MARIO MONTI..un moderato.. SIAM MESSI MALE"
    processed = cleaner.ProcessPhrase(test_str2)

    print(test_str2)
    print(processed)




'''
    test_phrase = 'ciaoooooo mondooooo we wellaaa'
    print(cleaner.FixMultipleLetters(test_phrase))'''
