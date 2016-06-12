import re

'''
@:author Davide
@:param phrase the string to be processed
'''
def RemovePunctuation(phrase):
    return re.sub(r'[^\w\s]',' ', phrase)

'''
@:author Davide
@:param phrase the string to be processed
'''
def RemoveURLandEmails(phrase):
    return re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', phrase, flags=re.IGNORECASE)

'''
@:author Davide
Stopwords from  http://www.ranks.nl/stopwords/italian
Added stopwords: 'dell','dall'
'''
def RemoveStopWords(phrase):
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

def RemoveTwitterUsernames(phrase):
    return re.sub(r'(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9_]+)', '', phrase, flags=re.IGNORECASE)


def ProcessPhrase(phrase, debug=False):
    lower_phrase = phrase.lower()
    if debug:
        print("lower phrase: '"+lower_phrase+"'")

    no_url = RemoveURLandEmails(lower_phrase)
    if debug:
        print("without urls: '" + no_url + "'")

    no_username = RemoveTwitterUsernames(no_url)
    if debug:
        print("without usernames: '" + no_username + "'")

    no_punctuation = RemovePunctuation(no_username)
    if debug:
        print("without punctuation: '" + no_punctuation + "'")

    no_stopwords = RemoveStopWords(no_punctuation)
    if debug:
        print("without stop words: '" + no_stopwords + "'")

    return no_stopwords


if __name__ == "__main__":
    #test_str2 = u"Mario Monti: False illusioni, sgradevoli realtà http://t.co/UrlhqFsd Dall'Italia la possibile disintegrazione dell'Unione Europea"
    test_str2 = u"@mauryred82 l'ho letto quell'articolo in treno sul CORRIERE DELLA SERA... parole durissime..ed e' MARIO MONTI..un moderato.. SIAM MESSI MALE"

    processed = ProcessPhrase(test_str2)

    print(test_str2)
    print(processed)
