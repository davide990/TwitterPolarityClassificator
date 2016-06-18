from sklearn.cross_validation import KFold

'''
num_docs numero totale di documenti nella collezione

 TODO DA PROVARE

'''
def kfold(tfidf, num_docs, nfold):
    kf = KFold(num_docs, n_folds=nfold)

    folds = []
    for traincv, testcv in kf:
        tf_idf_test = tfidf.copy()
        tf_idf_train = tfidf.copy()

        for word in list(tfidf.keys()):
            tf_idf_train[word] = [couple for couple in tf_idf_train[word] if couple[0] in traincv]
            tf_idf_test[word] = [couple for couple in tf_idf_test[word] if couple[0] in testcv]
        folds.append((tf_idf_train,tf_idf_test))
    return folds
