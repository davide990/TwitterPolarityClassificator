from sklearn.cross_validation import KFold

'''
@:param num_docs number of documents in the collection
@:param tfidf the tf-idf dictionary
@:param nfold number of folds to be constructed

Given a TF-IDF dictionary, this method generate a list of tuple where each tuple contains a possible separation in
training/test set.
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

            if not tf_idf_train[word]:
                del tf_idf_train[word]

            if not tf_idf_test[word]:
                del tf_idf_test[word]

        folds.append((tf_idf_train,tf_idf_test))
    return folds


def kfold2(num_docs, nfold):
    kf = KFold(num_docs, n_folds=nfold)

    folds = []
    for traincv, testcv in kf:
        folds.append((traincv,testcv))
    return folds