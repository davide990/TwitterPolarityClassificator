import matplotlib.pyplot as plt
import json
import numpy as np
import numpy.matlib
import os.path


class ClassifierEvaluation:
    def __init__(self, **kwargs):
        return super().__init__(**kwargs)

    '''
        @:author Davide
        @:param predicted the solution predicted by a classifier. It is a dictionary where the key is the ID of the
            document while the value is an integer in range 1-4 1 indicates no polarity, 2 indicates positive polarity,
            3 indicates negative polarity and 4 indicates mixed polarity
        @:param gold the gold solution. It has the same format of the predicted solution
    '''

    def Precision(self, predicted, gold):
        TP = 0
        FP = 0
        for doc in predicted:
            if predicted[doc] == gold[doc]:
                TP += 1
            if gold[doc] == 2 and predicted[doc] != gold[doc]:
                FP += 1
        return TP / (TP + FP)

    '''
        @:author Davide
        @:param predicted the solution predicted by a classifier. It is a dictionary where the key is the ID of the
            document while the value is an integer in range 1-4 1 indicates no polarity, 2 indicates positive polarity,
            3 indicates negative polarity and 4 indicates mixed polarity
        @:param gold the gold solution. It has the same format of the predicted solution

    '''

    def Recall(self, predicted, gold):
        TP = 0
        FN = 0
        for doc in predicted:
            if predicted[doc] == gold[doc]:
                TP += 1
            if gold[doc] == 3 and predicted[doc] != gold[doc]:
                FN += 1
        return TP / (TP + FN)

    '''
        @:author Davide
        @:param predicted the solution predicted by a classifier. It is a dictionary where the key is the ID of the
            document while the value is an integer in range 1-4 1 indicates no polarity, 2 indicates positive polarity,
            3 indicates negative polarity and 4 indicates mixed polarity
        @:param gold the gold solution. It has the same format of the predicted solution
    '''

    def F1score(self, predicted, gold):
        p = self.Precision(predicted, gold)
        r = self.Recall(predicted, gold)
        return 2 * ((p * r) / (p + r))

    '''
        @:author Davide
        @:param predicted the solution predicted by a classifier. It is a dictionary where the key is the ID of the
            document while the value is an integer in range 1-4 1 indicates no polarity, 2 indicates positive polarity,
            3 indicates negative polarity and 4 indicates mixed polarity
        @:param gold the gold solution. It has the same format of the predicted solution
    '''

    def Accuracy(self, predicted, gold):
        correct = 0
        for doc in predicted:
            if predicted[doc] == gold[doc]:
                correct += 1
        return correct / len(predicted.keys())

    '''
        @:author Davide
        @:param predicted the solution predicted by a classifier. It is a dictionary where the key is the ID of the
            document while the value is an integer in range 1-4 1 indicates no polarity, 2 indicates positive polarity,
            3 indicates negative polarity and 4 indicates mixed polarity
        @:param gold the gold solution. It has the same format of the predicted solution
    '''

    def plotPrecisionRecall(self, precision, recall):
        plt.plot(recall, precision)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.show()

    '''
    @:author Davide
    @:param fscore the fscore array
    @:param precision the precision array
    @:param recall the recall array
    @:param accuracy the accuracy array

    Given the 4 arrays containing the performance of a classifier, this method assembles them into a unique plot and
    save this to a png file
    '''

    def plot_performance(self, out_figure_fname, fscore, precision, recall, accuracy):
        N = len(fscore)
        folds = [i for i in range(1, N + 1)]
        y_pos = np.arange(len(folds))

        plt.figure(figsize=(11, 11))
        plt.subplot(2, 2, 1)
        plt.barh(y_pos, fscore, align='center', alpha=0.4, color='red')
        mean_fscore = list(numpy.matlib.repmat(np.mean(fscore), 1, N))[0]
        plt.plot(mean_fscore, y_pos, linestyle='-', linewidth=1.5)
        plt.yticks(y_pos, folds)
        plt.xlabel('F-Score')
        plt.ylabel('Folds')
        plt.grid(True)

        plt.subplot(2, 2, 2)
        plt.barh(y_pos, precision, align='center', alpha=0.4, color='green')
        mean_precision = list(numpy.matlib.repmat(np.mean(precision), 1, N))[0]
        plt.plot(mean_precision, y_pos, linestyle='-', linewidth=1.5)
        plt.yticks(y_pos, folds)
        plt.xlabel('Precision')
        plt.ylabel('Folds')
        plt.grid(True)

        plt.subplot(2, 2, 3)
        plt.barh(y_pos, recall, align='center', alpha=0.4, color='blue')
        mean_recall = list(numpy.matlib.repmat(np.mean(recall), 1, N))[0]
        plt.plot(mean_recall, y_pos, linestyle='-', linewidth=1.5)
        plt.yticks(y_pos, folds)
        plt.xlabel('Recall')
        plt.ylabel('Folds')
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.barh(y_pos, accuracy, align='center', alpha=0.4, color='yellow')
        mean_accuracy = list(numpy.matlib.repmat(np.mean(accuracy), 1, N))[0]
        plt.plot(mean_accuracy, y_pos, linestyle='-', linewidth=1.5)
        plt.yticks(y_pos, folds)
        plt.xlabel('Accuracy')
        plt.ylabel('Folds')
        plt.grid(True)
        plt.savefig(out_figure_fname)
        plt.close()

    '''
    @:author Davide
    @:param fname the complete path to a json file

    Given a json input path containing the dictionary of the classifier in the following format:

    "LSA-[numFeatures]":
    {
        "[kernelType]":
        {
            "[IDfold]":
            {
                "recall": #value
                "precision": #value
                "accuracy": #value
                "f1score": #value
            },
            ...
        },
        ...
    },
    ...

    Plot the performance of the classifiers and save a plot to file for each number of features/kernel combinations
    found in the json dictionary.
    '''

    def PlotPerformanceFromJson(self, fname):
        with open(fname, 'r') as file:
            data = json.load(file)

            for num_features in data:
                for kernel_type in data[num_features]:
                    accuracy = []
                    precision = []
                    recall = []
                    fscore = []
                    for fold in data[num_features][kernel_type]:
                        accuracy.append(data[num_features][kernel_type][fold]['accuracy'])
                        recall.append(data[num_features][kernel_type][fold]['recall'])
                        precision.append(data[num_features][kernel_type][fold]['precision'])
                        fscore.append(data[num_features][kernel_type][fold]['f1score'])

                    self.plot_performance('SVM_' + str(num_features) + '_' + str(kernel_type) + '.png', fscore,
                                          precision, recall, accuracy)

    '''
    @:author Davide
    @:param out_json_fname the output json file containing the converted dictionary
    @:param folder_path the path to a folder containing the performance of different classifiers

    This method assembles all the performance of the classifiers in the json files within the given directory and
    assemble a new dictionary containing the avarage performances. The new dictionary has the following format:

    "[precision/recall/accuracy/f1score]":
    {
        "[kernelType]":
        {
            "[NumFeatures]": value
            ...
        },
        ...
    },
    ...
    '''
    def get_overall_performance_dict(self, out_json_fname, folder_path):
        files = [entry.path for entry in os.scandir(folder_path) if entry.is_file()]
        perf_dict = {}
        perf_dict['accuracy'] = {}
        perf_dict['precision'] = {}
        perf_dict['recall'] = {}
        perf_dict['f1score'] = {}

        for the_file in files:
            if not the_file.endswith('.json'):
                continue
            with open(the_file, 'r') as file:
                data = json.load(file)
                for num_features in data:
                    for kernel_type in data[num_features]:
                        perf_dict['accuracy'][kernel_type] = {}
                        perf_dict['precision'][kernel_type] = {}
                        perf_dict['recall'][kernel_type] = {}
                        perf_dict['f1score'][kernel_type] = {}
                for num_features in data:
                    for kernel_type in data[num_features]:
                        accuracy = []
                        precision = []
                        recall = []
                        fscore = []
                        for fold in data[num_features][kernel_type]:
                            accuracy.append(data[num_features][kernel_type][fold]['accuracy'])
                            recall.append(data[num_features][kernel_type][fold]['recall'])
                            precision.append(data[num_features][kernel_type][fold]['precision'])
                            fscore.append(data[num_features][kernel_type][fold]['f1score'])

                        perf_dict['accuracy'][kernel_type][num_features] = np.mean(accuracy)
                        perf_dict['precision'][kernel_type][num_features] = np.mean(precision)
                        perf_dict['recall'][kernel_type][num_features] = np.mean(recall)
                        perf_dict['f1score'][kernel_type][num_features] = np.mean(fscore)
        jsonFile = open(out_json_fname, "w")
        json.dump(perf_dict, jsonFile)

    '''
    @:author Davide
    @:param out_plot_fname the output plot file name
    @:param fname the complete path to the json file containing the overall classifier performance (see
        get_overall_performance_dict(..))
    @:param kernel_type the kernel type to evaluate in string format


    '''

    def plot_overall_precision(self, out_plot_fname, fname, kernel_type):
        with open(fname, 'r') as file:
            data = json.load(file)
            plot_id = 1
            colors = ['red', 'green', 'blue', 'yellow']
            plt.figure(figsize=(17, 9))

            for measure in data:
                features = list(data[measure][kernel_type].keys())
                values = list(data[measure][kernel_type].values())
                plt.subplot(2, 2, plot_id)
                y_pos = np.arange(len(values))
                plt.barh(y_pos, values, align='center', alpha=0.4, color=colors[plot_id - 1])
                mean_precision = list(numpy.matlib.repmat(np.mean(values), 1, len(values)))[0]
                plt.plot(mean_precision, y_pos, linestyle='-', linewidth=1.5)
                plt.yticks(y_pos, features)
                plt.xlabel(measure)
                plt.ylabel('Num Features')
                plt.grid(True)
                plot_id += 1

        plt.savefig(out_plot_fname)
        plt.close()
