import ClassifierEvaluation
import os

if __name__ == "__main__":
    #Name of the file containing the overall classificators performances
    overall_performance_dict_fname = 'Dati/svm_overall_results.json'

    try:
        os.remove(overall_performance_dict_fname)
    except OSError:
        pass

    #Create a new classifier evaluator
    evaluator = ClassifierEvaluation.ClassifierEvaluation()

    #Get the overall classificators performance
    evaluator.get_overall_performance_dict(overall_performance_dict_fname, 'Dati/')

    #Create plots and save them to file
    evaluator.plot_overall_precision('svm_linear.png', overall_performance_dict_fname, 'linear')
    evaluator.plot_overall_precision('svm_rbf.png', overall_performance_dict_fname, 'rbf')