import ClassifierEvaluation
import os

if __name__ == "__main__":
    #Name of the file containing the overall classificators performances
    overall_svm_performance_dict_fname = 'Dati/svm_overall_results.json'
    overall_bayes_performance_dict_fname = 'Dati/bayes_overall_results.json'

    try:
        os.remove(overall_svm_performance_dict_fname)
        os.remove(overall_bayes_performance_dict_fname)
    except OSError:
        pass

    #Create a new classifier evaluator
    evaluator = ClassifierEvaluation.ClassifierEvaluation()

    #Get the overall classificators performance
    evaluator.get_overall_performance_dict(overall_svm_performance_dict_fname, 'Dati/SVM_results.json')
    evaluator.get_overall_performance_dict(overall_bayes_performance_dict_fname, 'Dati/Bayes_results.json')

    #Create plots and save them to file
    evaluator.plot_overall_precision('svm_linear.png', overall_svm_performance_dict_fname, 'linear')
    evaluator.plot_overall_precision('svm_rbf.png', overall_svm_performance_dict_fname, 'rbf')

    evaluator.plot_overall_precision('bayes_bernoulli.png', overall_bayes_performance_dict_fname, 'bernoulli')
    evaluator.plot_overall_precision('bayes_gaussian.png', overall_bayes_performance_dict_fname, 'gaussian')
    evaluator.plot_overall_precision('bayes_multinomial.png', overall_bayes_performance_dict_fname, 'multinomial')
    
    print('Done! All plots have been saved into folder \''+os.getcwd()+'\'')
    
    try:
        os.remove(overall_svm_performance_dict_fname)
        os.remove(overall_bayes_performance_dict_fname)
    except OSError:
        pass
