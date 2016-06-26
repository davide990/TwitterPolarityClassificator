import ClassifierEvaluation

if __name__ == "__main__":
    evaluator = ClassifierEvaluation.ClassifierEvaluation()
    evaluator.get_overall_performance_dict('Dati/performance/for_plots.json', 'Dati/performance')

    evaluator.plot_overall_precision('svm_linear.png', 'Dati/performance/for_plots/for_plots.json', 'linear')
    evaluator.plot_overall_precision('svm_rbf.png', 'Dati/performance/for_plots/for_plots.json', 'rbf')