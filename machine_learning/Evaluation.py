"""
This module provides the functions of the evaluation framework required to evaluate the different machine learning
algorithms on the multi-layer corpus. The evaluation is performed using randomized hyper-parameter optimization and and
cross-validation.
"""

import os
import time

import itertools

import math
import numpy
from datetime import date
from matplotlib import pyplot
from matplotlib.pyplot import savefig
from prettytable import PrettyTable
from sklearn.metrics import confusion_matrix, classification_report, recall_score, accuracy_score, precision_score, \
    f1_score

from sklearn.model_selection import RandomizedSearchCV


class Algorithm:
    """
    This class represents an algorithm to be evaluated by the evaluation framework. The algorithm has a predefined
    parameter grid for hyper-parameter optimization and also can have more than one implementation.

    Args:
        name (str):                 Algorithm's name.
        parameter_grid (dict):      Parameter grid for randomized hyper-parameter optimization. If no hyper-parameter
                                    optimization and just cross-validation should be performed: None.
        implementations (tuple):    Algorithm's implementations to be evluated.

    Attributes:
        name (str):                 Algorithm's name.
        parameter_grid (dict):      Parameter grid for randomized hyper-parameter optimization. If no hyper-parameter
                                    optimization and just cross-validation should be performed: None.
        implementations (tuple):    Algorithm's implementations to be evluated.
    """

    def __init__(self, name, parameter_grid, *implementations):
        self._name = name
        self._parameter_grid = parameter_grid
        self._implementations = implementations

    def get_name(self):
        return self._name

    def get_parameter_grid(self):
        return self._parameter_grid

    def get_implementations(self):
        return self._implementations


def unweighted_average_f1(y_true, y_pred):
    """
    Calculates the unweighted average ft-score.

    Args:
        y_true (numpy.ndarray[float]): Correct target values.
        y_pred (numpy.ndarray[float]): Predicted target values.

    Returns: float
        Unweighted average f1-score.
    """
    return f1_score(y_true, y_pred, average='macro')


def duration_to_string(duration_s):
    """
    Converts a duration in seconds into a readable string of the format '[[<h>:]<mm>:]<ss> <unit>'.

    Args:
        duration_s (int): Duration in seconds.

    Returns: str
        Human readable duration string.
    """
    duration_s = math.ceil(duration_s)
    duration_unit = 's'
    duration = ''

    if duration_s > 60:
        duration_min = math.floor(duration_s / 60)
        duration_s = duration_s - duration_min * 60
        duration_unit = ' min'

        if duration_min > 60:
            duration_h = math.floor(duration_min / 60)
            duration_min = duration_min - duration_h * 60
            duration_unit = ' h'
            duration = str(duration_h) + ':'

            if duration_min < 10:
                duration += '0'

        duration += str(duration_min) + ':'

        if duration_s < 10:
            duration += '0'

    duration += str(duration_s) + duration_unit
    return duration


def score_to_string(score):
    """
    Limits a score to 2 decimals and converts it into a string.

    Args:
        score (float): Score to convert.

    Returns: str
        Score as string.
    """
    return '%0.2f' % (score)


def plot_confusion_matrix(cm, classes, normalize=False, title=None, cmap=pyplot.cm.Blues, file=None, figsize=None):
    """
    This function returns the confusion matrix as string and plots it into a file or displays it graphically.

    Args:
        cm (numpy.ndarray[numpy.ndarray][int]): confusion matrix
        classes (list[str]):                    target labels
        normalize (bool):                       Enables normalization.
        title (str):                            Title for the plot.
        cmap (matplotlib.colors.LinearSegmentedColormap, default: pyplot.cm.Blues): Color map.
        file (str, default: None):              Filename, where the cm should be stored, if None it will be displayed graphically.
        figsize (tuple[int], default: None):    Size of the plotted figure.

    Returns: str
        Confusion matrix as string.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, numpy.newaxis]
        cm = cm.round(2)

        if title == None:
            title = 'Normalized confusion matrix'
    else:
        if title == None:
            title = 'Confusion matrix without normalization'

    output = "%s\n%s" % (title, cm)

    pyplot.figure(figsize=figsize)
    pyplot.imshow(cm, interpolation='nearest', cmap=cmap)
    pyplot.title(title)
    pyplot.colorbar()
    tick_marks = numpy.arange(len(classes))
    pyplot.xticks(tick_marks, classes, rotation=90)
    pyplot.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        pyplot.text(j, i, cm[i, j],
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")

    pyplot.ylabel('True label')
    pyplot.xlabel('Predicted label')
    pyplot.tight_layout()

    if file == None:
        pyplot.show()
    else:
        savefig(file, format='pdf')

    return output


class RandomizedSearchCVEvaluator:
    """
    Evaluates different algorithms by the given scorer and optimizes their hyper-parameters by randomized search + CV
    with the given hyper-parameter grid.

    Args:
        algorithms (list[Algorithm]):   Algorithms to evaluate.
        n_iter (int):                   Number of randomized search iterations/guesses.
        cv (int):                       Number of cross-validation folds.
        scorer (dict):                  Scorer to evaluate the algorithms.
        output_dir (str):               Directory, where the evaluation results will be stored.

    Attributes:
        algorithms (list[Algorithm]):   Algorithms to evaluate.
        n_iter (int):                   Number of randomized search iterations/guesses.
        cv (int):                       Number of cross-validation folds.
        scorer (dict):                  Scorer to evaluate the algorithms.
        output_dir (str):               Directory, where the evaluation results will be stored.
        scorer_name (str):              Name of the scorer.
        n_cpus (int):                   Number of CPUs for parallelization. -1 stands for all CPUs.
    """

    def __init__(self, algorithms, n_iter, cv, scorer, output_dir):
        self._algorithms = algorithms
        self._n_iter = n_iter
        self._cv = cv
        self._scorer = scorer
        self._scorer_name = next(iter(scorer))

        # windows has parallelization bugs
        if os.name == 'nt':  # windows
            self._n_cpus = 1
        else:
            self._n_cpus = -1  # all CPUs/parallelization

        output_dir += '%s_iter_%i_cv_%i' % (date.today(), n_iter, cv)
        if os.path.exists(output_dir):
            i = 1
            path = '-'.join([output_dir, str(i)])
            while os.path.exists(path):
                i += 1
                path = '-'.join([output_dir, str(i)])
            output_dir = path

        os.makedirs(output_dir)
        self._output_dir = output_dir

    def search(self, layer_name, X, y_true, target_names):
        """
        Executes the randomized search cross-validation with the given parameter grid. The evaluation report and all
        measured metrics will be stored to the output directory. During the search the different computed ML models will
        be compared by the scorer.

        Args:
            layer_name (str):                           Name of the evaluated layer.
            X (numpy.ndarray[numpy.ndarray][float]):    Samples to classify.
            y_true (numpy.ndarray[float]):              Correct target values.
            target_names (list[str]):                   Target names.
        """

        print(layer_name + '-layer...')
        output_dir = self._output_dir + '/' + layer_name + '/'
        os.makedirs(output_dir)
        start_time = time.time()
        layer_report = "Evaluating ML algorithms by '%s', randomized search with %i iterations and %i-fold-CV on layer '%s'" % (
            self._scorer_name, self._n_iter, self._cv, layer_name)
        best_score = 0.0
        best_estimator = None

        for algorithm in self._algorithms:

            # skip incompatible ANN classifiers (number of output neurons must map with the number of layer classes)
            if ('artificial-neural-network' in algorithm.get_name() and not layer_name in algorithm.get_name()):
                continue

            layer_report += '\n\n%s:' % algorithm.get_name()
            table = PrettyTable(
                ['estimator', 'uar', 'war', 'cci', 'ici', 'tin', 'uap', 'wap', 'ua-f1', 'wa-f1', 'hyper-parameters',
                 'duration'])

            for estimator in algorithm.get_implementations():
                search_start = time.time()
                param_distributions = algorithm.get_parameter_grid()

                if param_distributions == None:
                    param_distributions = {}
                    n_iter = 1
                else:
                    n_iter = self._n_iter

                r_search = RandomizedSearchCV(estimator=estimator, param_distributions=param_distributions,
                                              n_iter=n_iter, cv=self._cv, scoring=self._scorer, n_jobs=self._n_cpus,
                                              refit=next(iter(self._scorer)))
                r_search.fit(X, y_true)

                # report scoring steps
                file_path = output_dir + 'guessed-hyperparameters.txt'
                with open(file_path, 'a') as file:
                    for index, param in enumerate(r_search.cv_results_['params']):
                        file.write('%s > %s: %s=%0.2f, hyper-parameters: %s\n' % (
                            algorithm.get_name(), type(estimator).__name__, self._scorer_name,
                            r_search.cv_results_['mean_test_' + self._scorer_name][index], param))

                    file.write('--> standard deviation of the %s-score: %0.4f\n\n' % (
                        self._scorer_name, r_search.cv_results_['mean_test_' + self._scorer_name].std()))

                y_pred = r_search.predict(X)

                # compute evaluation metrics
                metrics = {}
                metrics['uar'] = recall_score(y_true, y_pred, average='macro')
                metrics['war'] = recall_score(y_true, y_pred, average='weighted')

                metrics['cci'] = accuracy_score(y_true, y_pred, normalize=False)
                metrics['cci-sum'] = metrics['cci'].sum()
                metrics['ici'] = len(y_pred) - metrics['cci']
                metrics['ici-sum'] = metrics['ici'].sum()

                metrics['uap'] = precision_score(y_true, y_pred, average='macro')
                metrics['wap'] = precision_score(y_true, y_pred, average='weighted')

                metrics['ua-f1'] = f1_score(y_true, y_pred, average='macro')
                metrics['wa-f1'] = f1_score(y_true, y_pred, average='weighted')

                if best_score < metrics[self._scorer_name]:
                    best_score = metrics[self._scorer_name]
                    best_estimator = r_search.best_estimator_
                    best_hyper_parameters = r_search.best_params_

                # plot scores
                table.add_row([
                    type(estimator).__name__,
                    score_to_string(metrics['uar']),
                    score_to_string(metrics['war']),
                    '%i (%i%%)' % (
                        metrics['cci-sum'],
                        round(100 * metrics['cci-sum'] / (metrics['cci-sum'] + metrics['ici-sum']), 0)),
                    '%i (%i%%)' % (
                        metrics['ici-sum'],
                        round(100 * metrics['ici-sum'] / (metrics['cci-sum'] + metrics['ici-sum']), 0)),
                    len(y_true),
                    score_to_string(metrics['uap']),
                    score_to_string(metrics['wap']),
                    score_to_string(metrics['ua-f1']),
                    score_to_string(metrics['wa-f1']),
                    r_search.best_params_,
                    duration_to_string(time.time() - search_start)
                ])

                file_path = '%s%s_%s_report.txt' % (output_dir, algorithm.get_name(), type(estimator).__name__)
                with open(file_path, 'w') as file:
                    file.write(classification_report(y_true, y_pred, target_names=target_names))

                cm = confusion_matrix(y_true, y_pred)
                file_path = '%s%s_%s_cm.txt' % (output_dir, algorithm.get_name(), type(estimator).__name__)
                cm_path = '%s%s_%s_cm.pdf' % (output_dir, algorithm.get_name(), type(estimator).__name__)
                cm_title = '%s > %s > %s' % (layer_name, algorithm.get_name(), type(estimator).__name__)
                with open(file_path, 'w') as file:
                    file.write(
                        plot_confusion_matrix(cm, target_names, title=cm_title, file=cm_path, figsize=(6.4, 5.5)))
            layer_report += '\n%s' % (table)
            with open(output_dir + 'report.txt', 'a') as file:
                file.write(layer_report)
                layer_report = ''
        layer_report += '\n--> best performing algorithm: %s (%s: %0.2f, hyper-parameters: %s), total evaluation time: %s' % (
            type(best_estimator).__name__, self._scorer_name, best_score, best_hyper_parameters,
            duration_to_string(time.time() - start_time))
        with open(output_dir + 'report.txt', 'a') as file:
            file.write(layer_report)
