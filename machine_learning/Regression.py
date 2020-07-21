"""
This module provides a full regression analysis. It is using scikit-learns Support Vector Regression and three different
kernels: RBF-Kernel, Linear-Kernel and Polynomial-Kernel. First cross validation is used on the scoring function
provided by SVR. Then 10 fold cross validation calculates the average of correct predicted target values and close
predictions up to a distance of 0.5 of the true target value. A graph is additionally drawn for each test-set.
The results are saved to the path provided by Config.EVALUATION_DIR
"""

import csv
from matplotlib import pyplot
from matplotlib.pyplot import savefig
import numpy as np
from datetime import date
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
import os

import Config

VERTICAL_SEPARATOR = '----------------------------------------------------------'
global feature_mapping
global dataset


def support_vector_regression(data):
    """
    Main method for support vector regression. Trains a RBF-Model, Polynomial-Model and Linear-Model.

    Args:
        data (DataML): The dataset to perform the regression on.
    """
    global dataset
    dataset = data
    global feature_mapping
    feature_mapping = create_dict()

    # Configure regression model
    svr_rbf = SVR(kernel='rbf', C=1, gamma=0.5, epsilon=0.165, cache_size=1000)
    svr_lin = SVR(kernel='linear', C=1, epsilon=0.165, cache_size=1000)
    svr_poly = SVR(kernel='poly', C=100000, degree=2, epsilon=0.165, cache_size=1000)

    output_dir = create_output_dir()

    results = ''
    results += calculate_crossvalidation(svr_lin, svr_poly, svr_rbf, Config.CV_FOLDS_REGRESSION)

    rbf_count_exact = []
    rbf_count_close = []
    poly_count_exact = []
    poly_count_close = []
    lin_count_exact = []
    lin_count_close = []
    test_set_start = Config.TRAIN_SET_START
    while test_set_start <= Config.TEST_SET_STOP:
        if test_set_start >= Config.TEST_SET_STOP:
            test_set_end = Config.TRAIN_SET_END
        else:
            test_set_end = test_set_start + Config.TEST_SET_SIZE

        X = np.concatenate(
            [data.data[Config.TRAIN_SET_START:test_set_start], data.data[test_set_end:Config.TRAIN_SET_END]])
        y = np.concatenate(
            [data.target[Config.TRAIN_SET_START:test_set_start], data.target[test_set_end:Config.TRAIN_SET_END]])
        X_test = data.data[test_set_start:test_set_end]
        y_test = data.target[test_set_start:test_set_end]

        # Do the mapping for target values
        do_mapping(y_test)
        do_mapping(y)

        rbf = svr_rbf.fit(X, y)
        lin = svr_lin.fit(X, y)
        poly = svr_poly.fit(X, y)

        score_rbf = svr_rbf.score(X_test, y_test)
        score_poly = svr_poly.score(X_test, y_test)
        score_lin = svr_lin.score(X_test, y_test)

        y_rbf_predicted = rbf.predict(X_test)
        y_lin_predicted = lin.predict(X_test)
        y_poly_predicted = poly.predict(X_test)

        results += add_to_results(test_set_end, test_set_start)
        count_rbf_exact, count_rbf_close, results = calculate_metrics(y_test, y_rbf_predicted, "RBF-Kernel", score_rbf,
                                                                      results)
        count_poly_exact, count_poly_close, results = calculate_metrics(y_test, y_poly_predicted, "Poly-Kernel",
                                                                        score_poly, results)
        count_lin_exact, count_lin_close, results = calculate_metrics(y_test, y_lin_predicted, "Linear-Kernel",
                                                                      score_lin, results)
        rbf_count_exact.append(count_rbf_exact)
        rbf_count_close.append(count_rbf_close)
        poly_count_exact.append(count_poly_exact)
        poly_count_close.append(count_poly_close)
        lin_count_exact.append(count_lin_exact)
        lin_count_close.append(count_lin_close)

        graph_dir = '%s/%s_%s_predicted_graph.png' % (output_dir, test_set_start, test_set_end)
        x_axis = np.arange(test_set_start, test_set_end)
        draw_results(y_test, y_lin_predicted, y_poly_predicted, y_rbf_predicted, x_axis, graph_dir)

        test_set_start = test_set_start + Config.TEST_SET_SIZE

    results = add_counts_to_results(lin_count_close, lin_count_exact, poly_count_close, poly_count_exact,
                                    rbf_count_close, rbf_count_exact, results)

    # save file
    with open(output_dir + "/scoring_results.txt", 'w') as file:
        file.write(results)


def create_dict():
    """
    Creates a dictionary and maps its values to the correct target values.

    Returns: dict
        Returns the dictionary.
    """
    target_mapping = dict()
    target_names = (list(csv.reader(open(Config.OUTPUT_PATHS[3] + Config.OUTPUT_FILETYPES[3], 'r'))))[0]
    target_names = target_names[0].split(Config.CSV_CHAR_NEW_COLUMN)
    for i in range(0, len(target_names)):
        target_mapping[i] = target_names[i]
    target_mapping.update({0: 0.0})  # Replace NONE value with 0.0
    return target_mapping


def create_output_dir():
    """
    Creates a unique directory to store the results in.

    Returns: str
        The path to the directory as a string.
    """
    output_dir = Config.EVALUATION_DIR + '%s_summary_regression' % (date.today())
    if os.path.exists(output_dir):
        i = 1
        path = '-'.join([output_dir, str(i)])
        while os.path.exists(path):
            i += 1
            path = '-'.join([output_dir, str(i)])
        output_dir = path
    os.makedirs(output_dir)
    return output_dir


def calculate_crossvalidation(svr_lin, svr_poly, svr_rbf, cv_folds):
    """
    Calculate the average score with the built in scoring-function using cross validation.

    Args:
        svr_lin (SVR):  The preconfigured linear model.
        svr_poly (SVR): The preconfigured polynomial model.
        svr_rbf (SVR):  The preconfigured rbf model.
        cv_folds (int): The amount of folds used for cross validation.

    Returns: str
        The results of the cross validation.
    """
    X = dataset.data
    y = dataset.target
    do_mapping(y)
    score_rbf = cross_val_score(svr_rbf, X, y, cv=cv_folds)
    score_poly = cross_val_score(svr_poly, X, y, cv=cv_folds)
    score_lin = cross_val_score(svr_lin, X, y, cv=cv_folds)

    result = ''
    result += add_cv_to_results(score_rbf, 'RBF-Kernel Scores   ')
    result += add_cv_to_results(score_poly, 'Poly-Kernel Scores  ')
    result += add_cv_to_results(score_lin, 'Linear-Kernel Scores')
    return result


def add_cv_to_results(score, kernel):
    """
    Creating a string with the specific results of one model.

    Args:
        score (array):  Array of scores for one model.
        kernel (str):   The specific model that the scores are from.

    Returns: str
        The string representation of cross validation from one model.
    """
    result = kernel + ': [ ' + '; '.join(
        score_to_string(item) for item in score) + ' ] | Mean: ' + score_to_string(
        np.mean(score)) + Config.CSV_CHAR_NEW_ROW
    return result


def do_mapping(y):
    """
    Map the target values to its original values.

    Args:
        y (array):  The target values to be updated.
    """
    i = 0
    for target in y:
        if target in feature_mapping:
            y[i] = feature_mapping[target]
        i = i + 1


def calculate_metrics(y_true, y_predicted, kernel_name, score, results_text):
    """
    Calculate mean, variance and standard deviation.

    Args:
        y_true (array):         The true target values.
        y_predicted (array):    The predicted target values.
        kernel_name (str):      The kernel name.
        score (float):          The score by the built in scoring function.
        results_text (str):     Containing previous results.

    Returns:
        count_exact (int):      The amount of exact predictions.
        count_close (int):      The amount of exact and close predictions.
        results_text (str):     The updated results.
    """
    diff = np.subtract(y_true, y_predicted)
    diff = np.absolute(diff)
    mean = np.mean(diff)
    var = np.var(diff)
    standard_deviation = np.math.sqrt(var)
    count_exact = 0
    count_close = 0
    for distance in diff:
        if distance <= 0.165:  # difference where a prediction can still be counted as correct
            count_exact = count_exact + 1
        if distance <= 0.5:  # including the direct neighbors of a true value
            count_close = count_close + 1

    results_text += '\n' + kernel_name + " score: " + score_to_string(score) + Config.CSV_CHAR_NEW_ROW
    results_text += "Mean: " + score_to_string(mean) + ", Variance: " + score_to_string(
        var) + ", Standard deviation: " + score_to_string(standard_deviation) + Config.CSV_CHAR_NEW_ROW
    results_text += kernel_name + " correct ones : " + str(count_exact / Config.TEST_SET_SIZE) + Config.CSV_CHAR_NEW_ROW
    results_text += kernel_name + " close ones : " + str(count_close / Config.TEST_SET_SIZE) + '\n'
    results_text += VERTICAL_SEPARATOR
    return count_exact, count_close, results_text


def draw_results(y, y_lin, y_poly, y_rbf, x_axis, graph_dir):
    """
    Draw the test-results and save them.

    Args:
        y (array):          The true target values.
        y_lin (array):      Predicted target values by linear kernel.
        y_poly (array):     Predicted target values by polynomial kernel.
        y_rbf (array):      Predicted target values by RBF kernel.
        x_axis (array):     The X-axis, from test-set start to test-set end.
        graph_dir (str):    The path to store the graph at.
    """
    pyplot.close()
    lw = 1
    pyplot.scatter(x_axis, y, color='darkorange', label='data')
    pyplot.plot(x_axis, y_rbf, color='red', lw=lw, label='RBF model')
    pyplot.plot(x_axis, y_lin, color='c', lw=lw, label='Linear model')
    pyplot.plot(x_axis, y_poly, color='cornflowerblue', lw=lw, label='Polynomial model')
    pyplot.xlabel('data')
    pyplot.ylabel('target')
    pyplot.title('Support Vector Regression')
    pyplot.legend()
    savefig(graph_dir, format='png')


def add_counts_to_results(lin_count_close, lin_count_exact, poly_count_close, poly_count_exact, rbf_count_close,
                          rbf_count_exact, results):
    """
    Adds all correct and close counts to the result file.

    Args:
        lin_count_close (list): Containing all close/neighboring counts for the linear model.
        lin_count_exact (list): Containing all correct counts for the linear model.
        poly_count_close (list):Containing all close/neighboring counts for the polynomial model.
        poly_count_exact (list):Containing all correct counts for the polynomial model.
        rbf_count_close (list): Containing all close/neighboring counts for the RBF model.
        rbf_count_exact (list): Containing all correct counts for the RBF model.
        results (str):          Containing previous results.

    Returns: str
        Updated string-results with all counts.
    """
    results += '\n\n'
    results += add_count_to_results('RBF correct   ', rbf_count_exact)
    results += add_count_to_results('RBF close     ', rbf_count_close)
    results += add_count_to_results('Poly correct  ', poly_count_exact)
    results += add_count_to_results('Poly close    ', poly_count_close)
    results += add_count_to_results('Linear correct', lin_count_exact)
    results += add_count_to_results('Linear close  ', lin_count_close)
    return results


def add_count_to_results(type, counts):
    """
    Adds a specific list of counts to results.

    Args:
        type (str):     The specific model the counts belong to.
        counts (list):  The list with the counts

    Returns: str
        The results as string.
    """
    results = type + ": [ " + '; '.join(
        str(item / Config.TEST_SET_SIZE) for item in counts) + ' ] | Mean: ' + score_to_string(
        np.mean(counts) / Config.TEST_SET_SIZE) + Config.CSV_CHAR_NEW_ROW
    return results


def add_to_results(test_set_end, test_set_start):
    """
    Add a line of Test-Set start and end to the results.

    Args:
        test_set_end (int):     The ending position of the Test-Set.
        test_set_start (int):   The starting position of the Test-Set.

    Returns: str
        A line of Test-Set start and end.
    """
    results = '\n\n######### Test-Set ' + str(test_set_start) + '-' + str(test_set_end) + ' #########'
    return results


def score_to_string(score):
    """
    Limits a score to 4 decimals and converts it into a string.

    Args:
        score (float): Score to convert.

    Returns: str
        Score as string.
    """
    return '%0.4f' % (score)
