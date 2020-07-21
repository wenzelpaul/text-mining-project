"""
This module provides the full evaluation. It is initialized with the machine learning algorithms (kNN, decision tree,
naive bayes, SVM, ANN) to evalaluate, their defined hyper-parameter ranges, and their different implementations.
By calling the execute method the randomized search cross-validation is executed with the parameters (randomized search
iteration number, cross validation folds, output directory) defined in the config file and its results (convolution
matrices, reports) are stored in a separate folder for each evaluated corpus layer in the defined output directory.
"""

import warnings

import time
from sklearn.exceptions import UndefinedMetricWarning

import re

from scipy.stats import uniform, randint

from sklearn.metrics import make_scorer
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense

import Config
from machine_learning import Evaluation
from machine_learning.Evaluation import Algorithm, RandomizedSearchCVEvaluator, unweighted_average_f1

from preprocessing.Dataset import DataML


class FullEvaluation:
    """
    Performs an evaluation of the defined machine learning algorithms on the pre-processed multi-layer corpus using
    randomized search cross-validation.

    Attributes:
        algorithms (list[Algorithm]):           Algorithms to evaluate on the corpus.
        n_classes (int):                        The current layer's class number.
        r_search (RandomizedSearchCVEvaluator): The randomized search cross-validation evaluator.
    """

    def __init__(self):
        # ignore undefined metric warnings, if no TPs can be predicted for a class.
        warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

        self._algorithms = [
            Algorithm('k-nearest-neighbors', None, KNeighborsClassifier(n_neighbors=3)),
            Algorithm('decision-tree', {'max_depth': randint(3, 30)}, DecisionTreeClassifier(random_state=0)),
            Algorithm('naive-bayes', None, GaussianNB(), MultinomialNB(), BernoulliNB()),
            Algorithm('svm-linear', {
                'C': uniform(0, 1)
            }, SVC(kernel='linear'), LinearSVC()),
            Algorithm('svm-rbf', {
                'gamma': uniform(0, 1),
                'C': (0.0000000001, 1)
            }, SVC(kernel='rbf')),
            Algorithm('svm-poly', {
                'degree': uniform(2, 5),
                'C': uniform(0, 1),
                'coef0': uniform(0, 1)
            }, SVC(kernel='poly'))
            # ANN estimator is created during runtime (number of output classes needed for network creation)
        ]

        self._n_classes = 0  # parameter for ANN estimator creation (corresponds to number of neurons in output layer)

        scorer = {'ua-f1': make_scorer(unweighted_average_f1)}
        self._r_search = RandomizedSearchCVEvaluator(self._algorithms, Config.RANDOMIZED_SEARCH_ITERATIONS,
                                                     Config.CROSS_VALIDATION_FOLDS, scorer, Config.EVALUATION_DIR)

    def _baseline_model(self):
        """
        Defines the baseline model for the ANN estimator.

        Returns:
            The model.
        """
        # create model
        model = Sequential()
        num_feature_input = 3 + Config.MAX_FEATURES_UNIGRAM + Config.MAX_FEATURES_BIGRAM + Config.MAX_FEATURES_TRIGRAM
        model.add(Dense(8, input_dim=num_feature_input, activation='relu'))
        model.add(Dense(self._n_classes, activation='softmax'))
        # Compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def execute(self):
        """
        Executes the evaluation of all algorithms on the whole corpus, measures the execution time and stores the
        evaluation results to the evaluation directory.
        """

        evaluation_time = time.time()

        print('Evaluation started:')

        for csv_path in Config.OUTPUT_PATHS:
            dataset = DataML(csv_path)
            layer_name = re.search('^.*/([aA-zZ]+?)/$', csv_path)

            if (layer_name):
                layer_name = layer_name.group(1)
            else:
                layer_name = csv_path

            # add Keras ANN classifiers during runtime according to the number of output classes
            self._n_classes = len(dataset.target_names)
            model = KerasClassifier(build_fn=self._baseline_model, nb_epoch=200, batch_size=5, verbose=0)
            self._algorithms.append(Algorithm("artificial-neural-network_" + str(layer_name), None,
                                              KerasClassifier(build_fn=self._baseline_model, nb_epoch=200, batch_size=5,
                                                              verbose=0)))

            self._r_search.search(layer_name, dataset.data, dataset.target, dataset.target_names)

        print('Evaluation finished!!! Total duration: ' + Evaluation.duration_to_string(time.time() - evaluation_time))
