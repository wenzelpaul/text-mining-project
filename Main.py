"""
This module combines all steps together and executes corpus loading, preprocessing, application and evaluation of
machine learning techniques and regression.
"""

import preprocessing.FullPreprocessing as pp
import machine_learning.FullEvaluation as ml
import machine_learning.FullRegression as rg

# load DRI corpus, execute all preprocessing steps and store result in csv files
preprocessing = pp.FullPreprocessing()
preprocessing.execute()

# evaluate k-nearest-neighbors, decision-tree, naive-bayes, SVM and ANN on the pre-processed multi-layer corpus.
evaluation = ml.FullEvaluation()
evaluation.execute()

# evaluate support vector regression
regression = rg.FullRegression()
regression.execute()
