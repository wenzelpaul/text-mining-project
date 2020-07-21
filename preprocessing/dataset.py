"""
This module provides the data structure for an object that contains all relevant data that isneeded for a machine
learning model's interface.
"""

import csv
import numpy as np

import Config


class DataML:
    """
    This class represents a data structure that contains all relevant data that is needed for a machine learning
    model's interface for learning tasks.

    Attributes:
        data (numpy.ndarray):   Stores all feature vectors that represent the sentences of the DRI corpus.
        feature_names (list):   The names of the feature vector attributes.
        target (numpy.ndarray): Stores all annotation labels as unique integer for all input feature vectors.
        target_names (list):    The names of the annotation labels (map of integer to string).
    """

    def __init__(self, path):
        """
        Initializes all parameters by reading out the respective csv-files.
        """
        self.data = np.genfromtxt(path + Config.OUTPUT_FILETYPES[0], delimiter=Config.CSV_CHAR_NEW_COLUMN)

        feature_names = (list(csv.reader(open(path + Config.OUTPUT_FILETYPES[1], 'r'))))[0]
        self.feature_names = feature_names[0].split(Config.CSV_CHAR_NEW_COLUMN)

        self.target = np.genfromtxt(path + Config.OUTPUT_FILETYPES[2], delimiter=Config.CSV_CHAR_NEW_COLUMN)

        target_names = (list(csv.reader(open(path + Config.OUTPUT_FILETYPES[3], 'r'))))[0]
        self.target_names = target_names[0].split(Config.CSV_CHAR_NEW_COLUMN)

    def print_data(self):
        """
        Prints all data parts to the console.
        """
        print(self.data)
        print(self.feature_names)
        print(self.target)
        print(self.target_names)
