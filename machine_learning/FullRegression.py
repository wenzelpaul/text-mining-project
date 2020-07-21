"""
This module executes a full regression analysis on the summary layer.
"""

from machine_learning.Regression import support_vector_regression
from preprocessing.Dataset import DataML
import Config


class FullRegression:
    """
    This class provides the execute method to run the regression.
    """

    def execute(self):
        """
        Execute the regression analysis on the summary layer
        """
        csv_path = Config.OUTPUT_PATHS[3]  # the summary layer
        dataset = DataML(csv_path)
        support_vector_regression(dataset)
