# Useful classes to train test and run a ML algorithm

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class ML:
    """
    A class to store a ML algorithm, basically a sklearn model but saves some information and
    the training data.
    """

    def __init__(self, name, skml, training_data=None, trained=False):

        """
        Attributes:
        name (str): Name for the model.
        skml (sklearn algorithm): A sklearn machine learning algorithm.
        training_data (dic of pd.DataFrames): The training data for the algorithm. 'X': X_train,
        'Y': Y_train TODO: might want to make this a little more user friendly
        trained (Bool): True if skml has been trained, False if not.
        """

        self.name = name
        self.skml = skml
        self.training_data = training_data
        self.trained = trained


class trainML:
    """
    A class to automatically train a ML algorithm.
    """

    def __init__(self, ml):
        self.ml = ml

    def train(self):
        return


class testML:
    """
    A class to automatically train a ML algorithm.
    """

    def __init__(self, ml):
        self.ml = ml

    def test(self):
        return


class runML:
    """
    A class to run a trained ML algorithm on new data.
    """

    def __init__(self, ml, X_data):
        """
        Attributes:
        ml (duckie.ML): duckie ML object.
        X_data (pd.DataFrame): X data.
        """

        self.ml = ml
        self.X_data = X_data

    def run(self, sample_size=1.0, sampling='random'):
        """ Run a trained ML algorithm on new data.

            Parameters:
                ml (pickle file): Pickled sklearn machine learning algorithm.
                X_data (pd.DataFrame): Pandas dataframe of the X data to run the algorithm on.
            Returns:
                y_pred (pd.dataframe): Pandas dataframe of the prediction values.
        """

        # This is mostly for demo purposes
        if sample_size != 1.0:
            if sampling == 'random':
                X_data = self.X_data.sample(frac=sample_size)
            else:
                raise NotImplementedError('Non random sampling not implemented yet.')

        y_pred = self.ml.skml.predict(X_data.values)
        y_pred = pd.DataFrame(y_pred, columns=self.ml.training_data['Y'].columns)

        return y_pred
