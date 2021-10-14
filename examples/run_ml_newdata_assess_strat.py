""" Module to run a pickled trained sklearn ML model on new data, and asses the
distribution of the target class compared to the that of the training sample."""

import numpy as np
import os.path
import pandas as pd
import pickle

from duckie.ml import runML
from duckie.mldata_assessments import DataStratAssessment


def new_data_target_var_distribution(ml_filenamel,X_data,sample_size=1.0,sampling='random'):
    """ Run a pickled ML algorithm and compare the distribution of the output
    target variables to that of the training data.
    """

    ml = pickle.load( open( ml_filename, "rb" ) )

    rML = runML(ml,X_data)

    y_pred = rML.run(sample_size=sample_size,sampling=sampling)

    DSA = DataStratAssessment(ml.training_data['Y'], y_pred, \
    ['virginica'], coltype='Target', datatype='New Data')

    DSA.assess()


if __name__ == "__main__":

    ml_filename = '../tests/test_data/KNN/ml'

    X_data = pd.read_pickle('../tests/test_data/KNN/X_test')

    new_data_target_var_distribution(ml_filename,X_data,sample_size=0.5)
