# Assessment tasks to run when a ML algorithm is being deployed on new data.

import numpy as np
import warnings
import matplotlib.pyplot as plt


# TODO! Change this to inherit from the train test strat class

# TODO! Change to the threshold comparison to still work when the indices don't match
# And work for multiclasses


class DataStratAssessment:
    """
    Check that the distribution of a test set is comparable to that of the
    training set for a list of categorical features of the data.
    """

    def __init__(self, train, test, features, coltype='Feature', datatype='Test', threshold=10.0):
        """
        Attributes:
        train (pd.DataFrame): The training data.
        test (pd.DataFrame): The test data.
        parameters (list): List of parameters to check.
        """
        self.train = train
        self.test = test
        self.features = features
        self.coltype = coltype
        self.datatype = datatype
        self.threshold = threshold

    def assess(self):
        """ Compute the distribution of the class types in the train and test set.
        Returns:
            norm_counts_test_full (list of pd.series): List of the normalized counts of the
            instances of each feature type for each feature in the test set.
            norm_counts_train_full (list of pd.series): List of the normalized counts of the
            instances of each feature type for each feature in the training set.
        """
        norm_counts_train_full = []
        norm_counts_test_full = []
        for feature in self.features:
            norm_counts_train = self.train[feature].value_counts(normalize=True)
            norm_counts_test = self.test[feature].value_counts(normalize=True)

            norm_counts_train_full.append(norm_counts_train)
            norm_counts_test_full.append(norm_counts_test)

            print(f'Checking feature {feature}.')

            if len(norm_counts_train) != len(norm_counts_test):
                # Warn the user if we have a different number of feature types
                # in the test data than the train data.
                warning_text = f'Warning! There are {len(norm_counts_train)} feature types of \
                    feature {feature} in the training data, but there are {len(norm_counts_test)}\
                    feature types of {feature} in the test data!'
                warnings.warn(warning_text)

            for ind in range(len(norm_counts_train)):
                feature_type = norm_counts_train.index[ind]
                print(
                    (f'Training sample is {"{0:.2%}".format(norm_counts_train[feature_type])}\
                     {feature_type} for {self.coltype} {feature}.'))
            for ind in range(len(norm_counts_test)):
                feature_type = norm_counts_test.index[ind]
                print(
                    (f'{self.datatype} sample is {"{0:.2%}".format(norm_counts_test[feature_type])}\
                     {feature_type}: for {self.coltype} {feature}.'))

            diff = abs(norm_counts_test[feature_type] - norm_counts_test[feature_type])
            if diff < self.threshold:
                print('This is a similiar distribution compared to the training data!')
            else:
                print('Warning! This is a pretty different distribution compared to the training!')

        return norm_counts_train_full, norm_counts_test_full
