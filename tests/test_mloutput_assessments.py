""" module to test mloutput assessments."""

import numpy as np
from duckie.mloutput_assessments import ConfusionMatrixAssessment
import os.path
import pandas as pd


def test_cma():
    """Test case for the confusion matrix assessment.
    """

    confusion_matrix = np.load('tests/test_data/test_confusion_matrix.npy')

    CMA = ConfusionMatrixAssessment(confusion_matrix, 'TestML')

    # We know the correct values for this matrix
    summary_data_true = {'Model Name': 'TestML', 'recall': 0.95,
                         'precision': 0.926829268292683, 'f1_score': 0.9382716049382716}

    # Test the summarize function

    summary_data = CMA.summarize()

    for stat in summary_data.columns:
        assert summary_data[stat].values[0] == summary_data_true[
            stat], f"{stat} incorrect, getting {summary_data[stat].values[0]}, should be \
                {summary_data_true[stat]}"

    # Test the get_distribution function

    model_name = np.array(['TestML', 'TestML', 'TestML', 'TestML'])
    category = np.array(['TN', 'FP', 'FN', 'TP'])
    true_counts = np.array([77, 3, 2, 38])
    true_fractions = np.array([0.64166667, 0.025, 0.01666667, 0.31666667])
    true_percentages = np.array([64.16666667, 2.5, 1.66666667, 31.66666667])

    distribution_data_true = {
        'Model Name': model_name,
        'category': category,
        'counts': true_counts,
        'fractions': true_fractions,
        'percentages': true_percentages}

    distribution_data = CMA.distribute()

    for stat in ['Model Name', 'category', 'counts']:
        assert np.array_equal(distribution_data[stat].values, distribution_data_true[stat]
                              ), f"{stat} incorrect, getting {distribution_data[stat].values}, \
                                should be {distribution_data_true[stat]}"

    for stat in ['fractions', 'percentages']:
        assert np.allclose(distribution_data[stat].values, distribution_data_true[stat]
                           ), f"{stat} incorrect, getting {distribution_data[stat].values}, \
                            should be {distribution_data_true[stat]}"

    # Test the plot function

    CMA.plot('confusion_matrix.png', 'Confusion Matrix')
    assert os.path.isfile('confusion_matrix.png'), "No plot generated."


if __name__ == "__main__":
    test_cma()
