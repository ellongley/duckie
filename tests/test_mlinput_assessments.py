""" module to test mlinput assessments."""

import pandas as pd
import pytest
from duckie.mlinput_assessments import TrainTestStratAssessment


def test_stratification():

    # Check that we get a warning on the non-stratified data

    no_strat_train = pd.read_pickle('tests/test_data/no_strat_train')
    no_strat_test = pd.read_pickle('tests/test_data/no_strat_test')

    TTSA = TrainTestStratAssessment(no_strat_train, no_strat_test, ['virginica'])
    with pytest.warns(Warning) as record:
        TTSA.assess()
        assert len(record) == 1, "Expected a warning on the stratified data!"

    strat_train = pd.read_pickle('tests/test_data/strat_train')
    strat_test = pd.read_pickle('tests/test_data/strat_test')

    # Check that we don't get a warning on the properly stratified data

    TTSA = TrainTestStratAssessment(strat_train, strat_test, ['virginica'])
    with pytest.warns(None) as record:
        TTSA.assess()
        assert len(record) == 0, "Expected no warnings on the stratified data, but got a warning!"


if __name__ == "__main__":
    test_stratification()
