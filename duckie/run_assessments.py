# Code to run our assessments at various stages of the ML lifecycle

from duckie.ml_tools import runML


class RunAssessments:
    """
    A class to run the assessments on a ML algorithm operating on a test set or new data.
    """

    def runDataAssessments(self, ml=None, y_pred=None, runml=False):

        if ml is None and y_pred is None:
            raise RuntimeError('Please either pass a pickled ml file to run or a prediction.')

        elif y_pred is None and ml is not None:
            y_pred = runML(ml)

        return
