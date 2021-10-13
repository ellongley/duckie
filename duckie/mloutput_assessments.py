import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class ConfusionMatrixAssessment:
    """
    A class to perform some useful assessments of a confusion matrix such as:
        - Create a heatmap with some useful quantities displayed.
        - Compute some summary statistics, (e.g. FP, TP, FN).
        - Compute the distribution of distribution across categories.
    """

    def __init__(self, confusion_matrix, model_names):
        """
        Attributes:
        confusion_matrix (np.ndarray or list of np.ndarrays): A confusion matrix for a
        classification algorithm.
        """
        if isinstance(confusion_matrix, np.ndarray):
            self.confusion_matrices = [confusion_matrix]
        elif isinstance(confusion_matrix, list):
            self.confusion_matrices = confusion_matrix
        else:
            raise TypeError('Whoops! Please pass either a list of np.ndarrays or a np.ndarray.')
        if isinstance(model_names, str):
            self.model_names = [model_names]
        else:
            self.model_names = model_names

    def summarize(self):
        """ Compute some summary information from the confusion matrix.

            Parameters:
                confusion_matrix (np.ndarray): Confusion matrix.
            Returns:
                summary_data (dic): Summary data about the matrix.
        """

        # first compute some basic useful quantities

        # TODO: add some quantities that are useful for multiclass matrices (cross-entropy)

        # if there are only two features, seperately store the FP / FN etc.
        def get_summary_data(matrix):

            if matrix.shape == (2, 2):
                FP = matrix[0, 1]
                FN = matrix[1, 0]
                TP = matrix[1, 1]

                recall = TP / (FN + TP)
                precision = TP / (FP + TP)

                f1_score = (2 * precision * recall) / (precision + recall)

                summary_data = pd.DataFrame({'precision': [precision],
                                             'recall': [recall], 'f1_score': [f1_score]})

                return summary_data
            else:
                print('Apologies! We can only compute summary data for two classes for now.')
                return {}

        sum_data_full = pd.concat([get_summary_data(matrix)
                                  for matrix in self.confusion_matrices], ignore_index=True)
        sum_data_full.insert(0, 'Model Name', self.model_names)
        return sum_data_full

    def distribute(self):
        """ Compute the distribution across cases.
        Parameters:
            confusion_matrix (np.ndarray): Confusion matrix.
        Returns:
            distribution_data (dic): Counts, fractions and percentages of cases.
        """

        def get_dist_data(matrix):
            # compute the distribution of the distribution

            counts = matrix.flatten()

            fractions = counts / np.sum(matrix)
            percentages = fractions * 100.

            # TODO need to fix this to work for higher order matrices

            if matrix.shape == (2, 2):
                self.cats = ['TN', 'FP', 'FN', 'TP']
            else:
                raise NotImplementedError("Whoops! Can't do this for higher order matrices yet.")

            dist_data = pd.DataFrame({'category': self.cats, 'counts': counts,
                                     'fractions': fractions, 'percentages': percentages})

            return dist_data

        dist_data_full = pd.DataFrame({})
        for i in range(len(self.confusion_matrices)):
            dist_data = get_dist_data(self.confusion_matrices[i])
            dist_data.insert(0, 'Model Name', self.model_names[i])
            dist_data_full = dist_data_full.append(dist_data, ignore_index=True)

        return dist_data_full

    def plot(self, save_name=None,
             title=None, cmap=None, figsize=(8, 8)):
        '''
        Plot a nice sklearn Confusion Matrix.  Display and optionally save as a
        .png.

            Parameters:
                save_name (str): Name of the figure to save as a png. default = None, automatic.
                title (str or list of strs): Title or multiple Titles, if None use model_names.
                cmap (str): cmap for the sns heatmap.
                figsize (tuple): Figure size.  default = (8,8).

        '''

        def make_plot(matrix, distribution, save_name, title, cmap, figsize):

            count_labels = ["{0:0.0f}\n".format(value) for value in distribution['counts']]
            percentage_labels = ["{0:.2%}".format(value) for value in distribution['fractions']]

            # TODO add an option to put TN etc. on plot

            # if matrix.shape == (2, 2):
            #    result_labels = ["TN", "FP", "FN", "TP"]

            mshape = matrix.shape

            labels = [f"{v1}{v2}".strip() for v1, v2, in zip(count_labels, percentage_labels)]
            labels = np.asarray(labels).reshape(mshape[0], mshape[1])

            plt.figure(figsize=figsize)

            # Make a Seaborn heatmap
            sns.heatmap(matrix, annot=labels, fmt="", cmap=cmap, cbar=False)

            plt.ylabel('True label')
            plt.xlabel('Predicted label')

            plt.title(title)
            plt.show()

            plt.savefig(save_name)

        if isinstance(save_name, str):
            save_name = [save_name]
        elif save_name is None:
            save_name = [f'{model_name}_confusion_matrix.png' for model_name in self.model_names]
        if isinstance(title, str):
            title = [title]
        elif title is None:
            title = [f'{model_name} Confusion Matrix' for model_name in self.model_names]
        if isinstance(cmap, str):
            cmap = [cmap]
        elif cmap is None:
            cmap = [cmap] * len(self.model_names)
        if isinstance(figsize, tuple):
            figsize = [figsize] * len(self.model_names)

        dist_data_full = self.distribute()

        for i in range(len(self.confusion_matrices)):
            model_data = dist_data_full[dist_data_full['Model Name'] == self.model_names[i]]
            make_plot(
                self.confusion_matrices[i],
                model_data,
                save_name[i],
                title[i],
                cmap[i],
                figsize[i])
