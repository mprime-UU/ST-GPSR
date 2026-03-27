"""Explicit Symbolic Regression

Explicit symbolic regression is the search for a function, f, such that
f(x) = y.

The classes in this module encapsulate the parts of bingo evolutionary analysis
that are unique to explicit symbolic regression. Namely, these classes are an
appropriate fitness evaluator and a corresponding training data container.
"""
import logging
import numpy as np

from ..evaluation.fitness_function import VectorBasedFunction
from ..evaluation.training_data import TrainingData

LOGGER = logging.getLogger(__name__)


class ExplicitRegressionMD(VectorBasedFunction):
    """ExplicitRegression

    The traditional fitness evaluation for symbolic regression

    Parameters
    ----------
    training_data : ExplicitTrainingData
        data that is used in fitness evaluation.
    metric : str
        String defining the measure of error to use. Available options are:
        'mean absolute error', 'mean squared error', and
        'root mean squared error'
    relative : bool
        Whether to use relative, pointwise normalization of errors. Default:
        False.
    """
    def __init__(self, training_data, metric="mae", relative=False, relative_type="division"):
        super().__init__(training_data, metric)
        self._relative = relative
        self._relative_type = relative_type

    def evaluate_fitness_vector(self, individual):
        """ Traditional fitness evaluation for symbolic regression

        fitness = y - f(x) where x and y are in the training_data (i.e.
        training_data.x and training_data.y) and the function f is defined by
        the input Equation individual.

        Parameters
        ----------
        individual : agraph
            individual whose fitness is evaluated on `training_data`
        """
        self.eval_count += 1
        f_of_x = individual.evaluate_equation_at(self.training_data.x)

        assert f_of_x.shape == self.training_data.y.shape
        error = f_of_x - self.training_data.y
        assert error.shape == self.training_data.y.shape
        if not self._relative:
            return error.flatten()

        if self._relative_type == "norm":
            y_ndim = self.training_data.y.ndim
            non_data_axes = tuple(range(1, y_ndim))
            y_norm = np.linalg.norm(self.training_data.y, axis=non_data_axes)

            # pad with extra dimensions for broadcasting
            y_norm = y_norm.reshape((-1, *np.ones(y_ndim - 1, dtype=int)))

            rel_err = error / y_norm
        elif self._relative_type == "percent difference":
            rel_err = 2.0 * error / (np.abs(f_of_x) + np.abs(self.training_data.y))  # RPD
            both_zero = (f_of_x == 0) & (self.training_data.y == 0)
            rel_err[both_zero] = 0
        else:  # relative type == division
            rel_err = error / self.training_data.y
        assert rel_err.shape == self.training_data.y.shape
        return rel_err.flatten()

    def get_fitness_vector_and_jacobian(self, individual):
        self.eval_count += 1
        f_of_x, df_dc = \
            individual.evaluate_equation_with_local_opt_gradient_at(
                    self.training_data.x)
        error = f_of_x - self.training_data.y
        if not self._relative:
            return error.flatten(), df_dc
        return (error / self.training_data.y).flatten(), \
            df_dc / self.training_data.y


class ExplicitTrainingDataMD(TrainingData):
    """
    ExplicitTrainingData: Training data of this type contains an input array of
    data (x)  and an output array of data (y).  Both must be 2 dimensional
    numpy arrays

    Parameters
    ----------
    x : 2D numpy array
        independent variable
    y : 2D numpy array
        dependent variable
    """
    def __init__(self, x, y):
        self._x = x
        if y.ndim == 1:
            y = y.reshape((-1, 1, 1))
        self._y = y

    @property
    def x(self):
        """independent x data"""
        return self._x

    @property
    def y(self):
        """dependent y data"""
        return self._y

    def __getitem__(self, items):
        """gets a subset of the `ExplicitTrainingData`

        Parameters
        ----------
        items : list or int
            index (or indices) of the subset

        Returns
        -------
        `ExplicitTrainingData` :
            a Subset
        """
        temp = ExplicitTrainingDataMD(self._x[items, :], self._y[items, :])
        return temp

    def __len__(self):
        """ gets the length of the first dimension of the data

        Returns
        -------
        int :
            index-able size
        """
        return self._x.shape[0]
