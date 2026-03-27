"""Explicit Symbolic Regression

Explicit symbolic regression is the search for a function, f, such that
f(x) = y.

The classes in this module encapsulate the parts of bingo evolutionary analysis
that are unique to explicit symbolic regression. Namely, these classes are an
appropriate fitness evaluator and a corresponding training data container.
"""
import logging
import numpy as np
from sklearn.decomposition import PCA

from ..evaluation.fitness_function import FitnessFunction

LOGGER = logging.getLogger(__name__)


class AnisotropyFitnessMD(FitnessFunction):
    def __init__(self, training_data):
        super().__init__(training_data)

    def _get_anisotropy(self, output_points):
        n = len(output_points)
        output_points = output_points.reshape(n, -1)
        point_dim = output_points.shape[1]
        if point_dim != 2:
            raise RuntimeError("Unsupported output dimension:", point_dim)

        pca_model = PCA(n_components=point_dim)
        pca_model.fit(output_points)
        return abs(pca_model.singular_values_[1] / pca_model.singular_values_[0] - 1)

    def __call__(self, individual):
        self.eval_count += 1
        mapped_points = individual.evaluate_equation_at(self.training_data)
        return self._get_anisotropy(mapped_points)
