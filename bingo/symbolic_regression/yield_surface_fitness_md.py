import logging
import numpy as np

from ..evaluation.fitness_function import VectorBasedFunction

LOGGER = logging.getLogger(__name__)


class YieldSurfaceFitnessMD(VectorBasedFunction):
    def __init__(self, training_data):
        super().__init__(training_data)
        self.training_data = training_data

    def evaluate_fitness_vector(self, individual):
        self.eval_count += 1
        mapping = individual.evaluate_equation_at([self.training_data])
        data = self.training_data
        err = np.abs(data[:, :, None, :] @ mapping[:, None, :, :] @ data[:, :, :, None] - 1).flatten()
        return err
