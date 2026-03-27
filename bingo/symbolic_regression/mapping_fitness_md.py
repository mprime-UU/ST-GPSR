import numpy as np
import torch.linalg

from bingo.evaluation.fitness_function import VectorBasedFunction
from bingo.evaluation.training_data import TrainingData


class MappingFitness(VectorBasedFunction):
    def __init__(self, training_data):
        super().__init__(training_data)

    def evaluate_fitness_vector(self, individual):
        # never use inverse here if we want to reverse aniso <-> vm, can just switch the input and outputs
        # i.e., just switch P_desired/P_actual as the input/output rather than using inverse—much faster

        mapping_matrices = individual.evaluate_equation_at(self.training_data.state_parameters)
        # P_mapped = mapping_matrices.transpose((0, 2, 1)) @ self.training_data.P_actual @ mapping_matrices
        P_mapped = mapping_matrices.transpose((0, 2, 1)) @ self.training_data.P_desired @ mapping_matrices

        # err = np.linalg.norm(self.training_data.P_desired - P_mapped, axis=(1, 2))
        # err = (self.training_data.P_desired - P_mapped).flatten()
        # err /= self.training_data.P_desired.flatten()

        # err = self.training_data.P_desired - P_mapped
        # err /= self.training_data.P_desired

        err = self.training_data.P_actual - P_mapped
        err /= self.training_data.P_actual

        err = err.flatten()

        # err = np.linalg.norm(err, axis=(1, 2))

        # relative error
        # denom = np.linalg.norm(self.training_data.P_desired, axis=(1, 2))
        # both_zero = (err == 0) & (denom == 0)
        # err /= denom
        # err[both_zero] = 0

        return err


class MappingFitnessTrainingData(TrainingData):
    def __init__(self, *, state_parameters, P_actual, P_desired):
        self.state_parameters = state_parameters
        self.P_actual = P_actual
        self.P_desired = P_desired

    def __getitem__(self, items):
        sliced_state_parameters = [param[items] for param in self.state_parameters]
        temp = MappingFitnessTrainingData(state_parameters=sliced_state_parameters,
                                          P_actual=self.P_actual[items],
                                          P_desired=self.P_desired[items])
        return temp

    def __len__(self):
        return len(self.P_actual)
