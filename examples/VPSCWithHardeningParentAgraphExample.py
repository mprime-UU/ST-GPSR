import dill

import numpy as np
import torch
# torch.set_num_threads(1)

from bingo.evaluation.fitness_function import VectorBasedFunction
from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
from bingo.evolutionary_optimizers.serial_archipelago import SerialArchipelago
from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_optimizers.island import Island

from bingo.local_optimizers.continuous_local_opt_md import ContinuousLocalOptimizationMD
from bingo.local_optimizers.local_opt_fitness import LocalOptFitnessFunction
from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.symbolic_regression.agraphMD.component_generator_md import ComponentGeneratorMD
from bingo.symbolic_regression.agraphMD.crossover_md import AGraphCrossoverMD
from bingo.symbolic_regression.agraphMD.generator_md import AGraphGeneratorMD
from bingo.symbolic_regression.agraphMD.mutation_md import AGraphMutationMD
from bingo.symbolic_regression.explicit_regression_md import ExplicitRegressionMD, ExplicitTrainingDataMD
from bingo.symbolic_regression.implicit_regression_md import ImplicitRegressionMD, ImplicitTrainingDataMD, \
    _calculate_partials
from bingo.symbolic_regression.implicit_regression_schmidt import ImplicitRegressionSchmidt

from bingo.symbolic_regression.agraphMD.validation_backend import validation_backend

POP_SIZE = 100
STACK_SIZE = 15


def get_ideal_eq():
    from bingo.symbolic_regression.agraphMD.pytorch_agraph_md import PytorchAGraphMD
    ideal_eq_cmd_arr = np.array([[1, 0, 3, 3],
                                 [0, 0, 0, 0],
                                 [1, 0, 0, 0],
                                 [4, 0, 1, 0],
                                 [4, 2, 3, 0]])
    ideal_eq = PytorchAGraphMD(input_dims=[(0, 0)], output_dim=(3, 3))
    ideal_eq.command_array = ideal_eq_cmd_arr
    ideal_eq._update()
    flattened_params = np.array([1, -0.5, -0.5, -0.5, 1, -0.5, -0.5, -0.5, 1, 1])
    ideal_eq.set_local_optimization_params(flattened_params, [(3, 3), (0, 0)])
    print(validation_backend.validate_individual(ideal_eq))
    return ideal_eq


class YieldSurfaceParentAGraphFitness(VectorBasedFunction):
    def __init__(self, *, principal_stresses, state_parameters):
        super().__init__(metric="rmse")
        self.principal_stresses = principal_stresses
        self.state_parameters = state_parameters

    def evaluate_fitness_vector(self, individual):
        P_matrices = individual.evaluate_equation_at(self.state_parameters)

        yield_stresses = self.principal_stresses.transpose((0, 2, 1)) @ P_matrices @ self.principal_stresses
        yield_stresses = yield_stresses.flatten()

        # want difference from mean to be 0 but yield_stresses to be 1
        difference_from_mean = yield_stresses - np.nanmean(yield_stresses)
        difference_from_1 = yield_stresses - 1

        fitness = np.hstack((difference_from_mean, difference_from_1))

        return fitness


def main():
    data = np.loadtxt("vpsc_evo_36_data_3d_points_implicit_format.txt")
    data = data[~np.all(np.isnan(data), axis=1)]

    state_param_dims = [(0, 0)]

    principal_stresses = data[:, :3].reshape((-1, 3, 1))
    eps = torch.from_numpy(data[:, 3].reshape((-1)))
    state_parameters = [eps]

    output_dim = (3, 3)
    print("Dimensions of X variables:", state_param_dims)
    print("Dimension of output:", output_dim)

    component_generator = ComponentGeneratorMD(state_param_dims, possible_dims=[(3, 3), (0, 0)])
    component_generator.add_operator("+")
    component_generator.add_operator("*")
    # component_generator.add_operator("/")
    # component_generator.add_operator("sin")
    # component_generator.add_operator("cos")

    crossover = AGraphCrossoverMD()
    mutation = AGraphMutationMD(component_generator)
    agraph_generator = AGraphGeneratorMD(STACK_SIZE, component_generator, state_param_dims, output_dim,
                                         use_simplification=False, use_pytorch=True)

    yield_surface_fitness = YieldSurfaceParentAGraphFitness(principal_stresses=principal_stresses, state_parameters=state_parameters)
    local_opt_fitness = ContinuousLocalOptimizationMD(yield_surface_fitness, algorithm='lm', param_init_bounds=[-1, 1])
    evaluator = Evaluation(local_opt_fitness, multiprocess=False)

    ideal_eq = get_ideal_eq()
    print("fitness before optimize:", yield_surface_fitness(ideal_eq))
    print("fitness after optimize:", local_opt_fitness(ideal_eq))
    print(ideal_eq.constants)

    ea = AgeFitnessEA(evaluator, agraph_generator, crossover, mutation, 0.3, 0.6, POP_SIZE)

    island = Island(ea, agraph_generator, POP_SIZE)
    island.evolve(1)

    opt_result = island.evolve_until_convergence(max_generations=10000,
                                                 fitness_threshold=1e-5)

    print(opt_result)
    best_indiv = island.get_best_individual()
    print(best_indiv.get_formatted_string("console"))
    print(best_indiv.fitness)
    print(repr(best_indiv.command_array))


if __name__ == '__main__':
    # import random
    # random.seed(7)
    # np.random.seed(7)

    main()
