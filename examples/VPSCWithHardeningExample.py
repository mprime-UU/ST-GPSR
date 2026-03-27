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


def get_ideal_eq_2():
    from bingo.symbolic_regression.agraphMD.operator_definitions import TRANSPOSE
    from bingo.symbolic_regression.agraphMD.pytorch_agraph_md import PytorchAGraphMD
    ideal_eq_cmd_arr = np.array([[0, 0, 3, 1],
                                 [TRANSPOSE, 0, 0, 0],
                                 [1, 0, 3, 3],
                                 [4, 1, 2, 0],
                                 [4, 3, 0, 0]])
    ideal_eq = PytorchAGraphMD(input_dims=[(3, 1), (1, 1)], output_dim=(1, 1))
    ideal_eq.command_array = ideal_eq_cmd_arr
    ideal_eq._update()
    # flattened_params = np.array([1, -0.5, -0.5, -0.5, 1, -0.5, -0.5, -0.5, 1])
    flattened_params = np.array([[1.36495482e-03, -5.26189078e+01, -4.95735860e+00],
                                 [4.59798156e+01, -6.64033389e+00, 2.23621520e+01],
                                 [3.35895304e+01, -3.71651774e-01, 2.86309023e+01]]).flatten()
    ideal_eq.set_local_optimization_params(flattened_params, [(3, 3)])
    print(validation_backend.validate_individual(ideal_eq))
    return ideal_eq


def get_ideal_eq():
    from bingo.symbolic_regression.agraphMD.operator_definitions import TRANSPOSE
    from bingo.symbolic_regression.agraphMD.pytorch_agraph_md import PytorchAGraphMD
    ideal_eq_cmd_arr = np.array([[0, 0, 3, 1],          # X_0; 0
                                 [TRANSPOSE, 0, 0, 0],  # X_0.T; 1
                                 [1, 0, 3, 3],          # C_0; 2
                                 [4, 1, 2, 0],          # X_0.T @ C_0; 3
                                 [4, 3, 0, 0],          # X_0.T @ C_0 @ X_0; 4
                                 [1, 1, 1, 1],          # C_1; 5
                                 [0, 1, 1, 1],          # X_1; 6
                                 [4, 5, 6, 0],          # C_1 * X_1; 7
                                 [2, 4, 7, 0],          # X_0.T @ C_0 @ X_0 + C_1 * X_1; 8
                                 [1, 2, 1, 1],          # C_2; 9
                                 [4, 9, 6, 0],          # C_2 * X_1; 10
                                 [4, 10, 6, 0],         # C_2 * X_1 * X_1; 11
                                 [2, 8, 11, 0]])        # X_0.T @ C_0 @ X_0 + C_1 * X_1 + C_2 * X_1 * X_1; 12
    ideal_eq = PytorchAGraphMD(input_dims=[(3, 1), (1, 1)], output_dim=(1, 1))
    ideal_eq.command_array = ideal_eq_cmd_arr
    ideal_eq._update()
    # flattened_params = np.array([1, -0.5, -0.5, -0.5, 1, -0.5, -0.5, -0.5, 1, -39600/2, -1960200/2])
    flattened_params = np.array([1, -0.5, -0.5, -0.5, 1, -0.5, -0.5, -0.5, 1, 0, 0])
    ideal_eq.set_local_optimization_params(flattened_params, [(3, 3), (0, 0), (0, 0)])
    print(validation_backend.validate_individual(ideal_eq))
    return ideal_eq


def get_dx_dt(x_with_nan):
    point_trajectories = []

    point_trajectory = []
    for row in x_with_nan:
        if not np.all(np.isnan(row)):
            point_trajectory.append(row)
        else:
            point_trajectories.append(point_trajectory)
            point_trajectory = []
    point_trajectories.append(point_trajectory)  # for last point which doesn't have a nan footer

    dx_dts = []

    for point_trajectory in point_trajectories:
        dx_dt = []
        prev_point = np.zeros(x_with_nan.shape[1])
        for point in point_trajectory:
            dx_dt.append(point - prev_point)
            prev_point = point
        dx_dts.append(dx_dt)

    dx_dts = np.array(dx_dts)
    dx_dts = dx_dts.reshape((-1, dx_dts.shape[2]))
    return dx_dts


def main():
    data = np.loadtxt("vpsc_evo_57_data_3d_points_implicit_format.txt")

    # with open("vm_data_full.pkl", "rb") as f:
    #     data = dill.load(f)[:1000, :4]

    x, dx_dt, _ = _calculate_partials(data, window_size=5)
    # dx_dt /= x_row_scale
    # x[:, :3] *= 1000
    # dx_dt[:, :3] *= 100

    training_data = ImplicitTrainingDataMD(x, dx_dt)
    y = np.ones((training_data._x.shape[0], 1, 1)) * 1
    x_dims = [(3, 1), (1, 1)]

    x_0 = torch.from_numpy(training_data._x[:, :3].reshape((-1, 3, 1)))
    x_1 = torch.from_numpy(training_data._x[:, 3].reshape((-1, 1, 1)))
    x = [x_0, x_1]
    training_data._x = x

    explicit_training_data = ExplicitTrainingDataMD(x, y)

    y_dim = (1, 1)
    print("Dimensions of X variables:", x_dims)
    print("Dimension of output:", y_dim)

    component_generator = ComponentGeneratorMD(x_dims, possible_dims=[(3, 3), (3, 1), (1, 1)], possible_dim_weights=[1, 0, 0], x_weights=[1, 0])
    component_generator.add_operator("+")
    component_generator.add_operator("*")
    component_generator.add_operator("transpose")
    component_generator.add_operator("sin")
    component_generator.add_operator("cos")

    crossover = AGraphCrossoverMD()
    mutation = AGraphMutationMD(component_generator)
    agraph_generator = AGraphGeneratorMD(STACK_SIZE, component_generator, x_dims, y_dim, use_simplification=False,
                                         use_pytorch=True)

    # implicit_fitness = ImplicitRegressionMD(training_data, required_params=4)
    implicit_fitness = ImplicitRegressionMD(training_data)
    explicit_fitness = ExplicitRegressionMD(explicit_training_data)
    local_opt_fitness = ContinuousLocalOptimizationMD(implicit_fitness, algorithm='lm', param_init_bounds=[-1, 1])
    test_fitness = local_opt_fitness

    class MyFitness(VectorBasedFunction):
        def __init__(self, implicit_fitness_function, explicit_fitness_function, algorithm='Nelder-Mead',
        param_init_bounds=None, **optimization_options_kwargs):
            super().__init__(metric="rmse")
            self._optimizer = ContinuousLocalOptimizationMD(explicit_fitness_function, algorithm,
                                                            param_init_bounds, **optimization_options_kwargs)
            self.end_fitness_function = implicit_fitness_function

        def evaluate_fitness_vector(self, individual):
            optim_fitness = self._optimizer(individual)
            # self.end_fitness_function.training_data._x.grad = None
            fitness = self.end_fitness_function.evaluate_fitness_vector(individual)
            # print(f"evaluating individual: {individual}, has fitness: {self._metric(fitness)}")
            return fitness

    # optimizer = ScipyOptimizer(explicit_fitness, method="lm", param_init_bounds=[-1, 1])
    # test_fitness = LocalOptFitnessFunction(implicit_fitness, optimizer)
    test_fitness = MyFitness(implicit_fitness, explicit_fitness, algorithm='lm', param_init_bounds=[-1, 1])
    evaluator = Evaluation(test_fitness, multiprocess=False)

    ideal_eq = get_ideal_eq()
    print(implicit_fitness(ideal_eq))
    print("before optimize")
    print("fitness of ideal_eq:", test_fitness(ideal_eq))
    print(ideal_eq.constants)

    # print(ideal_eq.evaluate_equation_at(training_data._x))

    # print(ideal_eq.constants)
    # print("fitness of ideal_eq:", fitness(ideal_eq))

    ea = AgeFitnessEA(evaluator, agraph_generator, crossover,
                      mutation, 0.3, 0.6, POP_SIZE)

    island = Island(ea, agraph_generator, POP_SIZE)

    island._generator.component_generator.possible_dim_weights = [1./3., 1./3., 1./3.]
    island._generator.component_generator.x_weights = [0.5, 0.5]
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
