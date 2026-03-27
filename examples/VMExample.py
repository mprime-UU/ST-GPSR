import dill

import numpy as np
import torch

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
from bingo.symbolic_regression.implicit_regression_md import ImplicitRegressionMD, ImplicitTrainingDataMD
from bingo.symbolic_regression.implicit_regression_schmidt import ImplicitRegressionSchmidt

POP_SIZE = 100
STACK_SIZE = 5


def get_ideal_eq():
    from bingo.symbolic_regression.agraphMD.operator_definitions import TRANSPOSE
    from bingo.symbolic_regression.agraphMD.pytorch_agraph_md import PytorchAGraphMD
    ideal_eq_cmd_arr = np.array([[0, 0, 3, 1],  # X_0; 0
                                 [1, 0, 3, 3],  # C_0; 1
                                 [TRANSPOSE, 0, 0, 0],  # X_0.T; 2
                                 [4, 2, 1, 0],  # X_0.T @ C_0; 3
                                 [4, 3, 0, 0],  # X_0.T @ C_0 @ X_0; 4
                                 ])
    # ideal_eq_cmd_arr = np.array([[ 0,  0,  3,  1],
    #        [ 3,  0,  0,  0],
    #        [16,  0,  0,  0],
    #        [ 3,  1,  1,  0],
    #        [ 4,  2,  1,  0]])
    ideal_eq = PytorchAGraphMD(input_dims=[(3, 1)], output_dim=[(1, 1)])
    ideal_eq.command_array = ideal_eq_cmd_arr
    ideal_eq._update()
    flattened_params = np.array([[1, -0.5, -0.5],
                                 [-0.5, 1, -0.5],
                                 [-0.5, -0.5, 1]]).flatten()
    # ideal_eq.set_local_optimization_params(flattened_params, [(3, 3)])
    return ideal_eq


def main():
    with open("vm_data.pkl", "rb") as f:
        x = dill.load(f)

    training_data = ImplicitTrainingDataMD(x)
    # training_data._x = training_data._x.reshape((-1, 3, 1))
    # training_data._dx_dt = training_data._dx_dt[:, None, :]
    # x = x.reshape((-1, 3, 1))
    x_dims = [(3, 1)]

    x = torch.from_numpy(training_data._x.reshape((1, -1, 3, 1))).double()
    training_data._x = x

    y = np.ones((x.size(0), 1)) * 100
    explicit_training_data = ExplicitTrainingDataMD(x, y)

    y_dim = (1, 1)
    print("Dimensions of X variables:", x_dims)
    print("Dimension of output:", y_dim)

    component_generator = ComponentGeneratorMD(x_dims, possible_dims=[(3, 3)])
    component_generator.add_operator("+")
    component_generator.add_operator("-")
    component_generator.add_operator("*")
    component_generator.add_operator("transpose")

    crossover = AGraphCrossoverMD()
    mutation = AGraphMutationMD(component_generator)
    agraph_generator = AGraphGeneratorMD(STACK_SIZE, component_generator, x_dims, y_dim, use_simplification=False,
                                         use_pytorch=True)

    implicit_fitness = ImplicitRegressionMD(training_data)
    explicit_fitness = ExplicitRegressionMD(explicit_training_data)
    local_opt_fitness = ContinuousLocalOptimizationMD(implicit_fitness, algorithm='lm', param_init_bounds=[-1, 1], options={"maxiter": 100})
    test_fitness = local_opt_fitness

    class MyFitness(VectorBasedFunction):
        def __init__(self, implicit_fitness_function, explicit_fitness_function, algorithm='Nelder-Mead',
        param_init_bounds=None, **optimization_options_kwargs):
            super().__init__(metric="mse")
            self._optimizer = ContinuousLocalOptimizationMD(explicit_fitness_function, algorithm,
                                                            param_init_bounds, **optimization_options_kwargs)
            self.end_fitness_function = implicit_fitness_function

        def evaluate_fitness_vector(self, individual):
            self._optimizer(individual)
            self.end_fitness_function.training_data._x.grad = None
            fitness = self.end_fitness_function.evaluate_fitness_vector(individual)
            # print(f"evaluating individual: {individual}, has fitness: {self._metric(fitness)}")
            return fitness

    # optimizer = ScipyOptimizer(explicit_fitness, method="lm", param_init_bounds=[-1, 1])
    # test_fitness = LocalOptFitnessFunction(implicit_fitness, optimizer)
    test_fitness = MyFitness(implicit_fitness, explicit_fitness, algorithm='lm', param_init_bounds=[-1, 1])
    evaluator = Evaluation(test_fitness)

    ideal_eq = get_ideal_eq()
    print("before optimize")
    print("fitness of ideal_eq:", test_fitness(ideal_eq))

    # TODO why is best fitness of the first generation always the fitness above, even with the wrong individual?
    # print(ideal_eq.constants)
    # print("fitness of ideal_eq:", fitness(ideal_eq))

    ea = AgeFitnessEA(evaluator, agraph_generator, crossover,
                      mutation, 0.4, 0.4, POP_SIZE)

    island = Island(ea, agraph_generator, POP_SIZE)
    archipelago = SerialArchipelago(island)

    opt_result = archipelago.evolve_until_convergence(max_generations=500,
                                                      fitness_threshold=1e-5)

    print(opt_result)
    best_indiv = archipelago.get_best_individual()
    print(best_indiv.get_formatted_string("console"))
    print(best_indiv.fitness)
    print(repr(best_indiv.command_array))


if __name__ == '__main__':
    # import random
    # random.seed(7)
    # np.random.seed(7)

    main()
