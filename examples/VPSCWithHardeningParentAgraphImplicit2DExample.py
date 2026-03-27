import dill

import numpy as np
import torch
torch.set_num_threads(1)
MULITPROCESS_CORES = 10

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
from bingo.symbolic_regression.implicit_regression_md import ImplicitRegressionMD, ImplicitTrainingDataMD, _calculate_partials
# from bingo.symbolic_regression.implicit_regression_md import ImplicitRegressionMD, ImplicitTrainingDataMD
# from bingo.symbolic_regression.implicit_regression import _calculate_partials
from bingo.symbolic_regression.implicit_regression_schmidt import ImplicitRegressionSchmidt

from bingo.symbolic_regression.agraphMD.validation_backend import validation_backend

POP_SIZE = 250
STACK_SIZE = 15


class ParentAgraphIndv:
    def __init__(self, P_matrix_indv):
        self.P_matrix_indv = P_matrix_indv

    def evaluate_equation_at(self, x, detach=True):
        principal_stresses, state_parameters = x

        P_matrices = self.P_matrix_indv.evaluate_equation_at_no_detach([state_parameters])

        yield_stresses = torch.transpose(principal_stresses, 1, 2) @ P_matrices @ principal_stresses

        if detach:
            yield_stresses = yield_stresses.detach().numpy()
            return yield_stresses / np.mean(yield_stresses)
        else:
            return yield_stresses

    def evaluate_equation_with_x_gradient_at(self, x):
        for input in x:
            input.grad = None
            input.requires_grad = True

        yield_stresses = self.evaluate_equation_at(x, detach=False)

        if yield_stresses.requires_grad:
            yield_stresses.sum().backward()

        full_derivative = []
        for input in x:
            input_derivative = input.grad
            if input_derivative is None:
                try:
                    input_derivative = torch.zeros((yield_stresses.shape[0], input.shape[1]))
                except IndexError:
                    input_derivative = torch.zeros((yield_stresses.shape[0]))
            full_derivative.append(input_derivative.detach().numpy())

        for input in x:
            input.requires_grad = False

        return yield_stresses.detach().numpy(), full_derivative


class YieldSurfaceParentAGraphFitness(VectorBasedFunction):
    def __init__(self, *, fitness_for_parent):
        super().__init__(metric="mse")
        self.fitness_fn = fitness_for_parent

    def evaluate_fitness_vector(self, individual):
        parent_agraph = ParentAgraphIndv(individual)
        fitness = self.fitness_fn.evaluate_fitness_vector(parent_agraph)
        return fitness


class TotalYieldSurfaceParentAGraphFitness(VectorBasedFunction):
    def __init__(self, *, implicit_fitness, explicit_fitness):
        super().__init__(metric="rmse")
        self.implicit_fitness = implicit_fitness
        self.explicit_fitness = explicit_fitness

    def evaluate_fitness_vector(self, individual):
        parent_agraph = ParentAgraphIndv(individual)
        implicit_fitness = self.implicit_fitness.evaluate_fitness_vector(parent_agraph)
        explicit_fitness = self.explicit_fitness.evaluate_fitness_vector(parent_agraph)
        return np.hstack((implicit_fitness, explicit_fitness))


class ExplicitOptimizedImplicitFitness(VectorBasedFunction):
    def __init__(self, implicit_fitness_function, explicit_fitness_function, algorithm='Nelder-Mead',
                 param_init_bounds=None, **optimization_options_kwargs):
        super().__init__(metric="mse")
        self._optimizer = ContinuousLocalOptimizationMD(explicit_fitness_function, algorithm,
                                                        param_init_bounds, **optimization_options_kwargs)
        self.end_fitness_function = implicit_fitness_function

    def evaluate_fitness_vector(self, individual):
        mean_value = np.mean(self._optimizer._fitness_function.evaluate_fitness_vector(
            individual) + self._optimizer._fitness_function.fitness_fn.training_data.y)
        self._optimizer._fitness_function.fitness_fn.training_data._y = np.repeat(mean_value[None, None, None], len(self._optimizer._fitness_function.fitness_fn.training_data.y), axis=0)
        optimizer_fitness = self._optimizer(individual)
        fitness = self.end_fitness_function.evaluate_fitness_vector(individual)
        return fitness


def main():
    # data = np.loadtxt("vpsc_evo_17_data_2d_implicit_format.txt")
    data = np.loadtxt("vpsc_evo_17_data_2d_transpose_implicit_format.txt")

    x, dx_dt, _ = _calculate_partials(data, window_size=5)
    print(dx_dt.shape)

    implicit_training_data = ImplicitTrainingDataMD(x, dx_dt)
    x_0 = torch.from_numpy(implicit_training_data._x[:, :2].reshape((-1, 2, 1))).double()
    x_1 = torch.from_numpy(implicit_training_data._x[:, 2].reshape((-1))).double()
    x = [x_0, x_1]
    implicit_training_data._x = x
    implicit_fitness = ImplicitRegressionMD(implicit_training_data)

    y = np.ones((x[0].shape[0], 1, 1)) * 100
    explicit_training_data = ExplicitTrainingDataMD(x, y)
    explicit_fitness = ExplicitRegressionMD(explicit_training_data, relative=True)

    parent_explicit_fitness = YieldSurfaceParentAGraphFitness(fitness_for_parent=explicit_fitness)
    parent_implicit_fitness = YieldSurfaceParentAGraphFitness(fitness_for_parent=implicit_fitness)
    yield_surface_fitness = TotalYieldSurfaceParentAGraphFitness(implicit_fitness=implicit_fitness, explicit_fitness=explicit_fitness)

    # ! yield surface fitness generally yields better results than parent implicit fitness
    # local_opt_fitness = ExplicitOptimizedImplicitFitness(yield_surface_fitness, parent_explicit_fitness, algorithm="lm")
    local_opt_fitness = ExplicitOptimizedImplicitFitness(parent_implicit_fitness, parent_explicit_fitness, algorithm="lm")

    # local_opt_fitness = ContinuousLocalOptimizationMD(parent_explicit_fitness, algorithm='lm', param_init_bounds=[-1, 1])
    # local_opt_fitness = ContinuousLocalOptimizationMD(parent_implicit_fitness, algorithm='lm', param_init_bounds=[-1, 1])
    # local_opt_fitness = ContinuousLocalOptimizationMD(yield_surface_fitness, algorithm='lm', param_init_bounds=[-1, 1])

    evaluator = Evaluation(local_opt_fitness, multiprocess=3)

    state_param_dims = [(0, 0)]
    output_dim = (2, 2)
    print("Dimensions of X variables:", state_param_dims)
    print("Dimension of output:", output_dim)

    component_generator = ComponentGeneratorMD(state_param_dims, possible_dims=[(2, 2), (0, 0)], possible_dim_weights=(0.4, 0.6))
    component_generator.add_operator("+")
    component_generator.add_operator("*")
    # component_generator.add_operator("/")
    component_generator.add_operator("sin")
    component_generator.add_operator("cos")

    crossover = AGraphCrossoverMD()
    mutation = AGraphMutationMD(component_generator)
    agraph_generator = AGraphGeneratorMD(STACK_SIZE, component_generator, state_param_dims, output_dim,
                                         use_simplification=False, use_pytorch=True)

    ea = AgeFitnessEA(evaluator, agraph_generator, crossover, mutation, 0.3, 0.6, POP_SIZE)

    island = Island(ea, agraph_generator, POP_SIZE)
    island.evolve(1)

    opt_result = island.evolve_until_convergence(max_generations=10000,
                                                 fitness_threshold=1e-5)

    print(opt_result)
    best_indiv = island.get_best_individual()
    yield_surface_fitness(best_indiv)
    print(best_indiv.get_formatted_string("console"))
    print(best_indiv.fitness)
    print(repr(best_indiv.command_array))
    print(repr(best_indiv.constants))


if __name__ == '__main__':
    # import random
    # random.seed(7)
    # np.random.seed(7)

    main()
