# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import numpy as np

from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_optimizers.island import Island

from bingo.local_optimizers.continuous_local_opt_md import ContinuousLocalOptimizationMD
from bingo.symbolic_regression.agraphMD.component_generator_md import ComponentGeneratorMD
from bingo.symbolic_regression.agraphMD.crossover_md import AGraphCrossoverMD
from bingo.symbolic_regression.agraphMD.generator_md import AGraphGeneratorMD
from bingo.symbolic_regression.agraphMD.mutation_md import AGraphMutationMD
from bingo.symbolic_regression.explicit_regression_md import ExplicitTrainingDataMD, ExplicitRegressionMD

POP_SIZE = 250
STACK_SIZE = 12


def equation_eval(x):
    C_0 = np.array([[1, 2, 3],
                    [2, 5, 6],
                    [3, 6, 9]])
    C_1 = np.array([[1, 1, 1],
                    [1, 1, 1],
                    [1, 1, 1]])
    return C_0 @ x[0] + C_1


def main():
    n_points = 100

    x = [np.random.rand(n_points, 3, 3)]
    x_dims = [np.shape(_x[0]) for _x in x]

    y = equation_eval(x)
    y_dim = y[0].shape
    training_data = ExplicitTrainingDataMD(x, y)

    print("Dimensions of X variables:", x_dims)
    print("Dimension of output:", y_dim)

    component_generator = ComponentGeneratorMD(x_dims)
    component_generator.add_operator("+")
    component_generator.add_operator("*")

    crossover = AGraphCrossoverMD()
    mutation = AGraphMutationMD(component_generator)
    agraph_generator = AGraphGeneratorMD(STACK_SIZE, component_generator,
                                         x_dims, y_dim,
                                         use_simplification=False,
                                         use_symmetric_constants=True)

    fitness = ExplicitRegressionMD(training_data=training_data, relative=True, relative_type="norm")
    local_opt_fitness = ContinuousLocalOptimizationMD(fitness, algorithm='lm', param_init_bounds=[0, 0])
    evaluator = Evaluation(local_opt_fitness)

    ea = AgeFitnessEA(evaluator, agraph_generator, crossover,
                      mutation, 0.4, 0.4, POP_SIZE)

    island = Island(ea, agraph_generator, POP_SIZE)

    opt_result = island.evolve_until_convergence(max_generations=500,
                                                 fitness_threshold=1.0e-5)

    print(opt_result)
    best_indiv = island.get_best_individual()
    print("found equation:", best_indiv.get_formatted_string("console"))


if __name__ == '__main__':
    import random
    random.seed(7)
    np.random.seed(7)

    main()
