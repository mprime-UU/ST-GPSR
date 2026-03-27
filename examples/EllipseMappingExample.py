# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import numpy as np
import random
import matplotlib.pyplot as plt

from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
from bingo.evolutionary_optimizers.serial_archipelago import SerialArchipelago
from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_optimizers.island import Island

from bingo.local_optimizers.continuous_local_opt_md import ContinuousLocalOptimizationMD
from bingo.symbolic_regression.agraphMD.component_generator_md import ComponentGeneratorMD
from bingo.symbolic_regression.agraphMD.crossover_md import AGraphCrossoverMD
from bingo.symbolic_regression.agraphMD.generator_md import AGraphGeneratorMD
from bingo.symbolic_regression.agraphMD.mutation_md import AGraphMutationMD
from bingo.symbolic_regression.anisostropy_fitness_md import AnisotropyFitnessMD

POP_SIZE = 250
STACK_SIZE = 10


def get_ellipse_points(n, a, b):
    t = np.random.rand(n) * 2 * np.pi
    points = np.array([[a * np.cos(t), b * np.sin(t)]]).transpose((2, 0, 1))
    return points


def execute_generational_steps():
    np.random.seed(2)
    n_points = 100
    points = get_ellipse_points(n_points, 5, 2)
    x = [points]

    x_dims = [np.shape(_x[0]) for _x in x]
    y_dim = x_dims[0]
    print("Dimensions of X variables:", x_dims)
    print("Dimension of output:", y_dim)

    component_generator = ComponentGeneratorMD(x_dims)
    component_generator.add_operator("+")
    component_generator.add_operator("*")

    crossover = AGraphCrossoverMD()
    mutation = AGraphMutationMD(component_generator)
    agraph_generator = AGraphGeneratorMD(STACK_SIZE, component_generator, x_dims, y_dim, use_simplification=False)

    fitness = AnisotropyFitnessMD(x)
    local_opt_fitness = ContinuousLocalOptimizationMD(fitness, algorithm='BFGS', param_init_bounds=[0, 0])
    evaluator = Evaluation(local_opt_fitness)

    from bingo.symbolic_regression.agraphMD.agraphMD import AGraphMD
    ideal_eq = AGraphMD([(1, 2)], (1, 2))
    ideal_eq.command_array = np.array([[0, 0, 1, 2],
                                       [1, 0, 2, 2],
                                       [4, 0, 1, 0]])
    ideal_eq._update()
    ideal_eq.set_local_optimization_params([0.5, 0, 0, 1], [(2, 2)])
    print("ideal eq fitness:", fitness(ideal_eq))

    ideal_x, ideal_y = ideal_eq.evaluate_equation_at(x).reshape(n_points, 2).T
    orig_x, orig_y = x[0].reshape(n_points, 2).T
    plt.scatter(orig_x, orig_y, label="original")
    plt.scatter(ideal_x, ideal_y, label="ideal mapping")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect("equal")
    plt.legend(loc="upper right")
    plt.show()

    ea = AgeFitnessEA(evaluator, agraph_generator, crossover,
                      mutation, 0.4, 0.4, POP_SIZE)

    island = Island(ea, agraph_generator, POP_SIZE)
    archipelago = SerialArchipelago(island)

    opt_result = archipelago.evolve_until_convergence(max_generations=500,
                                                      fitness_threshold=1.0e-5)

    print(opt_result)
    best_indiv = archipelago.get_best_individual()
    print(fitness(best_indiv))

    print(best_indiv.get_formatted_string("console"))
    print(best_indiv.fitness)

    mapped_data = best_indiv.evaluate_equation_at(x)
    mapped_x, mapped_y = mapped_data.reshape(n_points, 2).T
    plt.scatter(orig_x, orig_y, label="original")
    plt.scatter(mapped_x, mapped_y, label="found mapping")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.gca().set_aspect("equal")
    plt.legend(loc="upper right")
    plt.show()


if __name__ == '__main__':
    random.seed(7)
    np.random.seed(7)
    execute_generational_steps()
