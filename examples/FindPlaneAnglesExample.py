# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import numpy as np

from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
from bingo.evolutionary_optimizers.serial_archipelago import SerialArchipelago
from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_optimizers.island import Island

from bingo.local_optimizers.continuous_local_opt_md import ContinuousLocalOptimizationMD
from bingo.symbolic_regression.agraphMD.component_generator_md import ComponentGeneratorMD
from bingo.symbolic_regression.agraphMD.crossover_md import AGraphCrossoverMD
from bingo.symbolic_regression.agraphMD.generator_md import AGraphGeneratorMD
from bingo.symbolic_regression.agraphMD.mutation_md import AGraphMutationMD
from bingo.symbolic_regression.explicit_regression_md import ExplicitTrainingDataMD, ExplicitRegressionMD

POP_SIZE = 100
STACK_SIZE = 10


def get_orientation_of_plane(first_vector, second_vector):
    cross = np.cross(first_vector, second_vector, axis=1)
    magnitude = np.linalg.norm(cross, axis=1)
    norm_vec = cross / magnitude[:, None]

    # returns [alpha, beta, gamma] = [yaw, pitch, roll]
    return np.arccos(norm_vec)


def execute_generational_steps():
    N = 10

    a_pos = np.random.randn(N, 3, 1)
    b_pos = np.random.randn(N, 3, 1)

    x = [a_pos, b_pos]
    y = get_orientation_of_plane(x[0], x[1])

    training_data = ExplicitTrainingDataMD(x, y)

    x_dims = [np.shape(_x[0]) for _x in x]
    y_dim = y[0].shape
    print("Dimensions of X variables:", x_dims)
    print("Dimension of output:", y_dim)

    component_generator = ComponentGeneratorMD(x_dims)
    component_generator.add_operator("+")
    component_generator.add_operator("*")

    crossover = AGraphCrossoverMD()
    mutation = AGraphMutationMD(component_generator, command_probability=0.333, node_probability=0.333,
                                parameter_probability=0.333, prune_probability=0.0, fork_probability=0.0)
    agraph_generator = AGraphGeneratorMD(STACK_SIZE, component_generator, x_dims, y_dim, use_simplification=False)

    fitness = ExplicitRegressionMD(training_data=training_data, relative=True)
    local_opt_fitness = ContinuousLocalOptimizationMD(fitness, algorithm='lm', param_init_bounds=[0, 0])
    evaluator = Evaluation(local_opt_fitness)

    ea = AgeFitnessEA(evaluator, agraph_generator, crossover,
                      mutation, 0.4, 0.4, POP_SIZE)

    island = Island(ea, agraph_generator, POP_SIZE)
    # archipelago = SerialArchipelago(island)

    island.evolve(1)

    component_generator.add_operator("arccos")
    component_generator.add_operator("cross")
    component_generator.add_operator("normalize")

    opt_result = island.evolve_until_convergence(max_generations=500,
                                                      fitness_threshold=1.0e-10)

    print(opt_result)
    best_indiv = island.get_best_individual()
    print(best_indiv.get_formatted_string("console"))
    print(best_indiv.fitness)


if __name__ == "__main__":
    execute_generational_steps()
