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

ANISOTROPY_STRUCTURES = {"cubic": [3, np.array([[1, 2, 2, 0, 0, 0],
                                                [2, 1, 2, 0, 0, 0],
                                                [2, 2, 1, 0, 0, 0],
                                                [0, 0, 0, 3, 0, 0],
                                                [0, 0, 0, 0, 3, 0],
                                                [0, 0, 0, 0, 0, 3]])],
                         "monoclinic": [13, np.array([[1, 2, 3, 0,  0,  4],
                                                      [2, 5, 6, 0,  0,  7],
                                                      [3, 6, 8, 0,  0,  9],
                                                      [0, 0, 0, 10, 11, 0],
                                                      [0, 0, 0, 11, 12, 0],
                                                      [4, 7, 9, 0,  0,  13]])]}


def get_random_c_matrix(anisotropy):
    if anisotropy.lower() not in ANISOTROPY_STRUCTURES.keys():
        raise RuntimeError(f"Unsupported anistropy: {anisotropy.lower()}")

    # get info from mapping and replace constants with random constants
    n_constants, structure = ANISOTROPY_STRUCTURES[anisotropy]

    random_constants = np.random.rand(n_constants)

    constant_idxs = []
    for i in range(1, 1+n_constants):
        constant_idxs.append(structure == i)

    c_matrix = np.zeros((6, 6))
    for i, constant_idx in enumerate(constant_idxs):
        c_matrix[constant_idx] = random_constants[i]

    return c_matrix


def get_training_data(n_points, c_matrix_anisotropy):
    x_0 = np.random.rand(n_points, 6, 6)
    x = [x_0]

    C = get_random_c_matrix(c_matrix_anisotropy)
    print(f"Trying to find:\n{C}(X)")

    y = C @ x[0]

    training_data = ExplicitTrainingDataMD(x, y)

    return training_data


def get_evolutionary_optimizer(training_data):
    inputs, output = training_data.x, training_data.y
    input_dims = [np.shape(_x[0]) for _x in inputs]
    output_dim = output[0].shape

    component_generator = ComponentGeneratorMD(input_dims, possible_dims=[(6, 6)])
    component_generator.add_operator("+")
    component_generator.add_operator("*")

    crossover = AGraphCrossoverMD()
    mutation = AGraphMutationMD(component_generator, command_probability=0.333, node_probability=0.333,
                                parameter_probability=0.333, prune_probability=0.0, fork_probability=0.0)
    agraph_generator = AGraphGeneratorMD(STACK_SIZE, component_generator, input_dims, output_dim, use_simplification=False)

    fitness = ExplicitRegressionMD(training_data=training_data)
    local_opt_fitness = ContinuousLocalOptimizationMD(fitness, algorithm='lm', param_init_bounds=[0, 1])
    evaluator = Evaluation(local_opt_fitness)

    ea = AgeFitnessEA(evaluator, agraph_generator, crossover,
                      mutation, 0.4, 0.4, POP_SIZE)

    island = Island(ea, agraph_generator, POP_SIZE)

    return island


if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    POP_SIZE = 100
    STACK_SIZE = 6
    MAX_GENS = 500
    FITNESS_THRESHOLD = 1e-6

    N_POINTS = 100

    training_data = get_training_data(N_POINTS, "monoclinic")
    evo_opt = get_evolutionary_optimizer(training_data)
    opt_result = evo_opt.evolve_until_convergence(max_generations=MAX_GENS,
                                                  fitness_threshold=FITNESS_THRESHOLD)

    print(opt_result)
    best_indiv = evo_opt.get_best_individual()
    print(best_indiv.get_formatted_string("console"))
    print(best_indiv.fitness)
