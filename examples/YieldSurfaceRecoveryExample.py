# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import numpy as np
import random
#import h5py
import dill

from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
from bingo.evolutionary_optimizers.serial_archipelago import SerialArchipelago
from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_optimizers.island import Island

from bingo.local_optimizers.continuous_local_opt_md import ContinuousLocalOptimizationMD
from bingo.symbolic_regression.agraphMD.component_generator_md import ComponentGeneratorMD
from bingo.symbolic_regression.agraphMD.crossover_md import AGraphCrossoverMD
from bingo.symbolic_regression.agraphMD.generator_md import AGraphGeneratorMD
from bingo.symbolic_regression.agraphMD.mutation_md import AGraphMutationMD
from bingo.symbolic_regression.yield_surface_fitness_md import YieldSurfaceFitnessMD

POP_SIZE = 250
STACK_SIZE = 10


# def get_data(data_path):
#     f = h5py.File(data_path, "r")
#     dataset = f["generations"]["70000"]["hof_idxs"]["4"]["data"]
#
#     stress_data = dataset[()]
#     return stress_data


def get_data(data_path):
    with open(data_path, "rb") as f:
        data = dill.load(f)
    return data


def execute_generational_steps():
    # my_path = r"../data/poro_bingo_results.hdf5"
    my_path = r"../data/vm_data.pkl"
    data = get_data(my_path)

    x = [data]
    print(data.shape)

    x_dims = [np.shape(_x[0]) for _x in x]
    y_dim = (3, 3)
    print("Dimensions of X variables:", x_dims)
    print("Dimension of output:", y_dim)

    component_generator = ComponentGeneratorMD(x_dims, possible_dims=[(3, 3), (0, 0)])
    component_generator.add_operator("+")
    component_generator.add_operator("*")
    # component_generator.add_operator("sin")
    # component_generator.add_operator("cos")

    crossover = AGraphCrossoverMD()
    mutation = AGraphMutationMD(component_generator)
    agraph_generator = AGraphGeneratorMD(STACK_SIZE, component_generator, x_dims, y_dim, use_simplification=False)

    fitness = YieldSurfaceFitnessMD(data)
    local_opt_fitness = ContinuousLocalOptimizationMD(fitness, algorithm='lm', param_init_bounds=[0, 0])
    evaluator = Evaluation(local_opt_fitness)

    from bingo.symbolic_regression.agraphMD.agraphMD import AGraphMD
    test_eq = AGraphMD(x_dims, y_dim)
    test_eq.command_array = np.array([[1, 0, 3, 3],
                                      [1, 0, 0, 0],
                                      [4, 1, 0, 0]])
    test_eq._update()
    print("test_eq fitness:", local_opt_fitness(test_eq))
    print(test_eq.constants)

    ea = AgeFitnessEA(evaluator, agraph_generator, crossover,
                      mutation, 0.4, 0.4, POP_SIZE)

    island = Island(ea, agraph_generator, POP_SIZE)
    archipelago = SerialArchipelago(island)

    opt_result = archipelago.evolve_until_convergence(max_generations=500,
                                                      fitness_threshold=1.0e-5)

    print(opt_result)
    best_indiv = archipelago.get_best_individual()

    print(best_indiv.get_formatted_string("console"))
    print(best_indiv.constants)
    print(best_indiv.fitness)


if __name__ == '__main__':
    # random.seed(7)
    # np.random.seed(7)
    execute_generational_steps()
