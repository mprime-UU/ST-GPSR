# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import time

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)

from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
from bingo.evolutionary_optimizers.serial_archipelago import SerialArchipelago
from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_optimizers.island import Island

from bingo.local_optimizers.continuous_local_opt_md import ContinuousLocalOptimizationMD
from bingo.symbolic_regression.agraphMD.agraphMD import AGraphMD
from bingo.symbolic_regression.agraphMD.component_generator_md import ComponentGeneratorMD
from bingo.symbolic_regression.agraphMD.crossover_md import AGraphCrossoverMD
from bingo.symbolic_regression.agraphMD.generator_md import AGraphGeneratorMD
from bingo.symbolic_regression.agraphMD.mutation_md import AGraphMutationMD
from bingo.symbolic_regression.explicit_regression_md import ExplicitTrainingDataMD, ExplicitRegressionMD

from bingo.symbolic_regression.mapping_fitness_md import MappingFitnessTrainingData, MappingFitness

POP_SIZE = 100
STACK_SIZE = 10


def get_ideal_eq_1(use_pytorch):
    cmd_arr = np.array([[1, 0, 3, 3],
                        [1, 1, 0, 0],
                        [1, 2, 0, 0],
                        [0, 0, 0, 0],
                        [4, 2, 3, 0],
                        [4, 4, 3, 0],
                        [1, 3, 0, 0],
                        [4, 6, 3, 0],
                        [2, 5, 7, 0],
                        [3, 1, 8, 0],
                        [4, 0, 9, 0],
                        [12, 10, 10, 0]])
    if use_pytorch:
        from bingo.symbolic_regression.agraphMD.pytorch_agraph_md import PytorchAGraphMD
        ideal_eq = PytorchAGraphMD(input_dims=[(0, 0)], output_dim=(3, 3))
    else:
        ideal_eq = AGraphMD(input_dims=[(0, 0)], output_dim=(3, 3))
    ideal_eq.command_array = cmd_arr
    ideal_eq._update()
    flattened_params = [1, 0, 0, 0, 1, 0, 0, 0, 1, 1, -9801, -198]
    ideal_eq.set_local_optimization_params(flattened_params, [(3, 3), (0, 0), (0, 0), (0, 0)])
    print("ideal_eq:", ideal_eq)
    return ideal_eq


def get_ideal_eq(use_pytorch):
    cmd_arr = np.array([[1, 0, 3, 3],
                        [1, 1, 0, 0],
                        [0, 0, 0, 0],
                        [4, 1, 2, 0],
                        [1, 2, 0, 0],
                        [2, 3, 4, 0],
                        [4, 0, 5, 0]])
    if use_pytorch:
        from bingo.symbolic_regression.agraphMD.pytorch_agraph_md import PytorchAGraphMD
        ideal_eq = PytorchAGraphMD(input_dims=[(0, 0)], output_dim=(3, 3))
    else:
        ideal_eq = AGraphMD(input_dims=[(0, 0)], output_dim=(3, 3))
    ideal_eq.command_array = cmd_arr
    ideal_eq._update()
    flattened_params = [1, 0, 0,
                        0, 1, 0,
                        0, 0, 1,
                        99, 1]
    ideal_eq.set_local_optimization_params(flattened_params, [(3, 3), (0, 0), (0, 0)])
    print("ideal_eq:", ideal_eq)
    return ideal_eq


def main(use_pytorch=False):
    np.set_printoptions(suppress=True)
    df = pd.read_pickle("hardening_df.pkl")

    def get_numpy_matrix(df, column):
        column_list = df[column]
        numpy_version = []
        for data_entry in column_list:
            numpy_version.append(data_entry)
        return np.array(numpy_version)

    state_parameters = [get_numpy_matrix(df, "eps")]
    P_actual = get_numpy_matrix(df, "P_hardening")
    P_desired = get_numpy_matrix(df, "P_vm")

    import dill
    with open("vm_data_full.pkl", "rb") as f:
        full_vm_data = dill.load(f)[:953]
        vm_data = full_vm_data[:, :3].reshape((-1, 3, 1))
        state_parameters = [full_vm_data[:, 3].reshape((-1))]
    mapping_eq = get_ideal_eq(use_pytorch=False)
    mapping_tensors = mapping_eq.evaluate_equation_at(state_parameters)
    mapping_tensors = np.array([np.linalg.inv(tensor) for tensor in mapping_tensors])
    print(mapping_tensors.shape)
    mapped_stress = mapping_tensors @ vm_data
    P_vm = np.array([[1, -0.5, -0.5],
                     [-0.5, 1, -0.5],
                     [-0.5, -0.5, 1]])
    print((mapped_stress.transpose((0, 2, 1)) @ P_vm @ mapped_stress).flatten())
    print(df.columns)
    raise RuntimeError()

    x_dims = [(0, 0) if np.shape(x_[0]) == () else np.shape(x_[0]) for x_ in state_parameters]
    y_dim = P_desired[0].shape
    if use_pytorch:
        state_parameters = [torch.from_numpy(state_parameters[0]).double()]
    training_data = MappingFitnessTrainingData(state_parameters=state_parameters,
                                               P_actual=P_actual,
                                               P_desired=P_desired)

    print("Dimensions of X variables:", x_dims)
    print("Dimension of output:", y_dim)

    component_generator = ComponentGeneratorMD(x_dims, possible_dims=[(3, 3), (0, 0)])
    component_generator.add_operator("+")
    component_generator.add_operator("*")
    component_generator.add_operator("sin")
    component_generator.add_operator("cos")
    component_generator.add_operator("sqrt")

    crossover = AGraphCrossoverMD()
    mutation = AGraphMutationMD(component_generator)
    agraph_generator = AGraphGeneratorMD(STACK_SIZE, component_generator, x_dims, y_dim, use_simplification=False,
                                         use_pytorch=use_pytorch)

    fitness = MappingFitness(training_data=training_data)
    local_opt_fitness = ContinuousLocalOptimizationMD(fitness, algorithm='lm', param_init_bounds=[1, 1], options={"factor": 0.1, "xtol": 1e-15, "ftol": 1e-15, "gtol": 1e-15})
    # local_opt_fitness = ContinuousLocalOptimizationMD(fitness, algorithm='lm', param_init_bounds=[1, 1])
    evaluator = Evaluation(local_opt_fitness, multiprocess=7)

    ea = AgeFitnessEA(evaluator, agraph_generator, crossover,
                      mutation, 0.4, 0.4, POP_SIZE)

    island = Island(ea, agraph_generator, POP_SIZE)
    # archipelago = SerialArchipelago(island)

    ideal_eq = get_ideal_eq(use_pytorch)
    # ideal_eq.set_local_optimization_params()
    print("fitness before local opt:", fitness(ideal_eq))
    print("fitness after local opt:", local_opt_fitness(ideal_eq))
    print("ideal eq after optim:", ideal_eq)

    opt_result = island.evolve_until_convergence(max_generations=500,
                                                 fitness_threshold=1.0e-5)

    print(local_opt_fitness.eval_count)
    print(opt_result)
    best_indiv = island.get_best_individual()
    print(best_indiv.get_formatted_string("console"))
    print(best_indiv.fitness)


if __name__ == '__main__':
    import random
    random.seed(7)
    np.random.seed(7)

    start_time = time.perf_counter_ns()
    main(use_pytorch=True)
    end_time = time.perf_counter_ns()
    print("duration:", (end_time - start_time) / 1e9)
