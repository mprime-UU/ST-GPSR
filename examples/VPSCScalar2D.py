
# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import numpy as np

from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
from bingo.evolutionary_optimizers.serial_archipelago import SerialArchipelago
from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_optimizers.island import Island
from bingo.local_optimizers.scipy_optimizer import ScipyOptimizer
from bingo.local_optimizers.local_opt_fitness \
    import LocalOptFitnessFunction

from bingo.symbolic_regression import ComponentGenerator, \
                                      AGraphGenerator, \
                                      AGraphCrossover, \
                                      AGraphMutation, \
                                      ExplicitRegression, \
                                      ExplicitTrainingData
from bingo.symbolic_regression.implicit_regression import ImplicitTrainingData, ImplicitRegression
from bingo.symbolic_regression.implicit_regression_md import _calculate_partials
POP_SIZE = 100
STACK_SIZE = 32


def get_ideal_eq():
    from bingo.symbolic_regression.agraph.agraph import AGraph
    my_graph = AGraph(equation="X_2 * (C_0 * X_0 * X_0 + "
                               # "C_1 * X_0 * X_1 + "
                               "C_3 * X_1 * X_1)")
    return my_graph


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
    # data = np.loadtxt("vpsc_evo_16_data_implicit_format.txt")
    data = np.loadtxt("vpsc_evo_16_data_transpose_implicit_format.txt")
    data = np.hstack((data[:, :2], data[:, 3][:, None]))
    x, dx_dt, _ = _calculate_partials(data, window_size=5)

    training_data = ImplicitTrainingData(x, dx_dt)

    component_generator = ComponentGenerator(x.shape[1])
    component_generator.add_operator("+")
    component_generator.add_operator("-")
    component_generator.add_operator("*")
    component_generator.add_operator("sin")
    component_generator.add_operator("cos")
    component_generator.add_operator("/")

    crossover = AGraphCrossover()
    mutation = AGraphMutation(component_generator)

    agraph_generator = AGraphGenerator(STACK_SIZE, component_generator,
                                       use_simplification=False)

    # fitness = ImplicitRegression(training_data=training_data, required_params=3)
    fitness = ImplicitRegression(training_data=training_data)
    optimizer = ScipyOptimizer(fitness, method='lm')
    local_opt_fitness = LocalOptFitnessFunction(fitness, optimizer)

    ideal_eq = get_ideal_eq()
    print("fitness of ideal_eq:", fitness(ideal_eq))
    ideal_eq._needs_opt = True
    print("fitness of ideal_eq after optimization:", local_opt_fitness(ideal_eq))
    print("\t constants after optimization:", repr(ideal_eq.constants))

    evaluator = Evaluation(local_opt_fitness)

    ea = AgeFitnessEA(evaluator, agraph_generator, crossover,
                      mutation, 0.4, 0.4, POP_SIZE)

    island = Island(ea, agraph_generator, POP_SIZE)
    archipelago = SerialArchipelago(island)

    opt_result = archipelago.evolve_until_convergence(max_generations=10000,
                                                      fitness_threshold=1.0e-4)

    best_indv = archipelago.get_best_individual()
    print("found individual:", best_indv, "with fitness:", best_indv.fitness)
    print("\t cmd arr:", repr(best_indv.command_array))
    print("\t constants:", repr(best_indv.constants))

    print(opt_result.ea_diagnostics)


if __name__ == '__main__':
    main()
