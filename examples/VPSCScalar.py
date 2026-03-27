
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
    my_graph = AGraph(equation="X_3 * (C_0 * X_0 * X_0 + "
                               "C_1 * X_0 * X_1 + "
                               "C_2 * X_0 * X_2 + "
                               "C_3 * X_1 * X_1 + "
                               "C_4 * X_1 * X_2 + "
                               "C_5 * X_2 * X_2)")
    my_graph.set_local_optimization_params([879318.88118913, (412369.19557313 + 424291.95673347), (104907.94647826 + 804722.94102658), -42686.98460765, (-3412.64766773 + -8978.48287208), 30274.30560582])
    return my_graph


# def get_ideal_eq():
#     from bingo.symbolic_regression.agraph.agraph import AGraph
#     my_graph = AGraph(equation="X_0 + X_1 + X_2 + X_3 * C_0")
#     my_graph.set_local_optimization_params([1e-13])
#     return my_graph


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
        prev_point = point_trajectory[0]
        # prev_point = point_trajectory[0]
        for point in point_trajectory:
            dx_dt.append(point - prev_point)
            prev_point = point
        dx_dt.append(point_trajectory[-1] - point_trajectory[-2])
        dx_dt = dx_dt[1:]
        dx_dts.append(dx_dt)


    dx_dts = np.array(dx_dts)
    dx_dts = dx_dts.reshape((-1, dx_dts.shape[2]))
    return dx_dts


def main():
    data = np.loadtxt("vpsc_evo_17_data_3d_points_implicit_format.txt")
    # data[:, :3] /= 50
    x, dx_dt, _ = _calculate_partials(data, window_size=5)

    x = data[np.invert(np.all(np.isnan(data), axis=1))]
    dx_dt = get_dx_dt(data)

    # remove first row of every trajectory where dxdt of eps == 0
    good_rows = np.where(dx_dt[:, 3] != 0)[0]
    x = x[good_rows]
    dx_dt = dx_dt[good_rows]

    training_data = ImplicitTrainingData(x, dx_dt)

    component_generator = ComponentGenerator(x.shape[1])
    component_generator.add_operator("+")
    component_generator.add_operator("-")
    component_generator.add_operator("*")
    # component_generator.add_operator("sin")
    # component_generator.add_operator("cos")
    # component_generator.add_operator("/")

    crossover = AGraphCrossover()
    mutation = AGraphMutation(component_generator)

    agraph_generator = AGraphGenerator(STACK_SIZE, component_generator,
                                       use_simplification=False)

    fitness = ImplicitRegression(training_data=training_data, required_params=4)
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
