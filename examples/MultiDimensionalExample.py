import numpy as np
from bingo.local_optimizers.continuous_local_opt_md import ContinuousLocalOptimizationMD

from bingo.symbolic_regression.explicit_regression_md import ExplicitTrainingDataMD, ExplicitRegressionMD

from bingo.symbolic_regression.agraphMD.agraphMD import AGraphMD
from bingo.symbolic_regression.agraphMD.operator_definitions import *

if __name__ == '__main__':
    C = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    # x = np.array([[[[0], [1], [2]]]])
    N = 10
    x = np.random.randn(N, 3, 1).reshape((1, N, 3, 1))
    y = C @ x[0]
    print("true y", y)

    etd = ExplicitTrainingDataMD(x, y)
    fitness = ExplicitRegressionMD(etd, metric="mse")
    clo = ContinuousLocalOptimizationMD(fitness, algorithm="lm", param_init_bounds=[0, 0])

    test_graph = AGraphMD(input_dims=(3, 1), output_dim=y[0].shape)
    test_graph.command_array = np.array([[CONSTANT, 0, 3, 3],
                                         [VARIABLE, 0, 3, 1],
                                         [MULTIPLICATION, 0, 1, 0]], dtype=int)
    test_graph._update()
    test_graph.set_local_optimization_params(C.flatten(), [(3, 3)])
    print(test_graph.constants)
    print("pred y:", test_graph.evaluate_equation_at(x))
    print(fitness(test_graph))
    print(fitness.evaluate_fitness_vector(test_graph))
    test_graph._needs_opt = True

    print(clo(test_graph))
    print(test_graph.constants)
    # print(test_graph.evaluate_equation_at(x))
