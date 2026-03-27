# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import numpy as np
import random

from bingo.evolutionary_algorithms.age_fitness import AgeFitnessEA
from bingo.evolutionary_optimizers.serial_archipelago import SerialArchipelago
from bingo.evaluation.evaluation import Evaluation
from bingo.evolutionary_optimizers.island import Island

from bingo.local_optimizers.continuous_local_opt_md import ContinuousLocalOptimizationMD
from bingo.symbolic_regression.agraphMD.component_generator_md import ComponentGeneratorMD
from bingo.symbolic_regression.agraphMD.crossover_md import AGraphCrossoverMD
from bingo.symbolic_regression.agraphMD.generator_md import AGraphGeneratorMD
from bingo.symbolic_regression.agraphMD.mutation_md import AGraphMutationMD
from bingo.symbolic_regression.explicit_regression_md import ExplicitTrainingDataMD
from bingo.symbolic_regression.mapping_fitness_md import MappingFitness

POP_SIZE = 250
STACK_SIZE = 10


def get_input_hill_matrices(F, G, H):
    p_hill = np.array([[H + G, -H, -G],
                       [-H, F + H, -F],
                       [-G, -F, F + G]])
    return p_hill


def get_von_mises_p(yield_stress):
    p_vm = np.array([[1, -0.5, -0.5],
                     [-0.5, 1, -0.5],
                     [-0.5, -0.5, 1]])
    return 1/yield_stress**2 * p_vm


def execute_generational_steps():
    N = 10
    F = np.linspace(-10, 10, num=N)
    G = np.linspace(-20, 0, num=N)
    H = np.linspace(20, 0, num=N)

    hill_matrices = []
    for f in F:
        for g in G:
            for h in H:
                hill_matrices.append(get_input_hill_matrices(f, g, h))
    hill_matrices = np.array(hill_matrices)
    x = [hill_matrices]
    y = np.repeat(get_von_mises_p(525).reshape((1, 3, 3)), len(x[0]), axis=0)

    x_dims = [np.shape(_x[0]) for _x in x]
    y_dim = np.shape(y[0])
    print("Dimensions of X variables:", x_dims)
    print("Dimension of output:", y_dim)

    component_generator = ComponentGeneratorMD(x_dims)
    component_generator.add_operator("+")
    component_generator.add_operator("*")
    component_generator.add_operator("sin")
    component_generator.add_operator("cos")

    crossover = AGraphCrossoverMD()
    mutation = AGraphMutationMD(component_generator)
    agraph_generator = AGraphGeneratorMD(STACK_SIZE, component_generator, x_dims, y_dim, use_simplification=False)

    training_data = ExplicitTrainingDataMD(x, y)
    fitness = MappingFitness(training_data)
    local_opt_fitness = ContinuousLocalOptimizationMD(fitness, algorithm='BFGS', param_init_bounds=[0, 0])
    evaluator = Evaluation(local_opt_fitness)

    from bingo.symbolic_regression.agraphMD.agraphMD import AGraphMD
    ideal_eq = AGraphMD(x_dims, y_dim)
    ideal_eq.command_array = np.array([[1, 0, 3, 3]])
    ideal_eq._update()
    closed_form_mapping = np.array([[0.00102971, 0.00036147, 0.],
                                    [0.00036147, 0.00067918, 0.],
                                    [0., 0., 0.00015701]])
    ideal_eq.set_local_optimization_params(closed_form_mapping.flatten(), [(3, 3)])
    print("first_FGH_ideal fitness:", fitness(ideal_eq))

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
    random.seed(7)
    np.random.seed(7)
    execute_generational_steps()
