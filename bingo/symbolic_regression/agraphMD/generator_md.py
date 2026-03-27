"""Generator of acyclic graph individuals.

This module contains the implementation of the generation of random acyclic
graph individuals.
"""
import numpy as np

from .agraphMD import AGraphMD
from .pytorch_agraph_md import PytorchAGraphMD
from .validation_backend import validation_backend
from ...chromosomes.generator import Generator
from ...util.argument_validation import argument_validation


class AGraphGeneratorMD(Generator):
    """Generates acyclic graph individuals

    Parameters
    ----------
    agraph_size : int
                  command array size of the generated acyclic graphs
    component_generator : agraph.ComponentGenerator
                          Generator of stack components of agraphs
    """
    @argument_validation(agraph_size={">=": 1})
    def __init__(self, agraph_size, component_generator, input_dims, output_dim, use_simplification=False,
                 use_pytorch=False, use_symmetric_constants=False):
        self.agraph_size = agraph_size
        self.component_generator = component_generator
        self._input_dims = input_dims
        if output_dim == () or output_dim == (0, 0):
            output_dim = (1, 1)
        self._output_dim = output_dim
        self._use_simplification = use_simplification
        self._use_symmetric_constants = use_symmetric_constants

        self._backend_generator_function = self._python_generator_function
        if use_pytorch:
            self._backend_generator_function = self._pytorch_generator_function

    def __call__(self):
        """Generates random agraph individual.

        Fills stack based on random commands from the component generator.

        Returns
        -------
        Agraph
            new random acyclic graph individual
        """
        return self._create_individual()

    def _python_generator_function(self):
        return AGraphMD(input_dims=self._input_dims,
                        output_dim=self._output_dim,
                        use_simplification=self._use_simplification,
                        use_symmetric_constants=self._use_symmetric_constants)

    def _pytorch_generator_function(self):
        return PytorchAGraphMD(input_dims=self._input_dims,
                               output_dim=self._output_dim,
                               use_simplification=self._use_simplification,
                               use_symmetric_constants=self._use_symmetric_constants)

    def _create_individual(self):
        attempts = 0
        individual = self._generate_potential_individual()
        while not validation_backend.validate_individual(individual):
            if attempts >= 100000:
                raise RuntimeError("Could not generate valid agraph within 10000 attempts")  # TODO test
            individual = self._generate_potential_individual()
            attempts += 1
            # print('Attempts: ', attempts)
        # print(individual.get_complexity(), attempts)
        return individual

    def _generate_potential_individual(self):
        command_array = np.empty((self.agraph_size, 4), dtype=int)
        for i in range(self.agraph_size):
            command_array[i] = self.component_generator.random_command_full(i)
        individual = self._backend_generator_function()
        individual.command_array = command_array
        individual._update()  # TODO testing for this
        return individual
