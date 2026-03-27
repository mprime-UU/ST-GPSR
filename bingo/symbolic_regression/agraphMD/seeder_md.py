import numpy as np

from .agraphMD import AGraphMD
from .pytorch_agraph_md import PytorchAGraphMD
from .validation_backend import validation_backend
from ...chromosomes.generator import Generator
from ...util.argument_validation import argument_validation

class AGraphSeederMD(Generator):

    def __init__(self, command_array, input_dims, output_dim, use_simplification=False,
                 use_pytorch=False, use_symmetric_constants=False):
        self.command_array = command_array
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
        individual = self._backend_generator_function()
        individual.command_array = self.command_array
        individual._update()
        if not validation_backend.validate_individual(individual):
            raise RuntimeError("Invalid seed command array")
    
        return individual
