"""Component Generator for Agraph equations.

This module covers the random generation of components of an Acyclic graph
command stack. It can generate full commands or sub-components such as
operators, terminals, and their associated parameters.
"""
import logging
import numpy as np

from .operator_definitions import *
from .validation_backend.operator_validate import validate_operator
from .validation_backend.validation_backend import get_stack_command_dimensions
from ...util.probability_mass_function import ProbabilityMassFunction
from ...util.argument_validation import argument_validation

LOGGER = logging.getLogger(__name__)


class ComponentGeneratorMD:
    """Generates commands or components of a command for an agraph stack

    Parameters
    ----------
    input_x_dimension :
        number of independent variables
    num_initial_load_statements : int
        number of commands at the beginning of stack which are required to be
        terminals. Default 1
    terminal_probability : float [0.0-1.0]
        probability that a new node will be a terminal. Default 0.1
    constant_probability : float [0.0-1.0] (optional)
        probability that a new terminal will be a constant

    Attributes
    ----------
    input_x_dimension :
        number of independent variables
    """
    @argument_validation(num_initial_load_statements={">=": 1},
                         terminal_probability={">=": 0.0, "<=": 1.0},
                         constant_probability={">=": 0.0, "<=": 1.0})
    def __init__(self, input_x_dimensions=None,
                 input_y_dimension=None,
                 possible_dims=None,
                 possible_dim_weights=None,
                 x_weights=None,
                 num_initial_load_statements=1,
                 terminal_probability=0.1,
                 constant_probability=None):
        # TODO can we verify len of input_x_dimensions/
        #  each tuple in input_x_dimensions
        if input_x_dimensions is None:
            input_x_dimensions = []
        input_x_dimensions = [input_dim if input_dim != () else (0, 0) for input_dim in input_x_dimensions]
        self.input_x_dimensions = np.array(input_x_dimensions)
        self.x_dims = [tuple(row) for row in self.input_x_dimensions]
        self.input_y_dimension = input_y_dimension
        self.possible_dims = possible_dims
        self.possible_dim_weights = possible_dim_weights
        self.x_weights = x_weights
        self.num_x = len(input_x_dimensions)
        self._num_initial_load_statements = num_initial_load_statements

        # self._terminals = self.initiate_terminals(terminal_probability)
        self._dim_stack = []
        self._pos_operator_pmf = ProbabilityMassFunction()
        self._terminal_pmf = self._make_terminal_pdf(constant_probability)
        self._operator_pmf = ProbabilityMassFunction()
        self._random_command_function_pmf = \
            self._make_random_command_pmf(terminal_probability)

    def _make_terminal_pdf(self, constant_probability):
        if constant_probability is None:
            terminal_weight = [1, self.num_x]
        else:
            terminal_weight = [constant_probability,
                               1.0 - constant_probability]
        return ProbabilityMassFunction(items=[1, 0], weights=terminal_weight)

    def _make_random_command_pmf(self, terminal_probability):
        command_weights = [terminal_probability,
                           1.0 - terminal_probability]
        return ProbabilityMassFunction(items=[self.random_terminal_command,
                                              self.random_operator_command],
                                       weights=command_weights)

    def add_operator(self, operator_to_add, operator_weight=None):
        """Add an operator number to the set of possible operators

        Parameters
        ----------
        operator_to_add : int, str
            operator integer code (e.g. 2, 3) defined in Agraph operator maps
            or an operator string description (e.g. "+", "addition")
        operator_weight : number
                          relative weight of operator probability
        """
        if isinstance(operator_to_add, str):
            operator_number = self._get_operator_number_from_string(
                operator_to_add)
        else:
            operator_number = operator_to_add

        self._operator_pmf.add_item(operator_number, operator_weight)

    @staticmethod
    def _get_operator_number_from_string(operator_string):
        for operator_number, operator_names in OPERATOR_NAMES.items():
            if operator_string in operator_names:
                return operator_number
        raise ValueError("Could not find operator %s. " % operator_string)
        
    # def initiate_terminals(self, terminal_probability):
    #     terminals = []
    #     num_terminals = np.random.randint((self.stack_size*terminal_probability))
    #     for i in range(self._num_initial_load_statements):
    #         terminals.append(self.random_terminal_command(i))
        
    #     for i in range(self.num_initial_load_statements, num_terminals):
    #         terminals.append(self.random_terminal_command(i))
            
    #     return terminals
    
    def possible_operators(self):
        return self._operator_pmf.items
        
    def random_command(self, stack_location, dependent_stack):
        """Get a random command

        Parameters
        ----------
        stack_location : int
            location in the stack for the command

        Returns
        -------
        array of int
            a random command in the form [node, parameter 1, parameter 2, parameter 3]

        """
        if stack_location < self._num_initial_load_statements:
            return self.random_terminal_command(stack_location)
        
        self._pos_operator_pmf = ProbabilityMassFunction()

        constant_dims = []
        for i, (node, param1, param2, param3) in enumerate(dependent_stack):
            if node == 1:
                constant_dims.append((param2, param3))
        
        stack_dimensions = get_stack_command_dimensions(dependent_stack, self.x_dims, constant_dims)

        for i, dim1 in enumerate(stack_dimensions):
            for j, dim2 in enumerate(stack_dimensions):
                for operator in self.possible_operators():
                    
                    dimensions = np.array([dim1,
                                            dim2])
                    valid, op_dims = validate_operator(operator, 0, 1, 0, dimensions, 0, 0)
                    if valid:
                        self._pos_operator_pmf.add_item([operator, i, j, np.array(op_dims)])
                    
                    dimensions = np.array([dim2,
                                            dim1])
                    valid, op_dims = validate_operator(operator, 0, 1, 0, dimensions, 0, 0)
                    if valid:
                        self._pos_operator_pmf.add_item([operator, j, i, np.array(op_dims)])

        sample = self._pos_operator_pmf.draw_sample()
        return np.array([sample[0], sample[1], sample[2], 0])

    def random_command_full(self, stack_location):
        """Get a random command

        Parameters
        ----------
        stack_location : int
            location in the stack for the command

        Returns
        -------
        array of int
            a random command in the form [node, parameter 1, parameter 2, parameter 3]

        """
        # if stack_location < self._num_initial_load_statements:
        #     return self.random_terminal_command(stack_location)
        # return self._random_command_function_pmf.draw_sample()(stack_location)
        # print(len(self._dim_stack))
        if stack_location == 0:
            self._dim_stack = []
            self._pos_operator_pmf = ProbabilityMassFunction()
        if stack_location < self._num_initial_load_statements:
            if stack_location == 0 and self.input_y_dimension is not None:
                command = np.array([1, 0, self.input_y_dimension[0], self.input_y_dimension[1]])
                self._dim_stack.append(np.array(self.input_y_dimension))
                return command
            command = self.random_terminal_command(stack_location)
            self._dim_stack.append(np.array([command[2], command[3]]))
            return command
        else:
            random_command = self._random_command_function_pmf.draw_sample()
            if random_command == self.random_terminal_command:
                command = self.random_terminal_command(stack_location)
                self._dim_stack.append(np.array([command[2], command[3]]))
                return command
            else:
                for i, dim1 in enumerate(self._dim_stack):
                    for operator in self.possible_operators():
                        dim2 = self._dim_stack[-1]
                        
                        dimensions = np.array([dim1,
                                               dim2])
                        valid, op_dims = validate_operator(operator, 0, 1, 0, dimensions, 0, 0)
                        if valid:
                            self._pos_operator_pmf.add_item([operator, i, stack_location-1, np.array(op_dims)])
                        
                        dimensions = np.array([dim2,
                                               dim1])
                        valid, op_dims = validate_operator(operator, 0, 1, 0, dimensions, 0, 0)
                        if valid:
                            self._pos_operator_pmf.add_item([operator, stack_location-1, i, np.array(op_dims)])
                        
                # if stack_location == 29:
                #     attempts = 0
                #     sample = self._pos_operator_pmf.draw_sample()
                #     while not np.array_equal(sample[3], [3,3]):
                #         if attempts > 50:
                #             return np.array([sample[0], sample[1], sample[2], 0])
                #         sample = self._pos_operator_pmf.draw_sample()
                #         attempts += 1
                # else:
                #     sample = self._pos_operator_pmf.draw_sample()

                sample = self._pos_operator_pmf.draw_sample()
                self._dim_stack.append(np.array(sample[3]))
                return np.array([sample[0], sample[1], sample[2], 0])
                
                

    def random_operator_command(self, stack_location):
        """Get a random operator (non-terminal) command

        Parameters
        ----------
        stack_location : int
            location in the stack for the command

        Returns
        -------
        array of int
            a random command in the form [node, parameter 1, parameter 2, parameter 3]

        """
        return np.array([self.random_operator(),
                         self.random_operator_parameter(stack_location),
                         self.random_operator_parameter(stack_location),
                         0],
                        dtype=int)

    def random_operator(self):
        """Get a random operator

         Get a random operator from the list of possible operators.

        Returns
        -------
        int
            an operator number
        """
        return self._operator_pmf.draw_sample()

    @staticmethod
    def random_operator_parameter(stack_location):
        """Get random operator parameter

        Parameters
        ----------
        stack_location : int
            location of command in stack

        Returns
        -------
        int
            parameter to be used in an operator command


        Notes
        -----
        The returned random operator parameter is guaranteed to be less than
        stack_location.
        """
        return np.random.randint(stack_location)

    def random_terminal_command(self, _=None):
        """Get a random terminal (non-operator) command

        Returns
        -------
        array of int
            a random command in the form [node, parameter 1, parameter 2, parameter 3]

        """
        terminal = self.random_terminal()
        param = self.random_terminal_parameter(terminal)
        if terminal == VARIABLE:
            dim1, dim2 = self.input_x_dimensions[param]
        else:
            dim1, dim2 = self.random_dims()
        return np.array([terminal, param, dim1, dim2], dtype=int)

    def random_dims(self):
        if not self.possible_dims:
            max_dim = max(self.input_x_dimensions.flatten())
            dim1 = np.random.randint(0, max_dim + 1)
            if dim1 == 0:
                dim2 = 0
            else:
                dim2 = np.random.randint(1, max_dim + 1)
            return dim1, dim2
        else:
            if self.possible_dim_weights:
                possible_idx = np.arange(len(self.possible_dims))
                chosen_idx = np.random.choice(possible_idx, p=self.possible_dim_weights)
                return self.possible_dims[chosen_idx]
            else:
                return self.possible_dims[np.random.randint(len(self.possible_dims))]

    def random_terminal(self):
        """Get a random terminal

         Get a random VARIABLE or CONSTANT terminal.

        Returns
        -------
        int
            terminal number (VARIABLE or CONSTANT)
        """
        return self._terminal_pmf.draw_sample()

    def random_terminal_parameter(self, terminal_number):
        """Get random terminal parameter

        Parameters
        ----------
        terminal_number : int
            terminal number for which random parameter should be generated

        Returns
        -------
        int
            parameter to be used in a terminal command
        """
        if terminal_number == 0:
            if self.x_weights:
                param = np.random.choice(np.arange(self.num_x), p=self.x_weights)
            else:
                param = np.random.randint(self.num_x)
        else:
            param = -1
        return param

    def get_number_of_terminals(self):
        """Gets number of possible terminals

        Returns
        -------
        int :
            number of terminals
        """
        return len(self._terminal_pmf.items)

    def get_number_of_operators(self):
        """Gets number of possible operators

        Returns
        -------
        int :
            number of operators
        """
        return len(self._operator_pmf.items)
