import numpy as np

from bingo.symbolic_regression.agraphMD.validation_backend.operator_validate import validate_operator, is_scalar_shape


def validate_individual(individual):
    return validate(individual.command_array, individual.input_dims, individual.output_dim, individual.constant_shapes)


def validate(stack, x_dims, output_dim, constant_dims):
    dimensions = np.empty((len(stack), 2))
    for i, (node, param1, param2, param3) in enumerate(stack):
        valid, dimensions[i] = validate_operator(node, param1, param2, param3, dimensions, x_dims, constant_dims)
        if not valid:
            return False
    last_dim = tuple(dimensions[-1])
    if last_dim != output_dim:
        if is_scalar_shape(output_dim):
            return is_scalar_shape(last_dim)
        return False
    return True


def get_stack_command_dimensions(stack, x_dims, constant_dims):
    dimensions = np.empty((len(stack), 2))
    for i, (node, param1, param2, param3) in enumerate(stack):
        valid, dimensions[i] = validate_operator(node, param1, param2, param3, dimensions, x_dims, constant_dims)
        if not valid:
            raise RuntimeError("stack not valid")
    dimensions = [tuple(dim) for dim in dimensions]
    return dimensions
