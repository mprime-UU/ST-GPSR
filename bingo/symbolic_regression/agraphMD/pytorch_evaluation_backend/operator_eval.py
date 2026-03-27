import torch

from bingo.symbolic_regression.agraphMD.operator_definitions import *


# Integer value
def _integer_forward_eval(param1, param2, param3, x, _constants, _forwardeval):
    returned_integer = torch.tensor(float(param1))
    if (param2, param3) != (0, 0):
        returned_integer *= torch.ones(param2, param3)
    return torch.unsqueeze(returned_integer, 0).expand(len(x[0]), *([-1]*returned_integer.ndim))


# Load x column
def _loadx_forward_eval(param1, _param2, param3, x, _constants, _forwardeval):
    return x[param1]


# Load constant
def _loadc_forward_eval(param1, _param2, param3, x, constants, _forwardeval):
    return constants[param1]


# Addition
# def _add_forward_eval(param1, param2, param3, _x, _constants, forward_eval):
#     arg1 = forward_eval[param1]
#     arg2 = forward_eval[param2]
#     arg1_one_by_one = (arg1.size()[1:] == (1, 1))
#     arg2_one_by_one = (arg2.size()[1:] == (1, 1))
#     if arg1_one_by_one and not arg2_one_by_one:
#         return forward_eval[param1] + forward_eval[param2].view(-1, 1, 1)
#     elif not arg1_one_by_one and arg2_one_by_one:
#         return forward_eval[param1].view(-1, 1, 1) + forward_eval[param2]
#     else:
#         return forward_eval[param1] + forward_eval[param2]

def _add_forward_eval(param1, param2, param3, _x, _constants, forward_eval):
    arg1 = forward_eval[param1]
    arg2 = forward_eval[param2]
    arg1_dim = arg1.dim()
    arg2_dim = arg2.dim()
    if (arg1_dim == arg2_dim):
        return arg1 + arg2
    elif (arg1_dim == 1) and (arg2_dim == 3):
        return arg1.view(-1, 1, 1) + arg2
    else:
        return arg1 + arg2.view(-1, 1, 1)


# Subtraction
def _subtract_forward_eval(param1, param2, param3, _x, _constants, forward_eval):
    arg1 = forward_eval[param1]
    arg2 = forward_eval[param2]
    arg1_dim = arg1.dim()
    arg2_dim = arg2.dim()
    if (arg1_dim == arg2_dim):
        return arg1 - arg2
    elif (arg1_dim == 1) and (arg2_dim == 3):
        return arg1.view(-1, 1, 1) - arg2
    else:
        return arg1 - arg2.view(-1, 1, 1)


# Multiplication
def _multiply_forward_eval(param1, param2, param3, _x, _constants, forward_eval):
    fe_1 = forward_eval[param1]
    fe_2 = forward_eval[param2]

    # resizing to deal w/ scalar array cases
    len_size_1 = len(fe_1.size())
    len_size_2 = len(fe_2.size())
    if len_size_1 in {1, 2} or len_size_2 in {1, 2}:
        if len_size_1 == 1 and not len_size_2 == 1:
            # return fe_1[:, None, None] * fe_2
            return fe_1.view(-1, 1, 1) * fe_2
        if len_size_2 == 1 and not len_size_1 == 1:
            # return fe_1 * fe_2[:, None, None]
            return fe_1 * fe_2.view(-1, 1, 1)
        return fe_1 * fe_2

    if fe_1.size()[1:] == (1, 1) or fe_2.size()[1:] == (1, 1):
        return fe_1 * fe_2
    return fe_1 @ fe_2


# Division
def _divide_forward_eval(param1, param2, param3, _x, _constants, forward_eval):
    return forward_eval[param1] / forward_eval[param2]


# Sine
def _sin_forward_eval(param1, _param2, param3, _x, _constants, forward_eval):
    return torch.sin(forward_eval[param1])


# Cosine
def _cos_forward_eval(param1, _param2, param3, _x, _constants, forward_eval):
    return torch.cos(forward_eval[param1])


# Hyperbolic Sine
def _sinh_forward_eval(param1, _param2, param3, _x, _constants, forward_eval):
    return torch.sinh(forward_eval[param1])


# Hyperbolic Cosine
def _cosh_forward_eval(param1, _param2, param3, _x, _constants, forward_eval):
    return torch.cosh(forward_eval[param1])


# Exponential
def _exp_forward_eval(param1, _param2, param3, _x, _constants, forward_eval):
    return torch.exp(forward_eval[param1])


# Natural logarithm
def _log_forward_eval(param1, _param2, param3, _x, _constants, forward_eval):
    return torch.log(torch.abs(forward_eval[param1]))


# Power
def _pow_forward_eval(param1, param2, param3, _x, _constants, forward_eval):
    return torch.pow(forward_eval[param1], forward_eval[param2])


# Safe Power
def _safe_pow_forward_eval(param1, param2, param3, _x, _constants, forward_eval):
    return torch.pow(torch.abs(forward_eval[param1]), forward_eval[param2])


# Absolute value
def _abs_forward_eval(param1, _param2, param3, _x, _constants, forward_eval):
    return torch.abs(forward_eval[param1])


# Square root
def _sqrt_forward_eval(param1, _param2, param3, _x, _constants, forward_eval):
    return torch.sqrt(torch.abs(forward_eval[param1]))


def _transpose_forward_eval(param1, param2, param3, x, constants, forward_eval):
    dim = forward_eval[param1].dim()
    if dim == 1:
        return forward_eval[param1]
    else:
        return forward_eval[param1].mT


def _arctan_forward_eval(param1, param2, param3, x, constants, forward_eval):
    return torch.arctan(forward_eval[param1])


def _arccos_forward_eval(param1, param2, param3, x, constants, forward_eval):
    return torch.arccos(forward_eval[param1])


def _get_dim_idx_of_three(arr):
    return int((torch.tensor(arr.size()) == 3).nonzero()[0][0])


def _cross_forward_eval(param1, param2, param3, x, constants, forward_eval):
    first_vec = forward_eval[param1]
    second_vec = forward_eval[param2]

    param1_dim_with_three = _get_dim_idx_of_three(first_vec)
    param2_dim_with_three = _get_dim_idx_of_three(second_vec)
    assert param1_dim_with_three == param2_dim_with_three

    return torch.cross(first_vec, second_vec, dim=param1_dim_with_three)


def _normalize_forward_eval(param1, param2, param3, x, constants, forward_eval):
    vectors = forward_eval[param1]
    dim_with_three = _get_dim_idx_of_three(vectors)
    magnitude = torch.linalg.norm(vectors, dim=dim_with_three)

    return vectors / magnitude[:, None]


def _element_wise_forward_eval(param1, param2, param3, x, constants, forward_eval):
    return forward_eval[param1] * forward_eval[param2]


def _matrix_vec_forward_eval(param1, param2, param3, x, constants, forward_eval):
    return forward_eval[param1] @ forward_eval[param2]

def forward_eval_function(node, param1, param2, param3, x, constants, forward_eval):
    """Performs calculation of one line of stack"""
    # IMPORTANT: Assumes x is column-major
    return FORWARD_EVAL_MAP[node](param1, param2, param3, x, constants, forward_eval)


# Node maps
FORWARD_EVAL_MAP = {INTEGER: _integer_forward_eval,
                    VARIABLE: _loadx_forward_eval,
                    CONSTANT: _loadc_forward_eval,
                    ADDITION: _add_forward_eval,
                    SUBTRACTION: _subtract_forward_eval,
                    MULTIPLICATION: _multiply_forward_eval,
                    DIVISION: _divide_forward_eval,
                    SIN: _sin_forward_eval,
                    COS: _cos_forward_eval,
                    SINH: _sinh_forward_eval,
                    COSH: _cosh_forward_eval,
                    EXPONENTIAL: _exp_forward_eval,
                    LOGARITHM: _log_forward_eval,
                    POWER: _pow_forward_eval,
                    ABS: _abs_forward_eval,
                    SQRT: _sqrt_forward_eval,
                    SAFE_POWER: _safe_pow_forward_eval,
                    TRANSPOSE: _transpose_forward_eval,
                    ARCTAN: _arctan_forward_eval,
                    ARCCOS: _arccos_forward_eval,
                    CROSS: _cross_forward_eval,
                    NORMALIZE: _normalize_forward_eval,
                    ELEMENTWISE_MULT: _element_wise_forward_eval,
                    MATRIX_VEC_MULT: _matrix_vec_forward_eval}
