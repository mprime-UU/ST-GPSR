"""
This module provides the python implementation of the functions for each
mathematical nodes used in Agraph

Attributes
----------
FORWARD_EVAL_MAP : dictionary {int: function}
                   A map of node number to evaluation function
REVERSE_EVAL_MAP : dictionary {int: function}
                   A map of node number to derivative evaluation function
"""

import numpy as np

from bingo.symbolic_regression.agraphMD.operator_definitions \
    import INTEGER, VARIABLE, CONSTANT, ADDITION, SUBTRACTION, MULTIPLICATION, \
           DIVISION, SIN, COS, SINH, COSH, EXPONENTIAL, LOGARITHM, POWER, ABS, \
           SQRT, SAFE_POWER, CROSS, ELEMENTWISE_MULT, MATRIX_VEC_MULT, MATRIX_MULT


np.seterr(divide='ignore', invalid='ignore')


# Integer value
def _integer_forward_eval(param1, _param2, _x, _constants, _forwardeval):
    return np.full(_x[0].shape, np.nan)


# Load x column
def _loadx_forward_eval(param1, _param2, x, _constants, _forwardeval):
    return x[param1]



# Load constant
def _loadc_forward_eval(param1, _param2, _x, constants, _forwardeval):
    return np.full(_x[0].shape, np.nan)


# Addition
def _add_forward_eval(param1, param2, _x, _constants, forward_eval):
    if np.all(forward_eval[param1] == 99.) or np.all(forward_eval[param2] == 99.):
        return 99
    elif np.array_equal(forward_eval[param1], forward_eval[param2]):
        return forward_eval[param1]
    elif np.any(np.isnan(forward_eval[param1])):
        return forward_eval[param2]
    elif np.any(np.isnan(forward_eval[param2])):
        return forward_eval[param1]
    else:
        return 99


# Subtraction
def _subtract_forward_eval(param1, param2, _x, _constants, forward_eval):
    if np.all(forward_eval[param1] == 99.) or np.all(forward_eval[param2] == 99.):
        return 99
    elif np.array_equal(forward_eval[param1], forward_eval[param2]):
        return forward_eval[param1]
    elif np.any(np.isnan(forward_eval[param1])):
        return forward_eval[param2]
    elif np.any(np.isnan(forward_eval[param2])):
        return forward_eval[param1]
    else:
        return 99


# Multiplication
def _multiply_forward_eval(param1, param2, _x, _constants, forward_eval):
    if np.all(forward_eval[param1] == 99.) or np.all(forward_eval[param2] == 99.):
        return 99
    elif np.any(np.isnan(forward_eval[param1])) or np.any(np.isnan(forward_eval[param2])):
        return np.full(_x[0].shape, np.nan)
    # elif np.any(np.isnan(forward_eval[param2])):
    #     return forward_eval[param1] #- forward_eval[param1]
    else:
        return forward_eval[param1] + forward_eval[param2]



# Division
def _divide_forward_eval(param1, param2, _x, _constants, forward_eval):
    if np.all(forward_eval[param1] == 99.) or np.all(forward_eval[param2] == 99.):
        return 99
    elif np.any(np.isnan(forward_eval[param1])) or np.any(np.isnan(forward_eval[param2])):
        return np.full(_x[0].shape, np.nan)
    # elif np.any(np.isnan(forward_eval[param2])):
    #     return forward_eval[param1] #- forward_eval[param1]
    else:
        return forward_eval[param1] - forward_eval[param2]


# Sine
def _sin_forward_eval(param1, _param2, _x, _constants, forward_eval):
    zeros = np.zeros_like(forward_eval[param1])
    if np.all(forward_eval[param1] == 0.):
        return zeros
    elif np.any(np.isnan(forward_eval[param1])):
        return zeros
    else:
        return 99.


# Cosine
def _cos_forward_eval(param1, _param2, _x, _constants, forward_eval):
    zeros = np.zeros_like(forward_eval[param1])
    if np.all(forward_eval[param1] == 0.):
        return zeros
    elif np.any(np.isnan(forward_eval[param1])):
        return zeros
    else:
        return 99.


# Hyperbolic Sine
def _sinh_forward_eval(param1, _param2, _x, _constants, forward_eval):
    zeros = np.zeros_like(forward_eval[param1])
    if np.all(forward_eval[param1] == 0.):
        return zeros
    elif np.any(np.isnan(forward_eval[param1])):
        return zeros
    else:
        return 99.


# Hyperbolic Cosine
def _cosh_forward_eval(param1, _param2, _x, _constants, forward_eval):
    zeros = np.zeros_like(forward_eval[param1])
    if np.all(forward_eval[param1] == 0.):
        return zeros
    elif np.any(np.isnan(forward_eval[param1])):
        return zeros
    else:
        return 99.


# Exponential
def _exp_forward_eval(param1, _param2, _x, _constants, forward_eval):
    zeros = np.zeros_like(forward_eval[param1])
    if np.all(forward_eval[param1] == 0.):
        return zeros
    elif np.any(np.isnan(forward_eval[param1])):
        return zeros
    else:
        return 99.


# Natural logarithm
def _log_forward_eval(param1, _param2, _x, _constants, forward_eval):
    zeros = np.zeros_like(forward_eval[param1])
    if np.all(forward_eval[param1] == 0.):
        return zeros
    elif np.any(np.isnan(forward_eval[param1])):
        return zeros
    else:
        return 99.


# Power
# DOES NOT WORK
def _pow_forward_eval(param1, param2, _x, _constants, forward_eval):
    if np.all(forward_eval[param1] == 99):
        return 99.
    else:
        return forward_eval[param1] * forward_eval[param2]


# Absolute value
def _abs_forward_eval(param1, _param2, _x, _constants, forward_eval):
    return forward_eval[param1]


# Square root
def _sqrt_forward_eval(param1, _param2, _x, _constants, forward_eval):
    return forward_eval[param1]/2


def forward_eval_function(node, param1, param2, x, constants, forward_eval):
    """Performs calculation of one line of stack"""
    return FORWARD_EVAL_MAP[node](param1, param2, x, constants, forward_eval)


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
                    # TRANSPOSE: _transpose_forward_eval,
                    # ARCTAN: _arctan_forward_eval,
                    # ARCCOS: _arccos_forward_eval,
                    CROSS: _multiply_forward_eval,
                    # NORMALIZE: _normalize_forward_eval,
                    ELEMENTWISE_MULT: _multiply_forward_eval,
                    MATRIX_VEC_MULT: _multiply_forward_eval,
                    MATRIX_MULT: _multiply_forward_eval}

