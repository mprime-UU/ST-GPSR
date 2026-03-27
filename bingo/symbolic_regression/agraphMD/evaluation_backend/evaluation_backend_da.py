"""
This module represents the python backend associated with the Agraph equation
evaluation.  This backend is used to perform the the evaluation of the equation
represented by an `AGraph`.  It can also perform derivatives.
"""
import numpy as np
from .operator_eval_da import forward_eval_function

ENGINE = "Python"


def evaluate(stack, x, constants):
    """Evaluate an equation

    Evaluate the equation associated with an Agraph, at the values x.

    Parameters
    ----------
    stack : Nx3 numpy array of int.
        The command stack associated with an equation. N is the number of
        commands in the stack.
    x : MxD array of numeric.
        Values at which to evaluate the equations. D is the number of
        dimensions in x and M is the number of data points in x.
    constants : list-like of numeric.
        numeric constants that are used in the equation

    Returns
    -------
    Mx1 array of numeric
        :math`f(x)`
    """
    forward_eval, num_violations = _forward_eval(stack, x, constants)
    return forward_eval[-1], num_violations

# def _forward_eval(stack, x, constants):
#     forward_eval = []
#     for i, (node, param1, param2, param3) in enumerate(stack):
#         forward_eval.append(forward_eval_function(node, param1, param2, param3, x, constants, forward_eval))
#     return forward_eval

def _forward_eval(stack, x, constants):
    forward_eval = []
    num_violations = 0
    for i, (node, param1, param2, param3) in enumerate(stack):
        forward_eval.append(forward_eval_function(node, param1, param2, x,
                                                constants, forward_eval))
        
        if np.all(forward_eval[i] == 99.):
            num_violations += 1
        # print(stack[i,:],forward_eval[i])
    return forward_eval, num_violations



