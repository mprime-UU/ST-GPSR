"""
This module represents the python backend associated with the Agraph equation
evaluation.  This backend is used to perform the the evaluation of the equation
represented by an `AGraph`.  It can also perform derivatives.
"""
import numpy as np

import torch
from torch.autograd import grad
from .operator_eval import forward_eval_function

ENGINE = "Python"


def _get_torch_const(constants, data_len):
    with torch.no_grad():
        constants = torch.tensor([constants]).double().expand(data_len, -1).T
        return constants


def get_pytorch_repr(command_array):
    # TODO see if we can do this more efficiently
    # TODO this reruns on every eval, how to return just expression?

    def get_expr(X, constants):  # assumes X is column-order
        expr = []

        for (node, param1, param2) in command_array:
            expr.append(forward_eval_function(node, param1, param2, X, constants,
                                              expr))

        return expr[-1]

    return get_expr


def evaluate(pytorch_repr, x, constants, final=True):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x.T).double()
    if final:
        constants = _get_torch_const(constants, x.size(1))
    return_eval = get_pytorch_repr(pytorch_repr)(x, constants)
    if final:
        return return_eval.detach().numpy().reshape((-1, 1))
    return return_eval


def evaluate_with_derivative(pytorch_repr, x, constants, wrt_param_x_or_c):
    """Evaluate equation and take derivative

    Evaluate the derivatives of the equation associated with an Agraph, at the
    values x.

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
    wrt_param_x_or_c : boolean
        Take derivative with respect to x or constants. True signifies
        derivatives are wrt x. False signifies derivatives are wrt constants.

    Returns
    -------
    MxD array of numeric or MxL array of numeric:
        Derivatives of all dimensions of x/constants at location x.
    """
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x.T).double()
    constants = _get_torch_const(constants, x.size(1))
    eval, deriv = _evaluate_with_derivative(pytorch_repr, x, constants, wrt_param_x_or_c)
    return eval, deriv


def _evaluate_with_derivative(pytorch_repr, x, constants, wrt_param_x_or_c):
    inputs = x
    if not wrt_param_x_or_c:  # c
        inputs = constants
    inputs.requires_grad = True

    eval = evaluate(pytorch_repr, x, constants, final=False)

    if eval.requires_grad:
        derivative = grad(outputs=eval.sum(), inputs=inputs, create_graph=False, retain_graph=False, allow_unused=False)[0]
    else:
        derivative = None
    if derivative is None:
        derivative = torch.zeros((inputs.shape[0], eval.shape[0]))
    return eval.detach().numpy().reshape((-1, 1)), derivative.T.detach().numpy()


def evaluate_with_partials(pytorch_repr, x, constants, partial_order):
    x.requires_grad = True
    constants = _get_torch_const(constants, x.size(1))
    eval = evaluate(pytorch_repr, x, constants, final=False)

    partial = eval
    partials = []
    # TODO check that partial/eval requires grad
    for variable in partial_order:
        try:
            partial = grad(outputs=partial.sum(), inputs=x,
                           allow_unused=True,
                           create_graph=True)[0][variable]
            if partial is None:
                partial = torch.zeros_like(x[0])
        except IndexError:
            partial = torch.zeros_like(x[0])
        partials.append(partial.reshape((-1, 1)).detach().numpy())

    return eval.detach().numpy().reshape((-1, 1)), partials
