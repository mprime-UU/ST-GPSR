# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import numpy as np
import pytest

from bingo.symbolic_regression.agraphMD.operator_definitions import *
from bingo.symbolic_regression.agraphMD.evaluation_backend import evaluation_backend

OPERATOR_LIST = [INTEGER, VARIABLE, CONSTANT, ADDITION, SUBTRACTION, TRANSPOSE]


@pytest.fixture
def sample_x():
    return [-1.0,
            [[-1, 2, -3],
             [4, -5, 6],
             [-7, 8, -9]],
            [[1, 2, 3],
             [4, 5, 6]]]


@pytest.fixture
def sample_constants():
    return [np.array([[1, 1, 1],
                      [2, 2, 2],
                      [3, 3, 3]]), np.array(3.14)]


def _terminal_evaluations(terminal, x, constants, dims):
    # assumes parameter is 0
    if terminal == INTEGER:
        return np.zeros(dims)
    elif terminal == VARIABLE:
        return x[0]
    elif terminal == CONSTANT:
        return constants[0]
    raise NotImplementedError("No test for terminal: %d" % terminal)


def _function_evaluations(function, a, b):
    # returns: f(a,b)
    # or if the function takes only a single parameter: f(a)
    if function == ADDITION:
        return a + b
    elif function == SUBTRACTION:
        return a - b
    elif function == MULTIPLICATION:
        if np.shape(a) and np.shape(b):
            return a @ b
        else:
            return a * b
    elif function == TRANSPOSE:
        if len(np.shape(a)) == 3:
            return np.transpose(a, (0, 2, 1))
        else:
            return np.transpose(a)
    raise NotImplementedError("No test for operator: %d" % function)


@pytest.mark.parametrize("param", [-1, 0, 1])
@pytest.mark.parametrize("dims", [(3, 3), (1, 1), (3, 2), (0, 0)])
def test_integer_load_evaluate(param, dims):
    stack = np.array([[INTEGER, param, *dims]], dtype=int)
    if dims == (0, 0):
        return param
    else:
        expected_outcome = param * np.ones(dims)

    f_of_x = evaluation_backend.evaluate(stack, [], [])
    assert np.array_equal(expected_outcome, f_of_x) is True


@pytest.mark.parametrize("param", range(3))  # should be range(len(sample_x)) but you can't
# use fixtures in parametrize for some reason?
def test_variable_load_evaluate(sample_x, param):
    stack = np.array([[VARIABLE, param, 0, 0]], dtype=int)
    expected_outcome = sample_x[param]

    f_of_x = evaluation_backend.evaluate(stack, sample_x, [])
    assert np.array_equal(expected_outcome, f_of_x) is True


@pytest.mark.parametrize("param", range(2))  # likewise should be range(len(sample_constants))
def test_constant_load_evaluate(sample_constants, param):
    stack = np.array([[CONSTANT, param, 0, 0]], dtype=int)
    expected_outcome = sample_constants[param][None]

    f_of_x = evaluation_backend.evaluate(stack, [[1]], sample_constants)
    assert np.array_equal(expected_outcome, f_of_x) is True


@pytest.mark.parametrize("operator", [ADDITION, SUBTRACTION])
@pytest.mark.parametrize("x_1, x_2", [(-1.0 * np.ones((3, 3)), np.ones((3, 3))),
                                      (1.0, 2.0)])
def test_addition_like_evaluate(operator, x_1, x_2):
    stack = np.array([[VARIABLE, 0, 0, 0],
                      [VARIABLE, 1, 0, 0],
                      [operator, 0, 1, 0]])
    expected_outcome = _function_evaluations(operator, x_1, x_2)
    f_of_x = evaluation_backend.evaluate(stack, [x_1, x_2], [])
    assert np.array_equal(expected_outcome, f_of_x) is True


@pytest.mark.parametrize("x_1, x_2", [(np.array(1.0), np.array(-1.5)),
                                      (np.array(1.0), np.ones((3, 3))),
                                      (np.ones((3, 3)), np.array(1.0)),
                                      (np.ones((3, 2)), np.ones((2, 4))),
                                      (np.ones((1, 1)), np.ones((1, 1)))])
def test_multiplication_evaluate(x_1, x_2):
    stack = np.array([[VARIABLE, 0, 0, 0],
                      [VARIABLE, 1, 0, 0],
                      [MULTIPLICATION, 0, 1, 0]])
    expected_outcome = _function_evaluations(MULTIPLICATION, x_1, x_2)
    f_of_x = evaluation_backend.evaluate(stack, [x_1, x_2], [])
    assert np.array_equal(expected_outcome, f_of_x) is True


@pytest.mark.parametrize("x", [np.array([[1, 2, 3],
                                         [4, 5, 6]]),
                               1.0,
                               np.array([1.0]),
                               np.array([[[1, 2, 3]], [[4, 5, 6]]])])
def test_transpose_evaluate(x):
    stack = np.array([[VARIABLE, 0, 0, 0],
                      [TRANSPOSE, 0, 0, 0]])
    expected_outcome = _function_evaluations(TRANSPOSE, x, x)
    f_of_x = evaluation_backend.evaluate(stack, [x], [])
    assert np.array_equal(expected_outcome, f_of_x) is True
