import numpy as np
import pytest

from bingo.symbolic_regression.agraphMD.operator_definitions import *
from bingo.symbolic_regression.agraphMD.validation_backend.validation_backend import validate


# TODO add testing for new matrix operators (e.g., division, normalize, power, etc.)
# TODO add testing for different scalar types

@pytest.fixture
def complex_stack():
    return np.array([[VARIABLE, 0, 3, 1],        # 0, 3x1
                     [CONSTANT, 0, 3, 1],        # 1, 3x1
                     [TRANSPOSE, 1, 1, 0],       # 2, 1x3
                     [MULTIPLICATION, 0, 2, 0],  # 3, 3x3
                     [CONSTANT, 1, 0, 0],        # 4, 0x0
                     [MULTIPLICATION, 3, 4, 0],  # 5, 3x3
                     [CONSTANT, 2, 3, 3],        # 6, 3x3
                     [ADDITION, 5, 6, 0],        # 7, 3x3
                     [SIN, 7, 0, 0],             # 8, 3x3
                     [SUBTRACTION, 5, 8, 0]], dtype=int)  # 9, 3x3


@pytest.fixture
def complex_stack_load_shapes():
    return [(3, 1)], [(3, 1), (0, 0), (3, 3)]


def test_validate_integer_valid():
    stack = np.array([[INTEGER, 0, 3, 3]], dtype=int)
    assert validate(stack, [], (3, 3), []) is True


def test_validate_integer_invalid():
    stack = np.array([[INTEGER, 0, -1, -2]], dtype=int)
    assert validate(stack, [], (-1, -2), []) is False


def test_validate_load_valid():
    stack = np.array([[CONSTANT, 0, 1, 2],
                      [VARIABLE, 0, 2, 2]], dtype=int)
    assert validate(stack, [(2, 2)], (2, 2), [(1, 2)]) is True


def test_validate_load_invalid_mismatched_dims():
    stack = np.array([[CONSTANT, 0, 1, 2],
                      [VARIABLE, 0, 2, 2]], dtype=int)
    assert validate(stack, [(3, 3)], (2, 2), [(3, 3)]) is False


@pytest.mark.parametrize("node", [VARIABLE, CONSTANT])
@pytest.mark.parametrize("param1", [-2, 2])
def test_validate_load_invalid_bad_param1(node, param1):
    stack = np.array([[node, param1, 2, 2]], dtype=int)
    load_dims = [(2, 2)]
    assert validate(stack, load_dims, (2, 2), load_dims) is False


@pytest.mark.parametrize("node", [ADDITION, SUBTRACTION])
def test_validate_addition_like_valid(node):
    stack = np.array([[CONSTANT, 0, 1, 2],
                      [VARIABLE, 0, 1, 2],
                      [node, 0, 1, 0]], dtype=int)
    assert validate(stack, [(1, 2)], (1, 2), [(1, 2)]) is True


@pytest.mark.parametrize("node", [ADDITION, SUBTRACTION])
def test_validate_addition_like_invalid_different_dims(node):
    stack = np.array([[CONSTANT, 0, 2, 2],
                      [VARIABLE, 0, 1, 2],
                      [node, 0, 1, 0]], dtype=int)
    assert validate(stack, [(1, 2)], (1, 2), [(2, 2)]) is False


@pytest.mark.parametrize("op1_dims, op2_dims", [[(0, 0), (0, 0)],  # scalar * scalar
                                                [(1, 3), (0, 0)],  # matrix * scalar
                                                [(0, 0), (1, 3)],  # scalar * matrix
                                                [(1, 3), (3, 2)]])  # matrix * matrix
def test_validate_multiplication_valid(op1_dims, op2_dims):
    stack = np.array([[CONSTANT, 0, *op1_dims],
                      [CONSTANT, 1, *op2_dims],
                      [MULTIPLICATION, 0, 1, 0]], dtype=int)

    output_dim = (op1_dims[0], op2_dims[1])
    if op1_dims == (0, 0):
        output_dim = op2_dims
    elif op2_dims == (0, 0):
        output_dim = op1_dims
    assert validate(stack, [], output_dim, [op1_dims, op2_dims]) is True


def test_validate_multiplication_invalid():
    # matrix * matrix with bad dims
    stack = np.array([[CONSTANT, 0, 1, 3],
                      [CONSTANT, 0, 2, 1],
                      [MULTIPLICATION, 0, 1, 0]], dtype=int)
    assert validate(stack, [], (1, 1), [(1, 3), (2, 1)]) is False


@pytest.mark.parametrize("op_dims", [(2, 3), (3, 1), (0, 0)])
def test_validate_transpose_valid(op_dims):
    stack = np.array([[CONSTANT, 0, *op_dims],
                      [TRANSPOSE, 0, 0, 0]], dtype=int)
    assert validate(stack, [], (op_dims[1], op_dims[0]), [op_dims]) is True


ARITY_ONE_NO_SHAPE_CHANGE_OPERATORS = [SIN, COS, EXPONENTIAL, LOGARITHM,
                                       ABS, SQRT, SINH, COSH]


@pytest.mark.parametrize("cmd_node", ARITY_ONE_NO_SHAPE_CHANGE_OPERATORS)
@pytest.mark.parametrize("op_dims", [(2, 3), (3, 1), (0, 0)])
def test_validate_arity_one_no_change_valid(cmd_node, op_dims):
    stack = np.array([[CONSTANT, 0, *op_dims],
                      [cmd_node, 0, 0, 0]], dtype=int)
    assert validate(stack, [], op_dims, [op_dims]) is True


def test_validate_complex(complex_stack, complex_stack_load_shapes):
    x_shapes, constant_shapes = complex_stack_load_shapes
    assert validate(complex_stack, x_shapes, (3, 3), constant_shapes) is True


def test_validate_bad_output_dim(complex_stack, complex_stack_load_shapes):
    x_shapes, constant_shapes = complex_stack_load_shapes
    assert validate(complex_stack, x_shapes, (1, 3), constant_shapes) is False

# TODO test wrapper (validate_individual) with mocked validate
