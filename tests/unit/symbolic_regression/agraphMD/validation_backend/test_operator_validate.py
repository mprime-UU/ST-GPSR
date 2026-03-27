import pytest

from bingo.symbolic_regression.agraphMD.operator_definitions import *
from bingo.symbolic_regression.agraphMD.validation_backend.operator_validate import validate_operator


@pytest.mark.parametrize("integer_cmd", [[INTEGER, -1, 0, 0],  # -1 as a scalar
                                         [INTEGER, 1, 3, 2]])  # 3x2 array of 1s
def test_integer_validate_valid(integer_cmd):
    assert validate_operator(*integer_cmd, [], [], []) == (True, (integer_cmd[2], integer_cmd[3]))


@pytest.mark.parametrize("invalid_cmd", [[INTEGER, -1, -1, 0],  # bad dimension 1
                                         [INTEGER, 0, 0, -1],  # bad dimension 2
                                         [INTEGER, 3, -1, -1]])  # bad dimensions 1 and 2
def test_integer_validate_invalid(invalid_cmd):
    assert validate_operator(*invalid_cmd, [], [], [])[0] is False


@pytest.mark.parametrize("load_node", [VARIABLE, CONSTANT])
@pytest.mark.parametrize("load_params", [[1, 0, 0],  # scalar
                                         [2, 5, 1]])  # multi-dimensional
def test_load_validate_valid(load_node, load_params):
    # load_cmd = [load_node, load_params[0], load_params[1], load_params[2]]
    load_dims = [(0, 0), (0, 0), (5, 1)]
    assert validate_operator(load_node, *load_params, [], load_dims, load_dims) == (True, (load_params[1], load_params[2]))


@pytest.mark.parametrize("load_node, expected", [(VARIABLE, False), (CONSTANT, True)])
def test_load_validate_negative_one_param1(load_node, expected):
    print(validate_operator(load_node, -1, 0, 0, [], [(0, 0)], [(0, 0)]))
    assert validate_operator(load_node, -1, 0, 0, [], [(0, 0)], [(0, 0)]) == (expected, (0, 0))


@pytest.mark.parametrize("load_node", [VARIABLE, CONSTANT])
@pytest.mark.parametrize("load_params", [[-2, 0, 0],  # scalar w/ bad param1
                                         [1, -1, 0],  # bad dimension 1
                                         [0, 0, -1],  # bad dimension 2
                                         [1, -1, -1],  # bad dimensions 1 and 2
                                         [1, 1, 1],  # dims different from loaded
                                         [2, 0, 0]])  # load out of range
def test_load_validate_invalid(load_node, load_params):
    # load_cmd = [load_node, load_params[0], load_params[1], load_params[2]]
    load_dims = [(0, 0), (3, 3)]
    assert validate_operator(load_node, *load_params, [], load_dims, load_dims)[0] is False


@pytest.mark.parametrize("cmd_node", [ADDITION, SUBTRACTION])
@pytest.mark.parametrize("cmd_params", [[0, 1, 1],  # scalar + scalar
                                        [2, 3, -1]])  # multi-dim + multi-dim (same dims)
def test_addition_like_validate_valid(cmd_node, cmd_params):
    dimensions = [(0, 0), (0, 0), (1, 2), (1, 2)]
    assert validate_operator(cmd_node, *cmd_params, dimensions, [], []) == (True, dimensions[cmd_params[0]])


@pytest.mark.parametrize("cmd_node", [ADDITION, SUBTRACTION])
@pytest.mark.parametrize("cmd_params", [[0, 2, 1],  # scalar + multi-dim
                                        [2, 0, 0],  # multi-dim + scalar
                                        [2, 3, -1]])  # multi-dim + multi-dim (different dims)
def test_addition_like_validate_invalid(cmd_node, cmd_params):
    dimensions = [(0, 0), (0, 0), (1, 2), (2, 2)]
    assert validate_operator(cmd_node, *cmd_params, dimensions, [], [])[0] is False


@pytest.mark.parametrize("mult_cmd", [[MULTIPLICATION, 0, 1, 0],
                                      [MULTIPLICATION, 1, 0, 0]])
def test_multiplication_validate_valid_scalar_scalar(mult_cmd):
    dimensions = [(0, 0), (0, 0)]
    assert validate_operator(*mult_cmd, dimensions, [], []) == (True, (0, 0))


@pytest.mark.parametrize("mult_cmd", [[MULTIPLICATION, 0, 1, 0],  # scalar * multi-dim
                                      [MULTIPLICATION, 1, 0, 0]])  # multi-dim * multi-dim
def test_multiplication_validate_valid_scalar_multi_dim(mult_cmd):
    dimensions = [(0, 0), (1, 2)]
    assert validate_operator(*mult_cmd, dimensions, [], []) == (True, (1, 2))


def test_multiplication_validate_valid_multi_dim_multi_dim():
    dimensions = [(1, 3), (3, 2)]
    assert validate_operator(MULTIPLICATION, 0, 1, -1, dimensions, [], []) == (True, (1, 2))


def test_multiplication_validate_invalid():
    dimensions = [(1, 2), (3, 1)]
    assert validate_operator(MULTIPLICATION, 0, 1, 1, dimensions, [], [])[0] is False


@pytest.mark.parametrize("transpose_cmd, expected_dim",
                         [[[TRANSPOSE, 0, 1, -2], (2, 1)],
                          [[TRANSPOSE, 1, -1, 5], (0, 0)]])
def test_transpose_validate_valid(transpose_cmd, expected_dim):
    dimensions = [(1, 2), (0, 0)]  # TODO can we do transpose on a scalar?
    assert validate_operator(*transpose_cmd, dimensions, [], []) == (True, expected_dim)


def test_transpose_validate_invalid():
    assert validate_operator(TRANSPOSE, -1, 0, 0, [(0, 0)], [], [])[0] is False


NO_SHAPE_CHANGE_AR_ONE_OPS = [SIN, COS, EXPONENTIAL, LOGARITHM,
                              ABS, SQRT, SINH, COSH]


@pytest.mark.parametrize("cmd_node", NO_SHAPE_CHANGE_AR_ONE_OPS)
def test_no_change_operator_validate_invalid(cmd_node):
    assert validate_operator(cmd_node, -1, 0, 0, [(3, 1)], [], [])[0] is False


@pytest.mark.parametrize("cmd_node", NO_SHAPE_CHANGE_AR_ONE_OPS)
@pytest.mark.parametrize("cmd_params", [[0, 0, 0],  # scalar
                                        [1, 0, 0],  # multi-dimensional (vector)
                                        [2, 0, 0]])  # multi-dimensional (matrix)
def test_no_change_operator_validate_valid(cmd_node, cmd_params):
    dimensions = [(0, 0), (4, 1), (3, 2)]
    assert validate_operator(cmd_node, *cmd_params, dimensions, [], []) == (True, dimensions[cmd_params[0]])