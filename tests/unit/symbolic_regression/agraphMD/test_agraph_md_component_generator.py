# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
from copy import deepcopy
import pytest
import numpy as np

from bingo.symbolic_regression.agraphMD.operator_definitions import *
from bingo.symbolic_regression.agraphMD.component_generator_md import ComponentGeneratorMD


@pytest.fixture
def sample_component_generator():
    generator = ComponentGeneratorMD(input_x_dimensions=[(3, 1), (4, 1)],
                                     num_initial_load_statements=2,
                                     terminal_probability=0.4,
                                     constant_probability=0.5)
    generator.add_operator(ADDITION)
    generator.add_operator(SIN)
    return generator


@pytest.fixture
def empty_generator():
    generator = ComponentGeneratorMD(input_x_dimensions=[(3, 1), (4, 1)],
                                     num_initial_load_statements=2,
                                     terminal_probability=0.4,
                                     constant_probability=0.5)
    return generator


@pytest.mark.parametrize("param_name, param_value, expected_error", [
    ("num_initial_load_statements", 0, ValueError),
    ("num_initial_load_statements", "string", TypeError),
    ("terminal_probability", -0.1, ValueError),
    ("terminal_probability", "string", TypeError),
    ("terminal_probability", 2, ValueError),
    ("constant_probability", -0.1, ValueError),
    ("constant_probability", "string", TypeError),
    ("constant_probability", 2, ValueError)
])
def test_raises_error_invalid_init(param_name, param_value, expected_error):
    kwargs = {"input_x_dimensions": 1}
    kwargs[param_name] = param_value
    with pytest.raises(expected_error):
        _ = ComponentGeneratorMD(**kwargs)

def test_raises_error_random_operator_with_no_operators():
    no_operator_generator = ComponentGeneratorMD(input_x_dimensions=[(3, 1)],
                                                 terminal_probability=0.0)
    _ = no_operator_generator.random_command(0)
    with pytest.raises(IndexError):
        _ = no_operator_generator.random_command(1)
    with pytest.raises(IndexError):
        _ = no_operator_generator.random_operator()


def test_random_terminal():
    np.random.seed(0)
    generator = ComponentGeneratorMD(input_x_dimensions=[(3, 1)],
                                     constant_probability=0.25)
    terminals = [generator.random_terminal() for _ in range(100)]
    assert terminals.count(VARIABLE) == 72
    assert terminals.count(CONSTANT) == 28


def test_random_terminal_default_probability_about_25():
    np.random.seed(0)
    generator = ComponentGeneratorMD(input_x_dimensions=[(3, 1), (2, 1), (1, 1)])
    terminals = [generator.random_terminal() for _ in range(100)]
    assert terminals.count(VARIABLE) == 72
    assert terminals.count(CONSTANT) == 28


def test_random_terminal_parameter():
    generator = ComponentGeneratorMD(input_x_dimensions=[(3, 1), (2, 1), (1, 1)])
    for _ in range(20):
        assert generator.random_terminal_parameter(VARIABLE) in [0, 1, 2]
        assert generator.random_terminal_parameter(CONSTANT) == -1


def test_random_dims():
    generator = ComponentGeneratorMD(input_x_dimensions=[(3, 1), (2, 1), (1, 1)])
    for i in range(100):
        dims = generator.random_dims()
        if dims[0] == 0:
            assert dims[1] == 0
        else:
            assert 0 < dims[0] <= 3, 0 < dims[1] <= 3


def test_random_terminal_command():
    np.random.seed(1)
    generator = ComponentGeneratorMD(input_x_dimensions=[(3, 1)])

    terminal_command_1 = generator.random_terminal_command()
    expected_command_1 = np.array([CONSTANT, -1, 0, 0], dtype=int)
    np.testing.assert_array_equal(terminal_command_1, expected_command_1)

    terminal_command_2 = generator.random_terminal_command()
    expected_command_2 = np.array([VARIABLE, 0, 3, 1], dtype=int)
    np.testing.assert_array_equal(terminal_command_2, expected_command_2)


def test_random_operator():
    np.random.seed(0)
    generator = ComponentGeneratorMD(input_x_dimensions=[(3, 1), (2, 1), (1, 1)])
    generator.add_operator("addition")
    generator.add_operator("multiplication")
    operators = [generator.random_operator() for _ in range(100)]
    assert operators.count(ADDITION) == 51
    assert operators.count(MULTIPLICATION) == 49


def test_random_operator_parameter():
    generator = ComponentGeneratorMD(input_x_dimensions=[(3, 1), (2, 1), (1, 1)])
    for command_location in np.random.randint(1, 100, 50):
        command_param = generator.random_operator_parameter(command_location)
        assert command_param < command_location


def test_random_operator_command():
    np.random.seed(2)
    generator = ComponentGeneratorMD(input_x_dimensions=[(3, 1), (2, 1), (1, 1)])
    generator.add_operator("addition")
    generator.add_operator("multiplication")

    terminal_command_1 = generator.random_operator_command(2)
    expected_command_1 = np.array([ADDITION, 1, 0, 0], dtype=int)
    np.testing.assert_array_equal(terminal_command_1, expected_command_1)

    terminal_command_2 = generator.random_operator_command(2)
    expected_command_2 = np.array([MULTIPLICATION, 0, 1, 0], dtype=int)
    np.testing.assert_array_equal(terminal_command_2, expected_command_2)


@pytest.mark.parametrize(["expected_operator", "inputs"],
                         [(INTEGER, [INTEGER, "integer"]),
                          (VARIABLE, [VARIABLE, "load", "x"]),
                          (CONSTANT, [CONSTANT, "constant", "c"]),
                          (ADDITION, [ADDITION, "add", "addition", "+"]),
                          (SUBTRACTION, [SUBTRACTION, "subtract", "subtraction", "-"]),
                          (MULTIPLICATION, [MULTIPLICATION, "multiply", "multiplication", "*"]),
                          (DIVISION, [DIVISION, "divide", "division", "/"]),
                          (SIN, [SIN, "sine", "sin"]),
                          (COS, [COS, "cosine", "cos"]),
                          (EXPONENTIAL, [EXPONENTIAL, "exponential", "exp", "e"]),
                          (LOGARITHM, [LOGARITHM, "logarithm", "log"]),
                          # (POWER, [POWER, "power", "pow", "^"]),
                          (ABS, [ABS, "absolute value", "||", "|"]),
                          (SQRT, [SQRT, "square root", "sqrt"]),
                          # (SAFE_POWER, [SAFE_POWER, "safe power", "safe pow"]),
                          (SINH, [SINH, "sineh", "sinh"]),
                          (COSH, [COSH, "cosineh", "cosh"]),
                          (TRANSPOSE, [TRANSPOSE, "transpose", "T"])])
def test_add_operator(empty_generator, expected_operator, inputs):
    for str_or_num in inputs:
        generator = deepcopy(empty_generator)
        generator.add_operator(str_or_num)
        assert generator.get_number_of_operators() == 1
        assert generator.random_operator() == expected_operator


def test_raises_error_on_invalid_add_operator(sample_component_generator):
    generator = ComponentGeneratorMD(input_x_dimensions=[(3, 1), (2, 1), (1, 1)])
    np.random.seed(0)
    with pytest.raises(ValueError):
        generator.add_operator("invalid operator")


def test_add_operator_with_weight():
    generator = ComponentGeneratorMD(input_x_dimensions=[(3, 1), (2, 1), (1, 1)])
    generator.add_operator(ADDITION)
    generator.add_operator(SUBTRACTION, operator_weight=0.0)
    operators = [generator.random_operator() for _ in range(100)]
    assert SUBTRACTION not in operators


def test_random_command():
    generator = ComponentGeneratorMD(input_x_dimensions=[(3, 1), (2, 1)],
                                     terminal_probability=0.5)
    generator.add_operator(ADDITION)
    generator.add_operator(SUBTRACTION)
    np.random.seed(2)
    generated_commands = np.empty((6, 4))
    expected_commands = np.array([[VARIABLE, 1, 2, 1],
                                  [SUBTRACTION, 0, 0, 0],
                                  [CONSTANT, -1, 1, 2],
                                  [VARIABLE, 0, 3, 1],
                                  [SUBTRACTION, 3, 2, 0],
                                  [ADDITION, 2, 1, 0]], dtype=int)
    for stack_location in range(generated_commands.shape[0]):
        generated_commands[stack_location, :] = \
            generator.random_command(stack_location)
    np.testing.assert_array_equal(generated_commands, expected_commands)


def test_numbers_of_terminals_and_params():
    generator = ComponentGeneratorMD(input_x_dimensions=[(3, 1), (2, 1), (1, 1)])
    assert generator.get_number_of_terminals() == 2
    assert generator.get_number_of_operators() == 0
    generator.add_operator(ADDITION)
    assert generator.get_number_of_terminals() == 2
    assert generator.get_number_of_operators() == 1
    generator.add_operator(SUBTRACTION)
    assert generator.get_number_of_terminals() == 2
    assert generator.get_number_of_operators() == 2
