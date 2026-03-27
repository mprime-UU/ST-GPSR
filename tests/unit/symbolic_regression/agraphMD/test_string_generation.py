# Ignoring some linting rules in tests
# pylint: disable=redefined-outer-name
# pylint: disable=missing-docstring
import pytest
import numpy as np

from bingo.symbolic_regression.agraphMD.operator_definitions import *
from bingo.symbolic_regression.agraphMD.string_generation_md import \
    get_formatted_string


# TODO add new matrix operators to this (e.g., normalize, transpose, etc.)
@pytest.fixture
def all_funcs_cmd_arr_info():
    cmd_arr = np.array([[INTEGER, 5, 0, 0],
                        [VARIABLE, 0, 0, 0],
                        [CONSTANT, 0, 0, 0],
                        [ADDITION, 1, 0, 0],
                        [SUBTRACTION, 2, 3, 0],
                        [MULTIPLICATION, 4, 1, 0],
                        [DIVISION, 5, 1, 0],
                        [SIN, 6, 0, 0],
                        [COS, 7, 0, 0],
                        [EXPONENTIAL, 8, 0, 0],
                        [LOGARITHM, 9, 0, 0],
                        [POWER, 10, 0, 0],
                        [ABS, 11, 0, 0],
                        [SQRT, 12, 0, 0]])
    variable_shapes = [(0, 0)]
    constant_shapes = [(0, 0)]
    return cmd_arr, variable_shapes, constant_shapes


def test_latex_format(all_funcs_cmd_arr_info):
    expected_string = "\\sqrt{ |(log{ exp{ cos{ sin{ \\frac{ (2.0 - (X_0 + " + \
                      "5))(X_0) }{ X_0 } } } } })^{ (5) }| }"
    cmd_arr, variable_shapes, constant_shapes = all_funcs_cmd_arr_info
    generated_string = get_formatted_string("latex", cmd_arr, [2.0, ],
                                            variable_shapes, constant_shapes)
    assert generated_string == expected_string


def test_latex_format_no_consts(all_funcs_cmd_arr_info):
    expected_string = "\\sqrt{ |(log{ exp{ cos{ sin{ \\frac{ (? - (X_0 + " + \
                      "5))(X_0) }{ X_0 } } } } })^{ (5) }| }"
    cmd_arr, variable_shapes, constant_shapes = all_funcs_cmd_arr_info
    generated_string = get_formatted_string("latex", cmd_arr, [],
                                            variable_shapes, constant_shapes)
    assert generated_string == expected_string


def test_sympy_format(all_funcs_cmd_arr_info):
    expected_string = "sqrt(abs((log(exp(cos(sin(((2.0 - (X_0 + 5)) * (X_0))/" \
                      "(X_0))))))**(5)))"
    cmd_arr, variable_shapes, constant_shapes = all_funcs_cmd_arr_info
    generated_string = get_formatted_string("sympy", cmd_arr, [2.0, ],
                                            variable_shapes, constant_shapes)
    assert generated_string == expected_string


def test_sympy_format_no_consts(all_funcs_cmd_arr_info):
    expected_string = "sqrt(abs((log(exp(cos(sin(((? - (X_0 + 5)) * (X_0))" \
                      "/(X_0))))))**(5)))"
    cmd_arr, variable_shapes, constant_shapes = all_funcs_cmd_arr_info
    generated_string = get_formatted_string("sympy", cmd_arr, [],
                                            variable_shapes, constant_shapes)
    assert generated_string == expected_string


def test_console_format(all_funcs_cmd_arr_info):
    expected_string = "sqrt(|(log(exp(cos(sin(((2.0 - (X_0 + 5)) * (X_0))/" + \
                      "(X_0) )))))^(5)|)"
    cmd_arr, variable_shapes, constant_shapes = all_funcs_cmd_arr_info
    generated_string = get_formatted_string("console", cmd_arr, [2.0, ],
                                            variable_shapes, constant_shapes)
    assert generated_string == expected_string


def test_console_format_no_consts(all_funcs_cmd_arr_info):
    expected_string = "sqrt(|(log(exp(cos(sin(((? - (X_0 + 5)) * (X_0))/" + \
                      "(X_0) )))))^(5)|)"
    cmd_arr, variable_shapes, constant_shapes = all_funcs_cmd_arr_info
    generated_string = get_formatted_string("console", cmd_arr, [],
                                            variable_shapes, constant_shapes)
    assert generated_string == expected_string


def test_stack_format(all_funcs_cmd_arr_info):
    expected_string = "(0) <= 5 (integer)\n" + \
                      "(1) <= X_0\n" + \
                      "(2) <= C_0 = 2.0\n" + \
                      "(3) <= (1) + (0)\n" + \
                      "(4) <= (2) - (3)\n" + \
                      "(5) <= (4) * (1)\n" + \
                      "(6) <= (5) / (1)\n" + \
                      "(7) <= sin (6)\n" + \
                      "(8) <= cos (7)\n" + \
                      "(9) <= exp (8)\n" + \
                      "(10) <= log (9)\n" + \
                      "(11) <= (10) ^ (0)\n" + \
                      "(12) <= abs (11)\n" + \
                      "(13) <= sqrt (12)\n"
    cmd_arr, variable_shapes, constant_shapes = all_funcs_cmd_arr_info
    generated_string = get_formatted_string("stack", cmd_arr, [2.0, ],
                                            variable_shapes, constant_shapes)
    assert generated_string == expected_string


def test_stack_format_no_consts(all_funcs_cmd_arr_info):
    expected_string = "(0) <= 5 (integer)\n" + \
                      "(1) <= X_0\n" + \
                      "(2) <= C\n" + \
                      "(3) <= (1) + (0)\n" + \
                      "(4) <= (2) - (3)\n" + \
                      "(5) <= (4) * (1)\n" + \
                      "(6) <= (5) / (1)\n" + \
                      "(7) <= sin (6)\n" + \
                      "(8) <= cos (7)\n" + \
                      "(9) <= exp (8)\n" + \
                      "(10) <= log (9)\n" + \
                      "(11) <= (10) ^ (0)\n" + \
                      "(12) <= abs (11)\n" + \
                      "(13) <= sqrt (12)\n"
    cmd_arr, variable_shapes, constant_shapes = all_funcs_cmd_arr_info
    generated_string = get_formatted_string("stack", cmd_arr, [],
                                            variable_shapes, constant_shapes)
    assert generated_string == expected_string


@pytest.fixture
def multiplication_types_cmd_arr_info():
    cmd_arr = np.array([[VARIABLE, 0, 3, 3],          # X_0; 0
                        [CONSTANT, 0, 0, 0],          # C_0; 1
                        [CONSTANT, 1, 0, 0],          # C_1; 2
                        [VARIABLE, 1, 3, 3],          # X_1; 3
                        [MULTIPLICATION, 1, 2, 0],    # C_0 * C_1; 4
                        [MULTIPLICATION, 0, 4, 0],    # X_0 * (C_0 * C_1); 5
                        [MULTIPLICATION, 5, 3, 0],    # (X_0 * (C_0 * C_1)) @ X_1; 6  # pylint: disable=line-too-long
                        ])
    variable_shapes = [(3, 3), (3, 3)]
    constant_shapes = [(0, 0), (0, 0)]
    return cmd_arr, variable_shapes, constant_shapes


@pytest.mark.parametrize("format_type, expected_string",
                         [["latex", "((X_0)(({})({})))(X_1)"],
                          ["sympy", "((X_0) * (({}) * ({}))) @ (X_1)"],
                          ["console", "((X_0) * (({}) * ({}))) @ (X_1)"]])
def test_different_multiplication_types(multiplication_types_cmd_arr_info,
                                        format_type, expected_string):
    cmd_arr, variable_shapes, constant_shapes = \
        multiplication_types_cmd_arr_info

    # get dummy constants and format expected string to include them
    constants = []
    for i, constant_shape in enumerate(constant_shapes):
        if constant_shape == (0, 0):
            constant_shape = ()
        constants.append(np.ones(constant_shape) * i)
    constant_reprs = [repr(constant) for constant in constants]
    expected_string = expected_string.format(*constant_reprs)

    generated_string = get_formatted_string(format_type, cmd_arr,
                                            constants, variable_shapes,
                                            constant_shapes)

    assert generated_string == expected_string

# TODO add test for integer shaping
