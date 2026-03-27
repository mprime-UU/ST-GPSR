"""
Attributes
----------
STACK_PRINT_MAP : dict {int: str}
                  A map of node number to a format string for stack output
LATEX_PRINT_MAP : dict {int: str}
                  A map of node number to a format string for latex output
SYMPY_PRINT_MAP : dict {int: str}
                  A map of node number to a format string for sympy output
CONSOLE_PRINT_MAP : dict {int: str}
                  A map of node number to a format string for console output
"""
from bingo.symbolic_regression.agraphMD.operator_definitions \
    import INTEGER, VARIABLE, CONSTANT, ADDITION, SUBTRACTION, MULTIPLICATION, \
           DIVISION, SIN, COS, SINH, COSH, EXPONENTIAL, LOGARITHM, POWER, ABS, \
           SQRT, SAFE_POWER, TRANSPOSE, ARCTAN, ARCCOS, CROSS, NORMALIZE, \
           MATRIX_MULT, ELEMENTWISE_MULT, MATRIX_VEC_MULT
from bingo.symbolic_regression.agraphMD.validation_backend.validation_backend \
    import get_stack_command_dimensions, is_scalar_shape

STACK_PRINT_MAP = {ADDITION: "({}) + ({})",
                   SUBTRACTION: "({}) - ({})",
                   MULTIPLICATION: "({}) * ({})",
                   DIVISION: "({}) / ({}) ",
                   SIN: "sin ({})",
                   COS: "cos ({})",
                   SINH: "sinh ({})",
                   COSH: "cosh ({})",
                   EXPONENTIAL: "exp ({})",
                   LOGARITHM: "log ({})",
                   POWER: "({}) ^ ({})",
                   ABS: "abs ({})",
                   SQRT: "sqrt ({})",
                   SAFE_POWER: "(|{}|) ^ ({})",
                   TRANSPOSE: "({})T",
                   ARCTAN: "arctan ({})",
                   ARCCOS: "arccos ({})",
                   CROSS: "({}) x ({})",
                   NORMALIZE: "normalize ({})",
                   ELEMENTWISE_MULT: "({}) * ({})",
                   MATRIX_VEC_MULT: "({}) @ ({})"}
LATEX_PRINT_MAP = {ADDITION: "{} + {}",
                   SUBTRACTION: "{} - ({})",
                   MULTIPLICATION: "({})({})",
                   DIVISION: "\\frac{{ {} }}{{ {} }}",
                   SIN: "sin{{ {} }}",
                   COS: "cos{{ {} }}",
                   SINH: "sinh{{ {} }}",
                   COSH: "cosh{{ {} }}",
                   EXPONENTIAL: "exp{{ {} }}",
                   LOGARITHM: "log{{ {} }}",
                   POWER: "({})^{{ ({}) }}",
                   ABS: "|{}|",
                   SQRT: "\\sqrt{{ {} }}",
                   SAFE_POWER: "(|{}|)^{{ ({}) }}",
                   TRANSPOSE: "({})^T",
                   ARCTAN: "tan^{-1}{{ {} }}",
                   ARCCOS: "cos^{-1}{{ {} }}",
                   CROSS: "({}) \\times ({})",
                   NORMALIZE: "\\text{normalize}({})",
                   MATRIX_MULT: "({})({})",
                   ELEMENTWISE_MULT: "({})({})",
                   MATRIX_VEC_MULT: "({})({})"}
SYMPY_PRINT_MAP = {ADDITION: "{} + {}",
                   SUBTRACTION: "{} - ({})",
                   MULTIPLICATION: "({}) * ({})",
                   DIVISION: "({})/({})",
                   SIN: "sin({})",
                   COS: "cos({})",
                   SINH: "sinh({})",
                   COSH: "cosh({})",
                   EXPONENTIAL: "exp({})",
                   LOGARITHM: "log({})",
                   POWER: "({})**({})",
                   ABS: "abs({})",
                   SQRT: "sqrt({})",
                   SAFE_POWER: "abs({})**({})",
                   TRANSPOSE: "({})^T",
                   ARCTAN: "atan({})",
                   ARCCOS: "acos({})",
                   CROSS: "cross({}, {})",
                   NORMALIZE: "normalize({})",
                   MATRIX_MULT: "({}) @ ({})",
                   ELEMENTWISE_MULT: "({}) * ({})",
                   MATRIX_VEC_MULT: "({}) @ ({})"}
CONSOLE_PRINT_MAP = {ADDITION: "{} + {}",
                     SUBTRACTION: "{} - ({})",
                     MULTIPLICATION: "({}) * ({})",
                     DIVISION: "({})/({}) ",  # TODO extra space intentional?
                     SIN: "sin({})",
                     COS: "cos({})",
                     SINH: "sinh({})",
                     COSH: "cosh({})",
                     EXPONENTIAL: "exp({})",
                     LOGARITHM: "log({})",
                     POWER: "({})^({})",
                     ABS: "|{}|",
                     SQRT: "sqrt({})",
                     SAFE_POWER: "(|{}|)^({})",
                     TRANSPOSE: "({})^T",
                     ARCTAN: "arctan({})",
                     ARCCOS: "arccos({})",
                     CROSS: "({}) x ({})",
                     NORMALIZE: "normalize({})",
                     MATRIX_MULT: "({}) @ ({})",
                     ELEMENTWISE_MULT: "({}) * ({})",
                     MATRIX_VEC_MULT: "({}) @ ({})"}


# TODO reintroduce stack formatting
# TODO introduce compact formatting (i.e., matrix entries have equations
#   rather than the equation being a list of matrix operations)
def get_formatted_string(eq_format, command_array, constants, variable_shapes,
                         constant_shapes):
    """ Builds a formatted string from command array and constants

    Parameters
    ----------
    eq_format : str
        "stack", "latex", "sympy", or "console"
    command_array : Nx3 array of int
        stack representation of an equation
    constants : list(float)
        list of numerical constants in the equation

    Returns
    -------
    str
        equation formatted in the way specified
    """
    if eq_format == "latex":
        format_dict = LATEX_PRINT_MAP
    elif eq_format == "sympy":
        format_dict = SYMPY_PRINT_MAP
    else:  # "console"
        format_dict = CONSOLE_PRINT_MAP

    stack_shapes = get_stack_command_dimensions(command_array, variable_shapes,
                                                constant_shapes)
    str_list = []
    for stack_element in command_array:
        tmp_str = _get_formatted_element_string(stack_element,
                                                str_list,
                                                format_dict,
                                                constants,
                                                stack_shapes)
        str_list.append(tmp_str)
    return str_list[-1]


def _get_stack_string(command_array, constants):
    tmp_str = ""
    for i, stack_element in enumerate(command_array):
        tmp_str += _get_stack_element_string(i, stack_element, constants)

    return tmp_str


def _get_stack_element_string(command_index, stack_element, constants):
    node, param1, param2 = stack_element
    tmp_str = "(%d) <= " % command_index
    if node == VARIABLE:
        tmp_str += "X_%d" % param1
    elif node == CONSTANT:
        if param1 == -1 or param1 >= len(constants):
            tmp_str += "C"
        else:
            tmp_str += "C_{} = {}".format(param1, constants[param1])
    elif node == INTEGER:
        tmp_str += "{} (integer)".format(param1)
    else:
        tmp_str += STACK_PRINT_MAP[node].format(param1, param2)
    tmp_str += "\n"
    return tmp_str


def _get_formatted_element_string(stack_element, str_list,
                                  format_dict, constants,
                                  stack_shapes):
    node, param1, param2, param3 = stack_element
    if node == VARIABLE:
        tmp_str = "X_%d" % param1
    elif node == CONSTANT:
        if param1 == -1 or param1 >= len(constants):
            tmp_str = "?"
        elif format_dict == LATEX_PRINT_MAP and constants[param1].size>1:
            matrix = constants[param1]
            rows = [" & ".join(map(str, row)) for row in matrix]
            body = " \\\\ \n".join(rows)
            tmp_str = f"\\begin{{bmatrix}}\n{body}\n\\end{{bmatrix}}"
        else:
            tmp_str = repr(constants[param1])
    # TODO integers should be tensors, not just scalars
    elif node == INTEGER:
        tmp_str = str(int(param1))
    elif node == MULTIPLICATION:
        if is_scalar_shape(stack_shapes[param1]) or \
                is_scalar_shape(stack_shapes[param2]):
            tmp_str = format_dict[node].format(str_list[param1],
                                               str_list[param2])
        else:
            tmp_str = format_dict[MATRIX_MULT].format(str_list[param1],
                                                      str_list[param2])
    else:
        tmp_str = format_dict[node].format(str_list[param1], str_list[param2])
    return tmp_str
