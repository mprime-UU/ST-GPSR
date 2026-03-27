import time

import numpy as np
import torch
from torch_eval import evaluate as torch_cpp_eval

from bingo.symbolic_regression.agraphMD.evaluation_backend import evaluation_backend as bingo_eval
from bingo.symbolic_regression.agraphMD.pytorch_evaluation_backend import evaluation_backend as torch_eval

# my_cmd_arr = np.array([[0, 0, 3, 3],
#                        [2, 0, 0, 0],
#                        [4, 0, 1, 0]], dtype=int)

my_cmd_arr = np.array([[ 0,  2,  0,  0],  # 0
                       [ 6,  0,  0,  0],  # 1
                       [ 7,  1,  1,  1],  # 2
                       [ 6,  2,  2,  2],  # 3
                       [ 2,  3,  3,  3],  # 4
                       [ 6,  4,  4,  4],  # 5
                       [ 2,  5,  1,  5],  # 6
                       [ 6,  0,  0,  0],  # 7
                       [ 4,  6,  0,  6],  # 8
                       [ 1,  0,  0,  0],  # 9
                       [ 2,  1,  8,  1],  # 10
                       [ 1,  1,  3,  1],  # 11
                       [ 4, 10,  9, 10],  # 12
                       [ 7,  7,  7,  7],  # 13
                       [ 2, 13, 12, 13],  # 14
                       [ 4, 11, 14, 11]])  # 15
constants = [np.array(0), np.zeros((3, 1))]

def _get_torch_const(constants, data_len):
    with torch.no_grad():
        new_constants = []
        for constant in constants:
            # if constant.shape == ():
            #     constant = constant
            extended_constant = torch.from_numpy(constant).double()
            # extended_constant = extended_constant[None, :].expand(data_len, *([-1] * len(extended_constant.size())))
            new_constants.append(extended_constant)
        return new_constants

# torch_constants = torch.from_numpy(np.array(constants)).double()[None, :]
n_points = 533

torch_constants = _get_torch_const(constants, n_points)

numpy_x = [np.random.rand(n_points), np.random.rand(n_points), np.random.rand(n_points)]
# numpy_x = [numpy_x_0]

torch_x = torch.from_numpy(np.array(numpy_x))
print(torch_x.size())

print(bingo_eval.evaluate(my_cmd_arr, numpy_x, constants).shape)
print(torch_eval.evaluate(my_cmd_arr, torch_x, torch_constants).shape)
print(torch_cpp_eval(my_cmd_arr, torch_x, torch_constants).detach().numpy().shape)

# import time
#
# time_1 = time.perf_counter_ns()
# bingo_eval.evaluate(my_cmd_arr, numpy_x, constants)
# time_2 = time.perf_counter_ns()
# torch_eval.evaluate(my_cmd_arr, torch_x, torch_constants)
# time_3 = time.perf_counter_ns()
# torch_cpp_eval(my_cmd_arr, torch_x, torch_constants)
# time_4 = time.perf_counter_ns()
#
# print((time_2 - time_1) / 1e9)
# print((time_3 - time_2) / 1e9)
# print((time_4 - time_3) / 1e9)
