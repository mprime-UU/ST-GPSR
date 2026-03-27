import numpy as np
import pandas as pd
import dill


def P_indv(eps):
    X_0 = eps
    from numpy import array, sin, cos

    P = array([[-8.82173967e-03, -3.08191511e+00, -4.56283502e+00],
                [ 3.09082937e+00, -9.44079206e-03, -1.50211175e+02],
                [ 4.57255208e+00,  1.50220639e+02, -9.83244993e-03]])
    return P


if __name__ == "__main__":
    original_name = "vpsc_evo_17_data_3d_points_transpose_implicit_format"
    data = np.loadtxt(original_name + ".txt")
    data = data[np.invert(np.all(np.isnan(data), axis=1))]

    eps_index = data.shape[1] - 1

    eps = data[:, eps_index][:, None, None]

    P_actual = P_indv(eps)
    if P_actual.ndim == 2:
        P_actual = np.repeat(P_actual[None, :, :], repeats=eps.shape[0], axis=0)
    P_vm = np.array([[1, -0.5, -0.5],
                     [-0.5, 1, -0.5],
                     [-0.5, -0.5, 1]]) / 100

    P_vm = np.repeat(P_vm[None, :, :], repeats=len(P_actual), axis=0)

    df = pd.DataFrame(data={"eps": eps.squeeze().tolist(), "P_actual": P_actual.tolist(), "P_vm": P_vm.tolist()})

    df.to_pickle(original_name + "_mapping_df.pkl")
    print("done. saved to", original_name + "_mapping_df.pkl")
