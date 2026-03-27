import numpy as np
import pandas as pd


if __name__ == '__main__':
    df = pd.read_pickle("vpsc_evo_17_data_3d_points_transpose_implicit_format_mapping_df.pkl")

    eps = float(df.iloc[0]["eps"])
    P_actual = np.array(df.iloc[0]["P_actual"])
    P_desired = np.array(df.iloc[0]["P_vm"])

    print(P_desired)

    # make P_actual symmetric
    P_actual = (P_actual + P_actual.T) / 2
    print(repr(P_actual))


    def get_loss(mapping_mat_flattened, P_in, P_out, minimize=False):
        mapping_mat = mapping_mat_flattened.reshape((3, 3))

        P_in = P_desired
        P_out = P_actual

        P_pred = mapping_mat.T @ P_in @ mapping_mat
        # P_pred = mapping_mat @ P_in
        loss = (P_out - P_pred).flatten()
        loss /= P_out.flatten()
        if minimize:
            loss = np.mean(np.square(loss))
        return loss

    from scipy.optimize import root, minimize

    P_input = P_desired
    P_output = P_actual

    # P_input = P_actual
    # P_output = P_desired

    optim_result = root(get_loss, x0=np.ones(9), args=(P_input, P_output), method="lm")
    # optim_result = root(get_loss, x0=np.ones(9), args=(P_input, P_output))
    # optim_result = minimize(get_loss, x0=np.ones(9), args=(P_input, P_output, True), method="Nelder-Mead")
    # optim_result = minimize(get_loss, x0=np.ones(9), args=(P_input, P_output, True), method="BFGS")

    found_mapping_mat = optim_result.x.reshape((3, 3))
    print(optim_result)
    print("\nfound result:", repr(found_mapping_mat))
    print("loss of found matrix:", get_loss(found_mapping_mat, P_actual, P_desired))
    print("\tmse:", get_loss(found_mapping_mat, P_actual, P_desired, True))

    print("\n output mat:", P_output)
    # print("\n mapped input mat:", found_mapping_mat @ P_input)
    print("\n mapped input mat:", found_mapping_mat.T @ P_input @ found_mapping_mat)
