import dill
import numpy as np
import matplotlib.pyplot as plt

from bingo.symbolic_regression.implicit_regression_md import ImplicitRegressionMD, ImplicitTrainingDataMD, \
    _calculate_partials


def pi_plane_to_flat_rot():
    pi_vector = np.array([1, 1, 1]) / np.sqrt(3.)
    wanted_vector = np.array([0, 0, 1])
    added = (pi_vector + wanted_vector).reshape([-1, 1])

    # from Rodrigues' rotation formula, more info here: https://math.stackexchange.com/a/2672702
    rot_mat = 2 * (added @ added.T) / (added.T @ added) - np.eye(3)
    return rot_mat


def flat_to_pi_plane_rot():
    return np.linalg.inv(pi_plane_to_flat_rot())


def get_dx_dt(x_with_nan):
    point_trajectories = []

    point_trajectory = []
    for row in x_with_nan:
        if not np.all(np.isnan(row)):
            point_trajectory.append(row)
        else:
            point_trajectories.append(point_trajectory)
            point_trajectory = []
    point_trajectories.append(point_trajectory)  # for last point which doesn't have a nan footer

    dx_dts = []

    for point_trajectory in point_trajectories:
        dx_dt = []
        prev_point = point_trajectory[0]
        # prev_point = point_trajectory[0]
        for point in point_trajectory:
            dx_dt.append(point - prev_point)
            prev_point = point
        dx_dt.append(point_trajectory[-1] - point_trajectory[-2])
        dx_dt = dx_dt[1:]
        dx_dts.append(dx_dt)


    dx_dts = np.array(dx_dts)
    dx_dts = dx_dts.reshape((-1, dx_dts.shape[2]))
    return dx_dts


if __name__ == "__main__":
    # data = np.loadtxt("vpsc_evo_17_data_3d_points_implicit_format.txt")
    data = np.loadtxt("vpsc_evo_17_data_3d_points_implicit_format_shifted_idx.txt")
    # data = np.loadtxt("vpsc_evo_16_data_3d_points_implicit_format.txt")

    x, dx_dt, _ = _calculate_partials(data, window_size=5)

    # x = data[np.invert(np.all(np.isnan(data), axis=1))]
    # dx_dt = get_dx_dt(data)
    #
    # print(x[0])
    # next_index = 1
    # print(x[0] + dx_dt[0])
    # print(x[next_index])
    #
    # print(x.shape)
    # print(dx_dt.shape)

    # remove first row of every trajectory where dxdt of eps == 0
    # good_rows = np.where(dx_dt[:, 3] != 0)[0]
    # x = x[good_rows]
    # dx_dt = dx_dt[good_rows]

    pos = x[:, :3]
    pos_dx_dt = dx_dt[:, :3]
    rot_mat = pi_plane_to_flat_rot()
    rot_mat /= np.linalg.norm(rot_mat)

    fig, axes = plt.subplots(1, 1)

    flat_pos = pos @ rot_mat
    flat_dx_dt = pos_dx_dt @ rot_mat

    flat_x, flat_y, flat_z = flat_pos.T
    dx_dt_x, dx_dt_y, dx_dt_z = flat_dx_dt.T

    eps = x[:, 3]
    deps_dt = dx_dt[:, 3]

    deps_dz = np.hstack((deps_dt[:, None], dx_dt_z[:, None]))

    axes.scatter(flat_x, flat_y, s=3)
    # axes.quiver(flat_x, flat_y, dx_dt_x, dx_dt_y, scale=1, scale_units="xy")
    axes.quiver(flat_x, flat_y, dx_dt_x, dx_dt_y, scale=5, scale_units="xy")

    # deps_dz[:, 0] *= 100

    # axes[1].scatter(eps, flat_pos[:, 2], s=3)
    # axes[1].quiver(eps, flat_pos[:, 2], deps_dt, flat_dx_dt[:, 2], scale=0.7, scale_units="xy")

    # print(deps_dz[0].T)
    # axes[1].quiver(eps[0], flat_z[0], 100, 250)
    # plt.scatter(dx_dt_x, dx_dt_y)
    plt.show()
