import numpy as np
import pandas as pd

if __name__ == "__main__":
    original_data_name = "vpsc_evo_17_data_3d_points_implicit_format"
    data = np.loadtxt(original_data_name + ".txt")

    # get point trajectories by splitting on nan rows
    point_trajectories = []
    nan_rows = np.where(np.all(np.isnan(data), axis=1))[0]

    point_trajectory = []
    for i, row in enumerate(data):
        if i in nan_rows:
            point_trajectories.append(point_trajectory)
            point_trajectory = []
        else:
            point_trajectory.append(row)

    point_trajectories.append(point_trajectory)
    point_trajectories = np.array(point_trajectories)

    # convert point trajectories from shape
    # (n_points_per_yield x n_yield x point_dim) -> (n_yield x n_points_per_yield x point_dim)
    point_trajectories = point_trajectories.transpose((1, 0, 2))


    # shift point trajectories by one per yield surface
    all_shifted_yield_points = []
    n_points_per_yield = point_trajectories.shape[1]
    for i, yield_surface_points in enumerate(point_trajectories):
        shifted_yield_points = []
        for j in range(n_points_per_yield):
            shifted_yield_points.append(yield_surface_points[(i * 18 + j) % n_points_per_yield])

        all_shifted_yield_points.append(shifted_yield_points)

    all_shifted_yield_points = np.array(all_shifted_yield_points)
    shifted_point_trajectories = all_shifted_yield_points.transpose((1, 0, 2))

    # convert shifted point trajectories into implicit format and save
    new_data = []

    for point_trajectory in shifted_point_trajectories:
        new_data.extend(point_trajectory.tolist())
        new_data.append(np.full(shifted_point_trajectories.shape[2], np.nan).tolist())

    # get rid of last nan row
    new_data = new_data[:-1]
    new_data = np.array(new_data)

    np.savetxt(original_data_name + "_shifted_idx.txt", new_data)
