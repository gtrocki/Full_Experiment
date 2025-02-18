import isomap_fit
import leaf_segmentation
import numpy as np
import config
import os
import shutil

# Here we will get a folder for a full experiment, and we will create a function that iterates through every point
# cloud and for each isolates one leaf and plots the mean
def copy_fused_files(source_path, destination_path):

    # Iterate over all folders in the source path
    for session_folder in os.listdir(source_path):
        session_folder_path = os.path.join(source_path, session_folder)

        # Check if it's a directory
        if os.path.isdir(session_folder_path):
            # Create a destination folder with the same name in the destination path
            destination_folder_path = os.path.join(destination_path, session_folder)
            os.makedirs(destination_folder_path, exist_ok=True)

            # Specify the source file and destination file paths within the "dense" folder
            dense_folder_path = os.path.join(session_folder_path, 'dense')
            source_file_path = os.path.join(dense_folder_path, 'fused.ply')
            destination_file_path = os.path.join(destination_folder_path, 'fused.ply')

            # Copy the 'fused.ply' file from the source to the destination folder
            try:
                shutil.copy(source_file_path, destination_file_path)
                print(f"File 'fused.ply' copied from {session_folder} to {destination_folder_path}")
            except FileNotFoundError:
                print(f"File 'fused.ply' not found in {session_folder}")



def run_full_experiment(base_path: str, starting_frame: int, number_of_frames: int, color_range: np.ndarray = np.array([-0.5, 0.2, -0.7, 0]),
                        extra_color: int = None, extra_color_range: tuple = 0, result_option: int = 0,
                        color: str = 'red', blueprint_path: str = config.BLUEPRINT_PATH) -> int:
    """
    Runs a full experiment for a single leaf plotting the mean curvature.
    :param base_path: Path of the folder where the reconstructions are stored.
    :param number_of_frames: Number of reconstructions in the experiment
    :param color_range: Range of colors for the leaf of interest. The differences between G and R ,and B and G.
    :param result_option:
    0-
    1- saving flattened (2d) point cloud and scaled and unscaled 3d point cloud.
    11- Same as 1 but this time only check all of the initial conditions for GMMREG once every config.GMMREG_IC_PERIOD
        times. The times in between, we use the initial configuration that gave the best score for the previous round.
    2- saving flattened (2d) point cloud but without fitting and scaled and unscaled 3d point cloud.
    3- plot the already existing 2d point clouds overlayed with the point cloud of the blueprint.
    4- rescale the 3d point cloud so that it is the same size as the fitted 2d one. This is calculated by flattening
    the leaf and comparing its size to the fitted one.
    5- Same as in 4 for the first leaf, but rescale the remaining leaves by the same factor.
    :return:
    Return plots for the mean curvature (or some other such quantity of interest) of the leaf accross the
    range of the experiments reconstructions.
    """
    gmmreg_case_number = None
    for i in range(number_of_frames):
        j = i + starting_frame
        # =========> make sure the format of the path is the correct one <============
        # current_path = base_path + "fused" + str(10 * i + 180) + ".ply"
        # current_path = base_path + "Session_" + str(j) + "/fused.ply"
        current_path = os.path.join(base_path, "Session_" + str(j), "fused.ply")

        # Inside of the leaf segmentation we also rotate the point cloud so as to align it with global coordinates.
        # ACTUALLY NOT. NOW WE DO THAT SEPARATELY BEFORE STARTING.
        # Eventually we will find a more elegant method for doing this but for now it should suffice.
        pcd = leaf_segmentation.segment_single_leaf(current_path, color_range, extra_color=extra_color,
                                                    extra_color_range=extra_color_range)

        # visualize point cloud and extract points and normals for it
        pcd.estimate_normals()

        pcd = pcd.voxel_down_sample(voxel_size=0.02)
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)

        if result_option == 0:
            return 0
        if result_option == 1:
            fitter = isomap_fit.IsomapFit(blueprint_path,
                                           mapping=config.FIT_TYPE)

            # target_path_2d = base_path + "Session_" + str(j) + "/pcd_2d.csv"
            target_path_2d = os.path.join(base_path, "Session_" + str(j), "pcd_2d_" + color + ".csv")
            # target_path_3d = base_path + "Session_" + str(j) + "/pcd_3d.csv"
            target_path_3d = os.path.join(base_path, "Session_" + str(j), "pcd_3d_" + color + ".csv")
            target_path_3d_original_size = os.path.join(base_path, "Session_" + str(j),
                                                        "pcd_3d_" + color + "_original_size.csv")
            scores = fitter.flatten_scale_and_save(points, target_path_2d, target_path_3d, target_path_3d_original_size)
            print(f"scores = {scores}")
            print(f"Session {j}: done")
        if result_option == 11:
            fitter = isomap_fit.IsomapFit(blueprint_path,
                                           mapping=config.FIT_TYPE)
            # target_path_2d = base_path + "Session_" + str(j) + "/pcd_2d.csv"
            target_path_2d = os.path.join(base_path, "Session_" + str(j), "pcd_2d_" + color + ".csv")
            # target_path_3d = base_path + "Session_" + str(j) + "/pcd_3d.csv"
            target_path_3d = os.path.join(base_path, "Session_" + str(j), "pcd_3d_" + color + ".csv")
            target_path_3d_original_size = os.path.join(base_path, "Session_" + str(j),
                                                        "pcd_3d_" + color + "_original_size.csv")

            if (i / config.GMMREG_IC_PERIOD) == 0:
                scores = fitter.flatten_scale_and_save(points, target_path_2d, target_path_3d, target_path_3d_original_size)
                if config.SCORING_METHOD == 0:
                    best_score_index = np.argmin(scores)
                else:
                    best_score_index = np.argmax(scores)
                gmmreg_case_number = best_score_index
            else:
                scores = fitter.flatten_scale_and_save(points, target_path_2d, target_path_3d, target_path_3d_original_size,
                                                       gmmreg_case_number=gmmreg_case_number)
                print(f"scores = {scores}")
                print(f"Session {j}: done")
        if result_option == 2:
            fitter = isomap_fit.IsomapFit(blueprint_path,
                                           mapping=config.FIT_TYPE)

            # target_path_2d = base_path + "Session_" + str(j) + "/pcd_2d.csv"
            target_path_2d = os.path.join(base_path, "Session_" + str(j), "pcd_2d_" + color + ".csv")
            # target_path_3d = base_path + "Session_" + str(j) + "/pcd_3d.csv"
            target_path_3d = os.path.join(base_path, "Session_" + str(j), "pcd_3d_" + color + ".csv")
            target_path_3d_original_size = os.path.join(base_path, "Session_" + str(j),
                                                        "pcd_3d_" + color + "_original_size.csv")
            fitter.flatten_and_save(points, target_path_2d, target_path_3d, target_path_3d_original_size)
        if result_option == 3:
            fitter = isomap_fit.IsomapFit(blueprint_path,
                                           mapping=config.FIT_TYPE)

            print(f'Session: {j}')
            file_path_2d = os.path.join(base_path, "Session_" + str(j), "pcd_2d_" + color + ".csv")
            point_cloud_2d = np.loadtxt(file_path_2d, delimiter=',', skiprows=1)
            fitter.plot_of_fitting(point_cloud_2d)

        if result_option == 4:
            fitter = isomap_fit.IsomapFit(blueprint_path,
                                           mapping=config.FIT_TYPE)


            print(f'Session: {j}')
            file_path_2d = os.path.join(base_path, "Session_" + str(j), "pcd_2d_" + color + ".csv")
            point_cloud_2d = np.loadtxt(file_path_2d, delimiter=',', skiprows=1)
            file_path_3d = os.path.join(base_path, "Session_" + str(j), "pcd_3d_" + color + ".csv")
            point_cloud_3d = np.loadtxt(file_path_3d, delimiter=',', skiprows=1)

            fitter.scale_3d_point_cloud(point_cloud_3d, point_cloud_2d, file_path_3d)

        if result_option == 5:
            colors = ["red", "pink", "blue", "turquoise"]
            number_of_leaves = 4
            fitter = isomap_fit.IsomapFit(blueprint_path,
                                           mapping=config.FIT_TYPE)

            print(f'Session: {j}')
            file_path_2d_0 = os.path.join(base_path, "Session_" + str(j), "pcd_2d_" + colors[0] + ".csv")
            point_cloud_2d_0 = np.loadtxt(file_path_2d_0, delimiter=',', skiprows=1)
            file_path_3d_0 = os.path.join(base_path, "Session_" + str(j), "pcd_3d_" + colors[0] + ".csv")
            point_cloud_3d_0 = np.loadtxt(file_path_3d_0, delimiter=',', skiprows=1)

            scaling_factor = fitter.scale_3d_point_cloud(point_cloud_3d_0, point_cloud_2d_0, file_path_3d_0)

            for k in range(1, 4):
                file_path_3d = os.path.join(base_path, "Session_" + str(j), "pcd_3d_" + colors[k] + ".csv")
                points_3d_original = np.loadtxt(file_path_3d_0, delimiter=',', skiprows=1)

                # Scale the original 3D point cloud
                points_3d = points_3d_original * scaling_factor
                # Save the point cloud as a CSV file
                np.savetxt(file_path_3d, points_3d, delimiter=',', header='X,Y,Z', comments='')


    return 0

if __name__ == "__main__":


    if config.COLOR_CASE == 0:
        # Values for red leaf.
        # run_full_experiment(config.BASE_EXPERIMENT_PATH, 150, color_range=(-0.473 , -0.18, -0.1, -0.012), extra_color=1,
        #                     extra_color_range=(0, 0.5), result_option=1)
        trapezoid = np.array([
            [-0.46, -0.012],  # Point A
            [-0.187, -0.012],  # Point B
            [-0.198, -0.109],  # Point C
            [-0.46, -0.1]  # Point D
        ])
        run_full_experiment(config.BASE_EXPERIMENT_PATH, 122, 530, color_range=trapezoid, extra_color=1,
                            extra_color_range=(0, 0.68), result_option=1, color='red')
    elif config.COLOR_CASE == 1:
        # Values for pink leaf.
        # run_full_experiment(config.BASE_EXPERIMENT_PATH, 446, color_range=(-0.4, -0.144, 0.006, 0.167), extra_color=1,
        #                     extra_color_range=(0, 0.5), result_option=1)
        # Values for pink leaf 2
        trapezoid = np.array([
            [-0.350, 0.106],  # Point A
            [-0.241, 0.194],  # Point B
            [-0.054, 0.031],  # Point C
            [-0.113, -0.011]  # Point D
        ])
        run_full_experiment(config.BASE_EXPERIMENT_PATH, 203, 123, color_range=trapezoid, extra_color=1,
                            extra_color_range=(0, 0.68), result_option=1, color='pink')
    elif config.COLOR_CASE == 2:
        # Values for blue leaf.
        trapezoid = np.array([
            [-0.040, 0.02],  # Point A
            [-0.038, 0.215],  # Point B
            [0.170, 0.219],  # Point C
            [0.023, 0.038]  # Point D
        ])
        run_full_experiment(config.BASE_EXPERIMENT_PATH, 203, 123, color_range=trapezoid, extra_color=1,
                            extra_color_range=(0, 0.68), result_option=1, color='blue')
    elif config.COLOR_CASE == 3:
        # Values for green leaf.
        trapezoid = np.array([
            [-0.057, 0.02],  # Point A
            [-0.032, 0.174],  # Point B
            [0.164, 0.174],  # Point C
            [0.02, 0.038]  # Point D
        ])
        run_full_experiment(config.BASE_EXPERIMENT_PATH, 10, 530, color_range=trapezoid, extra_color=0,
                            extra_color_range=(0, 0.5), result_option=1, color='green')
    elif config.COLOR_CASE == 4:
        # Values for turquoise leaf.
        trapezoid = np.array([
            [0.02, 0.02],  # Point A
            [0.230, 0.170],  # Point B
            [0.280, -0.040],  # Point C
            [0.027, -0.061]  # Point D
        ])
        run_full_experiment(config.BASE_EXPERIMENT_PATH, 203, 123, color_range=trapezoid, extra_color=1,
                            extra_color_range=(0, 0.68), result_option=1, color='turquoise')
