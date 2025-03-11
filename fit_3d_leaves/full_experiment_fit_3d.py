"""
This file is intended to run the functions of fitter.py for all the frames in an experiment for a single color.
The idea is to separate the classes and methods from the running of the experiment so both files can be kept neatly.
"""

import fitter
from svgpath2mpl import parse_path
import config
import os

def run_full_experiment(base_path: str, starting_frame: int, number_of_frames: int, result_option: int = 0,
                        color: str = 'red', leaf_svg_path = config.LEAF_SVG_PATH, scaling: int = 1) -> int:
    """
    Runs a full experiment for a single leaf where it fits the 3d point cloud by a function. Some of the options are for
    plotting rather than fitting.
    :param base_path: Path of the folder where the reconstructions are stored and the fitting and plots will be too.
    :param starting_frame: The starting frame of the sequence we are evaluating.
    :param number_of_frames: Number of reconstructions in the experiment
    :param result_option:
    0- Check that the normalization of the path and the 2d point cloud is done properly by plotting the two together
    and making sure they fit.
    1- Find the mapping that takes points from the 2d point cloud to the 3d one amd save it.
    2- Plot the target point cloud together with the mapping we found evaluated at a number of random points within the
    contour of the leaf (they should form a pretty decent interpolation of the 3d point cloud).
    3- Plot the 2d point cloud together with the target 3d point cloud in the same space.
    4- Use the saved model to calculate the mean curvature on the leaf, which is turned into a 2d plot and saved also.
    :param color: The color of the leaf for which to do the work.
    :param leaf_svg_path: The scalable vector graphics version of the path describing the contour of the leaf.
    :param scaling: Choose which 3d point cloud to fit.
    0- The scaled point cloud so it has the same size as the 2d point cloud fitted to the blueprint.
    1- The originally sized point cloud so it preserves the size relation of the original scene between all leaves.
    :return:
    Return plots for the mean curvature (or some other such quantity of interest) of the leaf accross the
    range of the experiments reconstructions.
    """

    path = parse_path(leaf_svg_path)
    scaling_factor = None
    min_val_x = None
    min_val_y = None
    for i in range(number_of_frames):
        j = i + starting_frame

        file_path_2d = os.path.join(base_path, "Session_" + str(j), "pcd_2d_" + color + ".csv")
        if scaling == 0:
            file_path_3d = os.path.join(base_path, "Session_" + str(j), "pcd_3d_" + color + ".csv")
        elif scaling == 1:
            file_path_3d = os.path.join(base_path, "Session_" + str(j),
                                                        "pcd_3d_" + color + "_original_size.csv")
        else:
            file_path_3d = os.path.join(base_path, "Session_" + str(j),
                                                        "pcd_3d_" + color + "_original_size.csv")
        blueprint_path = os.path.join("../blueprints", color + "_leaf_simple.jpg")
        model_save_path = os.path.join(base_path, "Session_" + str(j), "fitting_model_" + color + ".pt")

        if result_option == 0:
            print(f'session number {j}')
            plotter = fitter.Plotter(path)
            plotter.plot_2d_point_cloud_with_blueprint(file_path_2d, blueprint_path)
            plotter.plot_transformed_2d_pcd_and_path(file_path_2d, blueprint_path)

        if result_option == 1:
            print(f'session number {j}')
            scaling_factor, min_val_x, min_val_y = fitter.fit_and_save_model(file_path_2d, file_path_3d, path,
                                                                                 blueprint_path,
                                                                                 save_model=model_save_path,
                                                                                 scaling_factor=scaling_factor,
                                                                                 min_val_x=min_val_x, min_val_y=min_val_y)
            print(f"Session {j}: done")

        if result_option == 2:
            model = fitter.load_model(model_save_path)
            print(f"frame number = {j}")
            plotter = fitter.Plotter(path)
            scaling_factor, min_val_x, min_val_y = plotter.plot_model_plus_target(model, file_path_2d, file_path_3d,
                                                                                  blueprint_path,
                                                                                  scaling_factor=scaling_factor,
                                                                                  min_val_x=min_val_x,
                                                                                  min_val_y=min_val_y)

        if result_option == 3:
            print(f'session {j}')
            plotter = fitter.Plotter(path)
            plotter.plot_2d_and_3d_point_clouds(file_path_2d, file_path_3d, j)

        if result_option == 4:
            print(f'session number {j}')
            plot_save_path = os.path.join(base_path, "Session_" + str(j), "mean_curvature_" + color + ".png")
            scaling_factor, min_val_x, min_val_y = fitter.plot_curvature(file_path_2d, file_path_3d, path,
                                                                         model_path=model_save_path,
                                                                         save_plot=plot_save_path)
    return 0

if __name__ == "__main__":

    base_path = config.BASE_PATH
    number_of_frames = config.NUMBER_OF_FRAMES
    offset = config.INITIAL_FRAME
    leaf_svg_path = config.LEAF_SVG_PATH

    if config.COLOR_CASE == 0:
        # Values for red leaf.
        run_full_experiment(config.BASE_PATH, offset, number_of_frames, result_option=2, color='red',
                            leaf_svg_path=leaf_svg_path, scaling=1)
    elif config.COLOR_CASE == 1:
        # Values for pink leaf.
        run_full_experiment(config.BASE_PATH, offset, number_of_frames,  result_option=1, color='pink',
                            leaf_svg_path=leaf_svg_path, scaling=1)
    elif config.COLOR_CASE == 2:
        # Values for blue leaf.
        run_full_experiment(config.BASE_PATH, offset, number_of_frames, result_option=1, color='blue',
                            leaf_svg_path=leaf_svg_path, scaling=1)
    elif config.COLOR_CASE == 3:
        # Values for green leaf.
        run_full_experiment(config.BASE_PATH, offset, number_of_frames, result_option=1, color='green',
                            leaf_svg_path=leaf_svg_path, scaling=1)
    elif config.COLOR_CASE == 4:
        # Values for turquoise leaf.
        run_full_experiment(config.BASE_PATH, offset, number_of_frames, result_option=1, color='turquoise',
                            leaf_svg_path=leaf_svg_path, scaling=1)