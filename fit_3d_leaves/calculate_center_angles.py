import matplotlib.path
import numpy as np
import torch
import matplotlib as mpl  # for path transforms
from matplotlib.patches import PathPatch
import matplotlib.pyplot as plt
import config
import os
import fitter
import math
from svgpath2mpl import parse_path



def find_leaf_center_angle(point_cloud_2d, blueprint_point_cloud, model_path, path: matplotlib.patches.Path,
                           show_plots=0):
    """
    Function for calculating the angle from the stem to the center of the leaf
    for one leaf over the time of the experiment. The function assumes that the leaves have been previously fitted
    by models, which are saved in their respective folders.
    :param point_cloud_2d: The 2d point cloud of the leaf for which we want to measure the angle.
    :param blueprint_point_cloud: The 2d point cloud of the leaf blueprint.
    :param model_path: The path to the model fitting the leaf for which we want to measure the angle.
    :param path: The parsed path of the model leaf, from where we reconstruct the points.
    :return:
    Return a list containing the angle for each frame.
    """

    bbox = path.get_extents()  # Obtain the bounding box of the path, an imaginary rectangular box that completely
    # encloses a geometric shape or a set of points.
    # An affine transformation (norm_trans) is created to translate the path such that its top-left corner is at the
    # origin (0, 0) and scale it so that the larger dimension is normalized to 1.
    norm_trans = mpl.transforms.Affine2D().translate(-bbox.x0, -bbox.y0).scale(1 / max(bbox.width, bbox.height))
    # The transformation is applied to the path using transform_path, and the bounding box is recalculated. The width
    # and height of the bounding box are stored in the tuple wh.
    path = norm_trans.transform_path(path)
    bbox = path.get_extents()
    wh = (bbox.width, bbox.height)

    # CHECK!!!!
    (point_cloud_2d, scaling_factor,
     min_val_x, min_val_y, blueprint_point_cloud) = fitter.flip_and_normalize(point_cloud_2d, blueprint_point_cloud, width=wh[0])
    # Not sure this is needed but still.

    model = fitter.load_model(model_path)

    # Find the index of the point with the lowest y-value
    index_of_min_y = np.argmin(point_cloud_2d[:, 1])
    # Extract the lowest y-value
    min_y = point_cloud_2d[index_of_min_y, 1]

    mid_y = wh[1] / 2
    mid_x = wh[0] / 2
    min_and_mid_points = np.array([[mid_x, min_y], [mid_x, mid_y]])

    if show_plots is True:
        # Create a figure and axis
        fig, ax = plt.subplots()
        # for plotting
        patch = PathPatch(path, facecolor="none", lw=2)
        ax.add_patch(patch)
        # Plot the point cloud.
        ax.scatter(point_cloud_2d[:, 0], point_cloud_2d[:, 1], color="blue", label="Blue Point")
        ax.scatter(min_and_mid_points[:, 0], min_and_mid_points[:, 1], color="red", label="Red Point")
        ax.set_aspect('equal')
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        # Add a legend
        ax.legend("plot for test")
        # Show the plot
        plt.show()

    vals = model(torch.tensor(min_and_mid_points, dtype=torch.float32)).detach().numpy()

    z_difference = vals[1, 2] - vals[0, 2]
    plane_difference = math.sqrt((vals[1, 0] - vals[0, 0]) ** 2 + (vals[1, 1] - vals[0, 1]) ** 2)

    angle_radians = math.atan2(z_difference, plane_difference)

    return angle_radians

# -----------------------------------------------------------------------------

def find_all_leaf_center_angles(base_path: str, starting_frame: int, number_of_frames: int, path,
                                colors= ["red", "pink", "blue", "turquoise"]):

    angle_lists = []
    for i in range(len(colors)):
        new_color_list = []
        angle_lists.append(new_color_list)

    for j in range(starting_frame, starting_frame + number_of_frames):
        for k, color in enumerate(colors):
            angle_list = angle_lists[k]
            print(f"angle_list = {angle_list}")

            model_path = os.path.join(base_path, "Session_" + str(j), "fitting_model_" + color + ".pt")
            file_path_2d_current = os.path.join(base_path, "Session_" + str(j), "pcd_2d_" + color + ".csv")
            file_path_3d_current = os.path.join(base_path, "Session_" + str(j), "pcd_3d_" + color + ".csv")
            point_cloud_2d = np.loadtxt(file_path_2d_current, delimiter=',', skiprows=1)
            blueprint_path = os.path.join("../blueprints", color + "_leaf_simple.jpg")
            blueprint_point_cloud = fitter.image_to_point_array(blueprint_path)

            #call function here
            angle_radians = find_leaf_center_angle(point_cloud_2d, blueprint_point_cloud, model_path, path)

            angle_list.append(angle_radians)
            print(f"angle_radians = {angle_radians}")

    print(print(f"angle_lists = {angle_lists}"))

    return angle_lists

# -----------------------------------------------------------------------------



if __name__ == "__main__":

    base_path = config.BASE_PATH
    number_of_frames = 320
    initial_frame = 1

    leaf_svg_path = config.LEAF_SVG_PATH
    path = parse_path(leaf_svg_path)

    center_angles = find_all_leaf_center_angles(base_path, initial_frame, number_of_frames, path,
                                                colors=["red", "pink", "blue", "turquoise"])
    center_angles_red = center_angles[0]
    center_angles_pink = center_angles[1]
    center_angles_blue = center_angles[2]
    center_angles_turquoise = center_angles[3]
    x = np.arange(initial_frame, (number_of_frames + initial_frame))

    # Create a figure and axis
    fig, ax = plt.subplots()
    # Plot the point cloud.
    ax.plot(x, center_angles_pink, color="pink", label="Pink Point")
    ax.plot(x, center_angles_blue, color="blue", label="Blue Point")
    ax.plot(x, center_angles_red, color="red", label="Red Point")
    ax.plot(x, center_angles_turquoise, color="turquoise", label="Turquoise Point")
    # ax.set_aspect('equal')
    ax.set_xlabel("Frame")
    ax.set_ylabel("Rads")
    # Add a legend
    ax.legend("plot for test")
    # Show the plot
    plt.show()

    # Combine arrays into a single 2D array
    combined_array = np.column_stack(
        (center_angles_red, center_angles_pink, center_angles_blue, center_angles_turquoise))

    # Save the combined array to a CSV file
    np.savetxt(os.path.join(base_path, 'angle_list_full.csv'), combined_array, delimiter=',', fmt='%f')

    print("CSV file saved successfully.")
