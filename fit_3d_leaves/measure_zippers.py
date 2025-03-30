# import matplotlib.path
import numpy as np
import pyvista as pv
from svgpath2mpl import parse_path
import torch
import config
import os
import fitter
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev
from scipy.integrate import quad
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from itertools import combinations

def get_point_clouds(path, models, num_points=3000, get_2d_point_cloud=None):
    """
    Receive a path and models ang obtain the point clouds resulting from evaluating the model
    using points inside the path.
    :param path: A parsed svg path describing the contour of a leaf.
    :param models: A list containing the models for each of the leaves.
    :param num_points: The number of points to use in each of the leaves.
    :return:
    Return the 3d point clouds for the evaluated models and the basis 2d point cloud used to get them.
    """
    points = fitter.sample_low_disc_seq(num_points, path)  # Sample the points from a low discrepancy series.

    vals = []
    for model in models:
        # Put the points in the model (neural network) to get the output.
        vals.append(model(torch.tensor(points, dtype=torch.float32)).detach().numpy())

    if get_2d_point_cloud==None:
        return vals
    else:
        return vals, points


# Function for plotting the result of the transformation of the model being applied to num_points points sampled
# from a low discrepancy sequence (a model surface).
def plot_all_leaves(point_clouds, save_orbit_animation=False, colors=["red", "pink", "blue", "turquoise"] ):
    """
    Function for plotting a full frame of the experiment with the fitted models of all the leaves.
    :param point_clouds: The point clouds of the leaves to be plotted.
    :param save_orbit_animation: A parameter to decide whether to save a video from the display.
    :param colors: A list of the colors of each of the leaves.
    :return:
    """

    # Set up a PyVista plotter and configure the settings.
    pv.set_plot_theme('dark')
    plotter = pv.Plotter()
    plotter.enable_terrain_style(mouse_wheel_zooms=True)
    plotter.enable_anti_aliasing()

    for j in range(len(colors)):
        mesh = pv.PolyData(point_clouds[j])
        # mesh['color'] = colors[j]
        plotter.add_mesh(mesh, color=colors[j], point_size=20,
                         render_points_as_spheres=True,
                         rgb=False)

    # enable eye_dome_lighting
    plotter.enable_eye_dome_lighting()

    # enable axes, with a large font size
    plotter.show_grid()
    plotter.show_bounds(all_edges=True, font_size=16, color='white', location='outer')

    if not save_orbit_animation:
        plotter.show(auto_close=False)
    else:
        path = plotter.generate_orbital_path(n_points=36, shift=mesh.length)
        # plotter.open_gif("orbit.gif")
        plotter.open_movie("orbit.mp4")
        plotter.orbit_on_path(path, write_frames=True)
        plotter.close()


def find_gluing_lines(point_cloud1, point_cloud2, distance_threshold):
    """
    Identifies the points that are "glued" between two 3D point clouds based on a distance threshold.

    Args:
        point_cloud1 (numpy.ndarray): First 3D point cloud, shape (n1, 3).
        point_cloud2 (numpy.ndarray): Second 3D point cloud, shape (n2, 3).
        distance_threshold (float): Maximum distance to consider points glued.

    Returns:
        list of tuple: Pairs of indices (i, j) such that point_cloud1[i] and point_cloud2[j] are glued.
    """
    # Create KD-trees for efficient nearest-neighbor search
    tree1 = cKDTree(point_cloud1)
    tree2 = cKDTree(point_cloud2)

    # Find all pairs of points within the threshold distance
    indices1 = tree1.query_ball_tree(tree2, r=distance_threshold)

    # Collect pairs of indices
    glued_pairs = []
    for i, neighbors in enumerate(indices1):
        for j in neighbors:
            glued_pairs.append((i, j))

    return glued_pairs

# generalization to multiple point clouds
def find_all_gluing_lines(point_clouds, distance_threshold):
    glued_lines = {}
    for i, j in combinations(range(len(point_clouds)), 2):
        glued_pairs = find_gluing_lines(point_clouds[i], point_clouds[j], distance_threshold)
        glued_lines[(i, j)] = glued_pairs
    return glued_lines

# =============================================================================
# Here begins the code to fit the gluings with splines and measure their lenghts.
def fit_spline_and_calculate_length(glued_points, smoothing_factor=0):
    """
    Fits a 3D spline to a set of points and computes the length of the spline.

    Args:
        glued_points (numpy.ndarray): 3D points, shape (n, 3).
        smoothing_factor (float): Smoothing factor for the spline fitting.
                                  Set to 0 for an interpolating spline.

    Returns:
        tck: Tuple containing the spline representation.
        spline_length: Length of the fitted spline.
    """
    # Ensure glued_points is a numpy array
    glued_points = np.asarray(glued_points)

    # Fit a parametric spline to the 3D data
    tck, u = splprep(glued_points.T, s=smoothing_factor)

    # Define a function to compute the magnitude of the derivative
    def curve_derivative_length(u):
        dx, dy, dz = splev(u, tck, der=1)  # First derivatives
        return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    # Integrate the magnitude of the derivative over the parameter range [0, 1]
    spline_length, _ = quad(curve_derivative_length, 0, 1)

    return tck, spline_length


def order_points_along_curve(points):
    """
    Orders a set of 3D points along a curve by iteratively connecting nearest neighbors.

    Args:
        points (numpy.ndarray): Unordered 3D points, shape (n, 3).

    Returns:
        ordered_points (numpy.ndarray): Points ordered along the curve.
    """
    points = np.asarray(points)
    n_points = len(points)

    # Create a KDTree for efficient nearest-neighbor search
    tree = KDTree(points)

    # Start from the first point
    ordered = [0]
    visited = set(ordered)

    for _ in range(1, n_points):
        last_point_idx = ordered[-1]
        # Find nearest neighbor not yet visited
        distances, indices = tree.query(points[last_point_idx], k=n_points)
        for idx in indices:
            if idx not in visited:
                ordered.append(idx)
                visited.add(idx)
                break

    return points[ordered]

def plot_ordering(original_points, ordered_points):
    """
    Visualizes the difference between unordered and ordered points.

    Args:
        original_points (numpy.ndarray): Unordered 3D points.
        ordered_points (numpy.ndarray): Ordered 3D points.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(*original_points.T, 'o-', label='Original (unordered)', color='blue')
    ax.plot(*ordered_points.T, 'o-', label='Ordered', color='orange')
    ax.legend()
    plt.title('Original vs Ordered Points')
    plt.show()

def plot_spline(points, tck):
    """
    Visualizes the fitted spline compared to the ordered points.

    Args:
        points (numpy.ndarray): Ordered 3D points.
        tck (tuple): Spline representation.
    """
    # Evaluate spline at fine intervals
    u_fine = np.linspace(0, 1, 100)
    spline_points = np.array(splev(u_fine, tck)).T

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(*points.T, 'o', label='Ordered Points', color='orange')
    ax.plot(*spline_points.T, '-', label='Fitted Spline', color='green')
    ax.legend()
    plt.title('Spline Fit vs Ordered Points')
    plt.show()

def remove_duplicates(points):
    """
    Removes duplicate points from a 2D or 3D point cloud.

    Args:
        points (numpy.ndarray): Point cloud, shape (n, d), where d is the dimension (e.g., 2 or 3).

    Returns:
        numpy.ndarray: Point cloud with duplicates removed.
    """
    unique_points = np.unique(points, axis=0)
    return unique_points

def merge_point_clouds(cloud1, cloud2):
    """
    Merges two point clouds and removes duplicate points.

    Args:
        cloud1 (numpy.ndarray): First point cloud, shape (n1, d).
        cloud2 (numpy.ndarray): Second point cloud, shape (n2, d).

    Returns:
        numpy.ndarray: Merged point cloud with duplicates removed.
    """
    combined = np.vstack((cloud1, cloud2))
    return remove_duplicates(combined)

# =============================================================================

def compute_gluing_lengths_and_save(starting_frame, number_of_frames, colors, output_file, path, distance_threshold):

    # Initialize an empty list to store rows
    rows = []

    for j in range(starting_frame, starting_frame + number_of_frames):
        models = []
        row = []  # Start a new row

        for color in colors:
            model_path = os.path.join(base_path, "Session_" + str(j), "fitting_model_" + color + ".pt")
            models.append(fitter.load_model(model_path))

        point_clouds_3d = get_point_clouds(path, models, num_points=3000)

        for (k, point_cloud1), (l, point_cloud2) in combinations(enumerate(point_clouds_3d), 2):

            glued_pairs = find_gluing_lines(point_cloud1, point_cloud2, distance_threshold)

            # Extract glued points from pairs
            glued_points_cloud1 = np.array(point_cloud1[[m for m, n in glued_pairs]])
            glued_points_cloud2 = np.array(point_cloud2[[n for m, n in glued_pairs]])

            glued_points_unordered = merge_point_clouds(glued_points_cloud1, glued_points_cloud2)
            if glued_points_unordered.size == 0:
                row.append(0)
            elif glued_points_unordered.shape[0] <= 3:
                row.append(0)
            else:
                tck, length = fit_spline_and_calculate_length(glued_points_unordered, smoothing_factor=5)
                row.append(length)

        rows.append(row)
        print(f'row = {row}')

    # Convert the list of rows to a 2D numpy array
    result_array = np.array(rows)
    # change the header if whe change the number of leaves.
    np.savetxt(output_file, result_array, delimiter=',', header='01,02,03,12,13,23', comments='')

def compute_gluing_heights_and_save(starting_frame, number_of_frames, colors, output_file, path, distance_threshold):

    # Initialize an empty list to store rows
    rows = []

    for j in range(starting_frame, starting_frame + number_of_frames):
        models = []
        row = []  # Start a new row

        for color in colors:
            model_path = os.path.join(base_path, "Session_" + str(j), "fitting_model_" + color + ".pt")
            models.append(fitter.load_model(model_path))

        point_clouds_3d, points_2d = get_point_clouds(path, models, num_points=3000, get_2d_point_cloud=1)

        for (k, point_cloud1), (l, point_cloud2) in combinations(enumerate(point_clouds_3d), 2):

            glued_pairs = find_gluing_lines(point_cloud1, point_cloud2, distance_threshold)

            # Extract glued points from pairs
            glued_points_cloud1 = np.array(points_2d[[m for m, n in glued_pairs]])
            glued_points_cloud2 = np.array(points_2d[[n for m, n in glued_pairs]])

            # ---------------------------------------Plot for test
            # # Extract x and y coordinates
            # print(f"gluing of k = {k} with l = {l} in frame {j}")
            # x1, y1 = points_2d[:, 0], points_2d[:, 1]
            # x2, y2 = glued_points_cloud2[:, 0], glued_points_cloud2[:, 1]
            #
            # # Create plot
            # fig, ax = plt.subplots()
            # ax.scatter(x1, y1, color='blue', alpha=0.6, label='Point Cloud 1')
            # ax.scatter(x2, y2, color='red', alpha=0.6, label='Point Cloud 2')  # Second point cloud in red
            #
            # # Set axis labels
            # ax.set_xlabel("X-axis")
            # ax.set_ylabel("Y-axis")
            #
            # # Add coordinate axes lines
            # ax.axhline(y=0, color='black', linewidth=1)  # X-axis line
            # ax.axvline(x=0, color='black', linewidth=1)  # Y-axis line
            #
            # # Ensure equal scaling of axes
            # ax.set_aspect('equal', adjustable='datalim')
            #
            # # Adjust plot limits to fit both point clouds
            # all_x = np.concatenate((x1, x2))
            # all_y = np.concatenate((y1, y2))
            # ax.set_xlim(min(all_x) - 1, max(all_x) + 1)
            # ax.set_ylim(min(all_y) - 1, max(all_y) + 1)
            #
            # # Display grid and legend
            # ax.grid(True, linestyle="--", alpha=0.5)
            # ax.legend()
            #
            # # Show plot
            # plt.show()
            # ---------------------------------------

            if glued_points_cloud1.size == 0:
                row.append(0)
            else:
                height1 = np.max(glued_points_cloud1[:, 1])
                row.append(height1)
            if glued_points_cloud2.size == 0:
                row.append(0)
            else:
                height2 = np.max(glued_points_cloud2[:, 1])
                row.append(height2)

        rows.append(row)
        print(f'row = {row}')


    # Convert the list of rows to a 2D numpy array
    result_array = np.array(rows)
    # change the header if whe change the number of leaves.
    np.savetxt(output_file, result_array, delimiter=',', header='01,10,02,20,03,30,12,21,13,31,23,32', comments='')


if __name__ == "__main__":
    # Base path of the directory with the experiment.
    base_path = config.BASE_PATH

    # Path of the leaf contour in space.
    leaf_svg_path = config.LEAF_SVG_PATH
    path = parse_path(leaf_svg_path)

    distance_threshold = 0.03  # Adjust based on your scale

    # colors = ["red", "pink", "blue", "turquoise"]
    # models = []
    # j = 310
    #
    # for color in colors:
    #     model_path = os.path.join(base_path, "Session_" + str(j), "fitting_model_" + color + ".pt")
    #     models.append(fitter.load_model(model_path))
    #
    # point_clouds = get_point_clouds(path, models, num_points=3000)
    # plot_all_leaves(point_clouds, colors=colors)
    #
    # # Example usage
    # # point_cloud1 and point_cloud2 should be numpy arrays with shape (n, 3)
    #
    # distance_threshold = 0.01  # Adjust based on your scale
    # point_cloud1 = point_clouds[1]
    # point_cloud2 = point_clouds[2]
    # glued_pairs = find_gluing_lines(point_cloud1, point_cloud2, distance_threshold)
    #
    # # Extract glued points from pairs
    # glued_points_cloud1 = point_cloud1[[i for i, j in glued_pairs]]
    # point_clouds.append(glued_points_cloud1)
    # glued_points_cloud2 = point_cloud2[[j for i, j in glued_pairs]]
    # point_clouds.append(glued_points_cloud2)
    #
    #
    # print(glued_points_cloud1)
    # print(glued_points_cloud2)
    #
    # colors2 = ["red", "pink", "blue", "turquoise", "yellow", "yellow"]
    # # new_point_clouds = glued_clouds + point_clouds
    # plot_all_leaves(point_clouds, colors=colors2)
    #
    # glued_points_unordered = merge_point_clouds(glued_points_cloud1, glued_points_cloud2)
    # glued_points_ordered = order_points_along_curve(glued_points_unordered)
    #
    # # just for checking
    # plot_ordering(glued_points_ordered, glued_points_ordered)
    #
    # tck, length = fit_spline_and_calculate_length(glued_points_unordered, smoothing_factor=8)
    #
    # plot_spline(glued_points_ordered, tck)
    # print(f"length = {length}")


    offset = 1
    number_of_frames = 320
    colors = ["red", "pink", "blue", "turquoise"]
    output_file = os.path.join(base_path, "length_of_gluings.CSV")
    output_file_2 = os.path.join(base_path, "height_of_gluings.CSV")

    compute_gluing_heights_and_save(offset, number_of_frames, colors, output_file_2, path, distance_threshold)
    compute_gluing_lengths_and_save(offset, number_of_frames, colors, output_file, path, distance_threshold)
