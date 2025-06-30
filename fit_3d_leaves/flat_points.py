# import numpy as np
# import open3d as o3d
# import matplotlib.pyplot as plt
# import os
# import fitter
# import measure_zippers
# import config
# from svgpath2mpl import parse_path
#
# def compute_transition_y(points_3d, points_2d, knn=30, num_highlight=20):
#     """
#     Computes the transition y-coordinate (in 2D) where the surface normal switches
#     from pointing inward to outward based on dot product with (X, Y, 0).
#
#     Returns:
#         transition_y: float or None
#         transition_indices: indices of points closest to transition y (in 2D)
#         dot_products: array of dot products between normals and (X, Y, 0)
#         sorted_y: sorted 2D y-coordinates
#         sorted_dot: dot products sorted by 2D y
#     """
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points_3d)
#     pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
#     pcd.orient_normals_to_align_with_direction(np.array([0, 0, 1]))
#     normals = np.asarray(pcd.normals)
#
#     XY_vectors_3d = np.hstack((points_3d[:, :2], np.zeros((len(points_3d), 1))))
#     dot_products = np.einsum("ij,ij->i", normals, XY_vectors_3d)
#
#     sorted_indices = np.argsort(points_2d[:, 1])
#     sorted_y = points_2d[sorted_indices, 1]
#     sorted_dot = dot_products[sorted_indices]
#
#     sign_changes = np.where((sorted_dot[:-1] < 0) & (sorted_dot[1:] >= 0))[0]
#
#     if len(sign_changes) > 0:
#         transition_index = sign_changes[0]
#         transition_y = (sorted_y[transition_index] + sorted_y[transition_index + 1]) / 2
#         distances_to_y0 = np.abs(points_2d[:, 1] - transition_y)
#         transition_indices = np.argsort(distances_to_y0)[:num_highlight]
#         return transition_y, transition_indices, dot_products, sorted_y, sorted_dot
#     else:
#         return None, [], dot_products, sorted_y, sorted_dot
#
# def plot_transition(points_3d, dot_products, sorted_y, sorted_dot, transition_y, transition_indices):
#     """
#     Plots the 3D point cloud and dot product curve highlighting the transition.
#     """
#     # 3D visualization
#     colors = np.tile([0.6, 0.6, 0.6], (len(points_3d), 1))
#     if len(transition_indices) > 0:
#         colors[transition_indices] = [1.0, 0.0, 0.0]
#
#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(points_3d)
#     pcd.colors = o3d.utility.Vector3dVector(colors)
#     o3d.visualization.draw_geometries([pcd], window_name="Normal Transition Highlight")
#
#     # Dot product plot
#     plt.figure(figsize=(8, 4))
#     plt.plot(sorted_y, sorted_dot, label="Dot product (normal · (X,Y,0))")
#     plt.axhline(0, color='gray', linestyle='--')
#     if transition_y is not None:
#         plt.axvline(transition_y, color='red', linestyle='--', label="Transition y")
#     plt.xlabel("2D y-coordinate")
#     plt.ylabel("Dot product")
#     plt.title("Normal · (X,Y,0) vs 2D y-coordinate")
#     plt.legend()
#     plt.grid(True, linestyle="--", alpha=0.5)
#     plt.tight_layout()
#     plt.show()
#
# # Main workflow
# if __name__ == "__main__":
#     base_path = os.path.join("C:", "\\", "Users", "michalro", "PycharmProjects", "Full_Experiment", "data", "2025-04-28-17-25-43")
#
#     leaf_svg_path = config.LEAF_SVG_PATH
#     path = parse_path(leaf_svg_path)
#     models = []
#     for color in ["red","blue","pink","turquoise"]:
#         model_path = os.path.join(base_path, "Session_1", f"fitting_model_{color}.pt")
#         models.append(fitter.load_model(model_path))
#
#     point_clouds_3d, points_2d = measure_zippers.get_point_clouds(path, models, num_points=3000, get_2d_point_cloud=1)
#     points_3d = point_clouds_3d[0]
#
#     transition_y, transition_indices, dot_products, sorted_y, sorted_dot = compute_transition_y(points_3d, points_2d)
#
#     if transition_y is not None:
#         print(f"Transition occurs around 2D y = {transition_y:.4f}")
#     else:
#         print("No transition detected.")
#
#     plot_transition(points_3d, dot_products, sorted_y, sorted_dot, transition_y, transition_indices)
"""
File to find the ridge in the leaves.
"""

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import os
import fitter
import measure_zippers
import config
from svgpath2mpl import parse_path

def compute_transition_y(points_3d, points_2d, knn=30, num_highlight=20):
    """
    Computes the transition y-coordinate (in 2D) where the surface normal switches
    from pointing inward to outward based on dot product with (X, Y, 0).

    Returns:
        transition_y: float or None
        transition_indices: indices of points closest to transition y (in 2D)
        dot_products: array of dot products between normals and (X, Y, 0)
        sorted_y: sorted 2D y-coordinates
        sorted_dot: dot products sorted by 2D y
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    pcd.orient_normals_to_align_with_direction(np.array([0, 0, 1]))
    normals = np.asarray(pcd.normals)

    XY_vectors_3d = np.hstack((points_3d[:, :2], np.zeros((len(points_3d), 1))))
    dot_products = np.einsum("ij,ij->i", normals, XY_vectors_3d)

    sorted_indices = np.argsort(points_2d[:, 1])
    sorted_y = points_2d[sorted_indices, 1]
    sorted_dot = dot_products[sorted_indices]

    sign_changes = np.where((sorted_dot[:-1] < 0) & (sorted_dot[1:] >= 0))[0]

    if len(sign_changes) > 0:
        transition_index = sign_changes[0]
        transition_y = (sorted_y[transition_index] + sorted_y[transition_index + 1]) / 2
        distances_to_y0 = np.abs(points_2d[:, 1] - transition_y)
        transition_indices = np.argsort(distances_to_y0)[:num_highlight]
        return transition_y, transition_indices, dot_products, sorted_y, sorted_dot
    else:
        return None, [], dot_products, sorted_y, sorted_dot

def compute_transition_ys_for_colors(base_path, colors, svg_path, session_number = 1):
    models = []
    for color in colors:
        model_path = os.path.join(base_path, "Session_" + str(session_number) , f"fitting_model_{color}.pt")
        models.append(fitter.load_model(model_path))

    point_clouds_3d, shared_2d_points = measure_zippers.get_point_clouds(svg_path, models, num_points=3000, get_2d_point_cloud=1)

    # Replicate the 2D points so we have one for each 3D point cloud
    point_clouds_2d = [shared_2d_points.copy() for _ in colors]

    y_0_list = []
    for points_3d, points_2d in zip(point_clouds_3d, point_clouds_2d):
        transition_y, *_ = compute_transition_y(points_3d, points_2d)
        y_0_list.append(transition_y)

    return y_0_list, point_clouds_3d, point_clouds_2d

def plot_transition(points_3d, dot_products, sorted_y, sorted_dot, transition_y, transition_indices):
    """
    Plots the 3D point cloud and dot product curve highlighting the transition.
    """
    # 3D visualization
    colors = np.tile([0.6, 0.6, 0.6], (len(points_3d), 1))
    if len(transition_indices) > 0:
        colors[transition_indices] = [1.0, 0.0, 0.0]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd], window_name="Normal Transition Highlight")

    # Dot product plot
    plt.figure(figsize=(8, 4))
    plt.plot(sorted_y, sorted_dot, label="Dot product (normal · (X,Y,0))")
    plt.axhline(0, color='gray', linestyle='--')
    if transition_y is not None:
        plt.axvline(transition_y, color='red', linestyle='--', label="Transition y")
    plt.xlabel("2D y-coordinate")
    plt.ylabel("Dot product")
    plt.title("Normal · (X,Y,0) vs 2D y-coordinate")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def get_left_right_extrema_indices(point_clouds_2d, y_0_list, y_tolerance=0.01):
    """
    For each 2D point cloud and corresponding y_0, find the points within a tolerance of y_0,
    then among those pick the min and max x value.

    Args:
        point_clouds_2d: list of (N, 2) numpy arrays
        y_0_list: list of float values
        y_tolerance: float, allowable distance from y_0 to include in search

    Returns:
        left_points: list of indices of points with minimal x near y_0
        right_points: list of indices of points with maximal x near y_0
    """
    left_points = []
    right_points = []

    for points_2d, y_0 in zip(point_clouds_2d, y_0_list):
        if y_0 is None:
            left_points.append(None)
            right_points.append(None)
            continue

        # Select points within the y_tolerance
        y_diff = np.abs(points_2d[:, 1] - y_0)
        candidate_mask = y_diff < y_tolerance
        candidate_indices = np.where(candidate_mask)[0]

        if len(candidate_indices) == 0:
            # Fallback to closest point if no one falls within tolerance
            candidate_indices = [np.argmin(y_diff)]

        x_values = points_2d[candidate_indices, 0]
        min_x_idx = candidate_indices[np.argmin(x_values)]
        max_x_idx = candidate_indices[np.argmax(x_values)]

        left_points.append(min_x_idx)
        right_points.append(max_x_idx)

    return left_points, right_points

def compute_consecutive_distances(point_clouds_3d, left_indices, right_indices):
    """
    Computes Euclidean distances between the right point of one leaf and the
    left point of the next leaf (in a circular fashion).

    Args:
        point_clouds_3d: list of (N, 3) numpy arrays
        left_indices: list of ints or None (indices of left points)
        right_indices: list of ints or None (indices of right points)

    Returns:
        distances: list of floats or None (distance between right_i and left_{i+1})
    """
    n = len(point_clouds_3d)
    distances = []

    for i in range(n):
        right_idx = right_indices[i]
        left_idx = left_indices[(i + 1) % n]  # wrap around

        # Check for None (missing transition point)
        if right_idx is None or left_idx is None:
            distances.append(None)
            continue

        point_right = point_clouds_3d[i][right_idx]
        point_left = point_clouds_3d[(i + 1) % n][left_idx]

        distance = np.linalg.norm(point_right - point_left)
        distances.append(distance)

    return distances

def visualize_transition_points_all_leaves_fixed_colors(point_clouds_3d, point_clouds_2d, y_0_list, left_indices, right_indices):
    """
    Visualizes all 3D point clouds together with:
        - Points closest to y_0 in yellow
        - Left-most points at y_0 in blue
        - Right-most points at y_0 in red
        - All other points get a fixed base color per leaf index

    Leaf color map:
        Leaf 0 -> Green
        Leaf 1 -> Cyan
        Leaf 2 -> Magenta
        Leaf 3 -> Orange

    Args:
        point_clouds_3d: list of (N, 3) numpy arrays
        point_clouds_2d: list of (N, 2) numpy arrays
        y_0_list: list of float or None
        left_indices: list of ints or None
        right_indices: list of ints or None
    """
    geometries = []

    # Fixed colors for each leaf index (up to 4)
    base_colors = [
        [0.0, 1.0, 0.0],   # Green      (Leaf 0)
        [0.0, 1.0, 1.0],   # Cyan       (Leaf 1)
        [1.0, 0.0, 1.0],   # Magenta    (Leaf 2)
        [1.0, 0.5, 0.0],   # Orange     (Leaf 3)
    ]

    for i, (points_3d, points_2d, y_0, left_idx, right_idx) in enumerate(zip(point_clouds_3d, point_clouds_2d, y_0_list, left_indices, right_indices)):
        N = len(points_3d)
        base_color = base_colors[i % 4]  # Safe fallback even if >4 by accident
        colors = np.tile(base_color, (N, 1))  # Default color for this leaf

        if y_0 is not None:
            y_diff = np.abs(points_2d[:, 1] - y_0)
            min_y_indices = np.where(y_diff == np.min(y_diff))[0]
            colors[min_y_indices] = [1.0, 1.0, 0.0]  # Yellow

            if left_idx is not None:
                colors[left_idx] = [0.0, 0.0, 1.0]  # Blue

            if right_idx is not None:
                colors[right_idx] = [1.0, 0.0, 0.0]  # Red

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(pcd)

    o3d.visualization.draw_geometries(geometries, window_name="Fixed Colors for Leaves + Transition Points")

def visualize_leaf_split_by_y0(points_3d, points_2d, y_0, buffer=0.005):
    """
    Visualizes a single 3D leaf, coloring:
      - Points with y > y_0 in red
      - Points with y < y_0 in blue
      - Points with y in (y_0 - buffer, y_0 + buffer) in white

    Args:
        points_3d: (N, 3) numpy array of 3D coordinates
        points_2d: (N, 2) numpy array of corresponding 2D coordinates
        y_0: float, transition y value
        buffer: float, small window around y_0 for white highlight
    """
    N = len(points_3d)
    colors = np.zeros((N, 3))  # Start with black (not necessary, just explicit)

    y_coords = points_2d[:, 1]
    upper_mask = y_coords > y_0 + buffer
    lower_mask = y_coords < y_0 - buffer
    transition_mask = ~ (upper_mask | lower_mask)

    colors[upper_mask] = [1.0, 0.0, 0.0]     # Red
    colors[lower_mask] = [0.0, 0.0, 1.0]     # Blue
    colors[transition_mask] = [1.0, 1.0, 1.0]  # White

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([pcd], window_name="Leaf split by y₀")

# Main workflow
if __name__ == "__main__":
    base_path = os.path.join("C:", "\\", "Users", "michalro", "PycharmProjects", "Full_Experiment", "data", "2024-05-29-11-18-42")

    leaf_svg_path = config.LEAF_SVG_PATH
    path = parse_path(leaf_svg_path)
    colors = ["red", "blue", "pink", "turquoise"]
    y_0_list, point_clouds_3d, point_clouds_2d = compute_transition_ys_for_colors(base_path, colors, path,
                                                                                  session_number=203)

    left_indices, right_indices = get_left_right_extrema_indices(point_clouds_2d, y_0_list)

    distances = compute_consecutive_distances(point_clouds_3d, left_indices, right_indices)
    print(f"List of transition y's: {y_0_list}")
    print(f"Sequential distances: {distances}")

    points_3d = point_clouds_3d[0]
    points_2d = point_clouds_2d[0]

    print(f"points_3d = {points_3d}")
    print(f"left_indices = {left_indices}")
    print(f"right_indices = {right_indices}")


    transition_y, transition_indices, dot_products, sorted_y, sorted_dot = compute_transition_y(points_3d, points_2d)

    if transition_y is not None:
        print(f"Transition occurs around 2D y = {transition_y:.4f}")
    else:
        print("No transition detected.")

    plot_transition(points_3d, dot_products, sorted_y, sorted_dot, transition_y, transition_indices)

    visualize_transition_points_all_leaves_fixed_colors(point_clouds_3d, point_clouds_2d, y_0_list, left_indices,
                                                        right_indices)

    visualize_leaf_split_by_y0(point_clouds_3d[0], point_clouds_2d[0], y_0_list[0])