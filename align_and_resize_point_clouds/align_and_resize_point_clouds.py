import config
import open3d as o3d
import numpy as np
import os
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from general_functions import segmentation_and_masking


# -----------------------------------------------------------------------------
def complement_crop(pcd, min_bound, max_bound):
    """
    Crop a point cloud to get all the points outside of a specific bounding box.
    :param pcd: The original open3d point cloud with all the points.
    :param min_bound: The min_bound of the bounding box. (np.ndarray)
    :param max_bound: The max_bound of the bounding box. (np.ndarray)
    :return:
    Return the complement point cloud.
    """

    crop_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    cropped_indices = crop_box.get_point_indices_within_bounding_box(pcd.points)

    # Get the indices of points that were not inside the crop region
    all_indices = np.arange(len(pcd.points))

    complement_indices = np.setdiff1d(all_indices, cropped_indices, assume_unique=True)

    # Create a new point cloud with the complement points
    complement_points = np.asarray(pcd.points)[complement_indices]
    complement_colors = np.asarray(pcd.colors)[complement_indices]
    complement_normals = np.asarray(pcd.normals)[complement_indices]

    complement_pcd = o3d.geometry.PointCloud()
    complement_pcd.points = o3d.utility.Vector3dVector(complement_points)
    complement_pcd.colors = o3d.utility.Vector3dVector(complement_colors)
    complement_pcd.normals = o3d.utility.Vector3dVector(complement_normals)

    return complement_pcd
# -----------------------------------------------------------------------------

def translate_point_cloud(point_cloud, new_origin):
    """
    Function to performa translation in a point clouds coordinate system, so that the origin
    falls on a desired point.
    :param point_cloud: The point cloud in the original coordinates.
    :param new_origin: The point at which to set the origin.
    :return:
    Return the translated point cloud.
    """
    # Calculate translation vector
    translation_vector = -new_origin

    # Apply translation to all points in the point cloud
    translated_points = point_cloud + translation_vector

    return translated_points
# -----------------------------------------------------------------------------

def fit_circle(points):
    """
    Function to find the center and radius of the circle that best fits a point cloud.I believe this is after the
    directions of the coordinates have been fixed.
    :param points: The point cloud.
    :return:
    Return the center and radius of the circle.
    """
    # Initial guess for circle center and radius
    initial_guess = np.concatenate((np.mean(points, axis=0), [np.max(np.linalg.norm(points - np.mean(points, axis=0), axis=1))]))

    # Define the objective function (sum of squared distances)
    def objective(params):
        center = params[:2]
        radius = params[2]
        return np.sum((np.linalg.norm(points - center, axis=1) - radius) ** 2)

    # Minimize the objective function to find the optimal circle parameters
    result = minimize(objective, initial_guess)

    # Extract the optimal circle parameters
    center = result.x[:2]
    radius = result.x[2]

    return center, radius
# -----------------------------------------------------------------------------

def find_orientation(pcd):
    """
    Rotate the point cloud around the z axis, so that the x axis of the cloud points in the direction of the
    blue column. This function works under the assumption that we have already fitted the red circle, such that
    its plane corresponds with the xy plane and the center with the coordinate center of the point cloud.
    :return:
    Return the coordinates of the rotated point cloud.
    """

    # Extract points
    points = np.asarray(pcd.points)

    # Define the cropping box
    min_bound = np.array([-2.0, -2.0, -2.0])
    max_bound = np.array([2.0, 2.0, 3.0])

    complement_pcd = complement_crop(pcd, min_bound, max_bound)

    # Find the color differences for the complement point clouds.
    cropped_colors = np.asarray(complement_pcd.colors)

    cropped_color_differences = segmentation_and_masking.get_color_differences(cropped_colors)


    # Set up and apply a second mask for the blue points depending on those color differences.
    trapezoid = np.array([
        [-0.007, 0.218],  # Point A
        [0.102, 0.218],  # Point B
        [0.102, 0.038],  # Point C
        [-0.007, 0.038]  # Point D
    ])
    mask2 = segmentation_and_masking.mask_points_within_trapezoid(cropped_colors, cropped_color_differences, trapezoid,
                                                                 extra_color=2, extra_color_range=(0.2, 1.0))

    indexes2 = np.where(mask2 == 1)[0]
    pcd_filtered2 = complement_pcd.select_by_index(indexes2)
    pcd_filtered2, outliers2 = pcd_filtered2.remove_radius_outlier(5, 0.05)

    if config.SHOW_ALL_PLOTS == True:
        o3d.visualization.draw_geometries([pcd_filtered2])
    blue_points = np.asarray(pcd_filtered2.points)

    # Compute the average of the blue points
    average_blue = np.mean(blue_points, axis=0)

    # Project the average onto the xy-plane
    average_blue_xy = average_blue[:2]

    # Compute the angle to rotate around the z-axis
    angle = np.arctan2(average_blue_xy[1], average_blue_xy[0])

    # Create rotation matrix for rotating around the z-axis
    # The reason for the negative angle is that we are rotating the point cloud and not the
    # set of axes. But the angle calculated is the one for the set of axes, so we need to move the
    # point cloud in the opposite direction.
    rotation_axis = np.array([0, 0, 1])
    rotation_angle = -angle
    rotation_vector = rotation_axis * rotation_angle

    # Create rotation matrix from rotation vector
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_vector)

    # Apply rotation to the point cloud
    pcd.rotate(rotation_matrix, center=(0, 0, 0))

    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)

    # Save or visualize the transformed point cloud

    if config.SHOW_ALL_PLOTS == True:
        print("fully rotated pcd")
        o3d.visualization.draw_geometries([pcd, axes])

    return pcd
# -----------------------------------------------------------------------------

def plot_point_cloud(point_cloud, color='b', label=None):
    plt.scatter(point_cloud[:, 0], point_cloud[:, 1], c=color, label=label)
# -----------------------------------------------------------------------------

def filter_by_color_differences(pcd):

    # Find the color differences for the complement point clouds.
    cropped_colors = np.asarray(pcd.colors)

    cropped_color_differences = segmentation_and_masking.get_color_differences(cropped_colors)

    # Set up and apply a mask depending on those color differences.
    trapezoid = np.array([
        [-0.56, 0.05],  # Point A
        [-0.3, 0.05],  # Point B
        [-0.3, -0.055],  # Point C
        [-0.56, -0.055]  # Point D
    ])
    mask = segmentation_and_masking.mask_points_within_trapezoid(cropped_colors,cropped_color_differences, trapezoid, extra_color=1, extra_color_range=(0,0.5))

    indexes = np.where(mask == 1)[0]
    pcd_filtered = complement_pcd.select_by_index(indexes)
    pcd_filtered, outliers = pcd_filtered.remove_radius_outlier(5, 0.05)

    return pcd_filtered
# NEED TO MODIFY THIS FUNCTION TO INCLUDE THE SETUP.
# -----------------------------------------------------------------------------

def find_rotation_matrix(pcd_filtered):
    points = np.asarray(pcd_filtered.points)

    # Randomly select points from the filtered point cloud.
    subset_indices = np.random.choice(range(len(points)), size=50, replace=False)
    subset_points = points[subset_indices]

    # Convert subset of points into a point cloud.
    subset_cloud = o3d.geometry.PointCloud()
    subset_cloud.points = o3d.utility.Vector3dVector(subset_points)

    # Use RANSAC to estimate a plane from the subset of points.
    plane_model, inliers = subset_cloud.segment_plane(distance_threshold=0.01,
                                                      ransac_n=3,
                                                      num_iterations=1000)

    # Extract plane parameters from the model.
    [a, b, c, d] = plane_model

    # Print the equation of the fitted plane
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    # Define mesh vertices for the plane
    mesh_vertices = []
    mesh_vertices.append([-1, -1, (a + b - d) / c])  # Point 1
    mesh_vertices.append([1, -1, (-a + b - d) / c])  # Point 2
    mesh_vertices.append([1, 1, (-a - b - d) / c])  # Point 3
    mesh_vertices.append([-1, 1, (a - b - d) / c])  # Point 4

    # Define mesh triangles
    mesh_triangles = [[0, 1, 2], [0, 2, 3]]

    # Create mesh object
    plane_mesh = o3d.geometry.TriangleMesh()
    plane_mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices)
    plane_mesh.triangles = o3d.utility.Vector3iVector(mesh_triangles)

    # Visualize the plane and points using Open3D.
    plane_mesh.paint_uniform_color([0.7, 0.7, 0.7])
    subset_cloud.paint_uniform_color([1, 0, 0])

    if config.SHOW_ALL_PLOTS == True:
        print("fitting ring to plane")
        o3d.visualization.draw_geometries([subset_cloud, plane_mesh])

    # Calculate rotation angle and axis
    sign_d = 1 if d > 0 else -1
    normal = sign_d * np.array([a, b, c])
    target_normal = np.array([0, 0, 1])  # Z unit vector
    rotation_axis = np.cross(normal, target_normal)
    rotation_axis /= np.linalg.norm(rotation_axis)
    angle = np.arccos(np.dot(normal, target_normal) / (np.linalg.norm(normal) * np.linalg.norm(target_normal)))

    # Create rotation matrix
    rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)

    return rotation_matrix
# -----------------------------------------------------------------------------

def find_circle_center_and_radius(pcd_filtered):
    projected_ring = np.asarray(pcd_filtered.points)[:, :2]
    projected_ring_height = np.mean(np.asarray(pcd_filtered.points)[:, 2])
    print(f'projected_ring = {projected_ring}')
    print(f'projected_ring_height = {projected_ring_height}')

    # Fit the circle to the 2D points
    points_2d = projected_ring
    # points_2d = np.array([projected_ring[:, 0], projected_ring[:, 1]])
    print(f'points_2d shape = {points_2d.shape}')
    circle_center, circle_radius = fit_circle(points_2d)

    full_circle_center = np.append(circle_center, projected_ring_height)

    print("Center of the circle:", full_circle_center)
    print("Radius of the circle:", circle_radius)

    if config.SHOW_ALL_PLOTS == True:
        center_point = np.array([[full_circle_center[0], full_circle_center[1], full_circle_center[2]]])
        center_pcd = o3d.geometry.PointCloud()
        center_pcd.points = o3d.utility.Vector3dVector(center_point)
        print("Ring together with its center.")
        o3d.visualization.draw_geometries([pcd_filtered, center_pcd])

    return full_circle_center, circle_radius
# -----------------------------------------------------------------------------

def normalize_point_cloud(pcd, circle_radius):
    """ Scales the point cloud so that the detected circle's radius becomes 1.5. """
    target_radius = config.red_circle_radius_scale
    scale_factor = target_radius / circle_radius
    pcd.scale(scale_factor, center=(0, 0, 0))
    return pcd
# -----------------------------------------------------------------------------



if __name__ == "__main__":
    number_of_frames = config.number_of_frames

    for i in range(number_of_frames):
        j = i + config.starting_frame

        pcd_path = os.path.join(config.base_path, "Session_" + str(j), "fused.ply")
        pcd = o3d.io.read_point_cloud(pcd_path)


        complement_pcd = complement_crop(pcd, config.min_bound, config.max_bound)

        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)

        if config.SHOW_ALL_PLOTS == True:
            # Visualize the complement point cloud
            print("Complement point cloud.")
            o3d.visualization.draw_geometries([complement_pcd, axes])

        pcd_filtered = filter_by_color_differences(complement_pcd)

        if config.SHOW_ALL_PLOTS == True:
            print("Filtered pcd with just the ring.")
            o3d.visualization.draw_geometries([pcd_filtered])

        rotation_matrix = find_rotation_matrix(pcd_filtered)

        # Apply rotation to point cloud
        pcd.rotate(rotation_matrix, center=(0, 0, 0))
        pcd_filtered.rotate(rotation_matrix, center=(0, 0, 0))

        # Create a coordinate frame (axes)
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)

        if config.SHOW_ALL_PLOTS == True:
            print("Rotated point cloud.")
            o3d.visualization.draw_geometries([pcd, axes])

        circle_center, circle_radius = find_circle_center_and_radius(pcd_filtered)


        # Example 3D point cloud (replace this with your actual point cloud)
        point_cloud = np.asarray(pcd.points)

        # Example desired point to be the new origin (replace this with your desired point)
        new_origin = np.array([circle_center[0], circle_center[1], circle_center[2]])

        # Apply translation to the point cloud
        translated_pcd_points = translate_point_cloud(point_cloud, new_origin)

        translated_pcd = o3d.geometry.PointCloud()
        translated_pcd.points = o3d.utility.Vector3dVector(translated_pcd_points)
        translated_pcd.colors = pcd.colors
        translated_pcd.normals = pcd.normals

        final_pcd = find_orientation(translated_pcd)

        scaled_final_pcd = normalize_point_cloud(final_pcd, circle_radius)

        if config.SHOW_ALL_PLOTS == True:
            print("Scaled final point cloud for saving.")
            o3d.visualization.draw_geometries([scaled_final_pcd, axes])

        o3d.io.write_point_cloud(pcd_path, scaled_final_pcd)