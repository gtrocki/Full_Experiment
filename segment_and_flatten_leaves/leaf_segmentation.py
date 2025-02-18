import numpy as np
import open3d as o3d
import config
from general_functions import point_cloud_manipulations as pcd_man
from general_functions import segmentation_and_masking
import matplotlib.pyplot as plt
from pathlib import Path

def segment_single_leaf(pcd_path: str, trapezoid: np.ndarray, extra_color: int = None, extra_color_range: tuple = 0) -> o3d.geometry.PointCloud:
    """
    Receive information about the path of a point cloud and the color ranges for the points to be kept,
    and segment the points that fit the color range. Keep only those points whose difference between red and
    green, and between green and blue fall within the specified ranges. Return the point cloud of only the
    relevant points.
    :param pcd_path: String with the full path to the point cloud file.
    :param trapezoid: An ndarray representing the four vertices of a trapezoid which contains all the colors we want
    to keep.
    :param extra_color: The number of an extra color to filter by if necessary, because having just info about
    differences disregards one piece of information.
    :param extra_color_range: The range by which to filter in the extra color.
    :return:
    Return the point cloud of only the relevant points.
    """
    # create test data
    input_file = pcd_path
    pcd = o3d.io.read_point_cloud(Path(input_file))  # Read the point cloud

    # Obtain a rotation matrix and rotate the point cloud to align the global coordinates.
    # R = pcd.get_rotation_matrix_from_xyz((-np.pi * 2.75 / 4, 0, 0))
    # pcd.rotate(R, center=(0, 0, 0))

    # Create a coordinate frame (axes) in order to plot and check coordinates are alright.
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1)


    if config.SHOW_ALL_PLOTS == True:
        # Visualize the point cloud with the coordinate frame
        o3d.visualization.draw_geometries([pcd, axes])
        o3d.visualization.draw_geometries([pcd])
    # cropped_pcd = pcd_man.crop_point_cloud(pcd, "3d_images/volume_for_cropping3.json")
    # cropped_pcd = pcd_man.crop_point_cloud(pcd, "3d_images/volume_for_cropping6.json")
    # Define the region to keep (CropBox)
    min_bound = config.CROPPING_MIN_BOUND  # Minimum bounds of the box
    max_bound = config.CROPPING_MAX_BOUND  # Maximum bounds of the box
    crop_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
    cropped_pcd = pcd.crop(crop_box)

    cropped_pcd.estimate_normals()
    if config.SHOW_ALL_PLOTS == True:
        o3d.visualization.draw_geometries([cropped_pcd])

    cropped_colors = np.asarray(cropped_pcd.colors)
    points = np.asarray(cropped_pcd.points)

    cropped_color_differences = segmentation_and_masking.get_color_differences(cropped_colors)

    if config.SHOW_ALL_PLOTS == True:
        plt.scatter(cropped_color_differences[:, 0], cropped_color_differences[:, 1], s=1, c=cropped_colors)
        plt.show()

    mask = segmentation_and_masking.mask_points_within_trapezoid(cropped_colors, cropped_color_differences, trapezoid,
                                                        extra_color= extra_color, extra_color_range=extra_color_range)

    pcd_filtered = pcd_man.filter_pcd_from_mask(cropped_pcd, mask)
    # o3d.visualization.draw_geometries([pcd_filtered])
    # pcd_filtered = pcd_man.remove_radius_outliers(pcd_filtered, min_points=10, radius=0.025)
    pcd_filtered = pcd_man.remove_radius_outliers(pcd_filtered, min_points=50, radius=0.3)
    pcd_filtered = pcd_man.remove_radius_outliers(pcd_filtered, min_points=15, radius=0.05)
    # Repeat first filtering just in case.
    pcd_filtered = pcd_man.remove_radius_outliers(pcd_filtered, min_points=50, radius=0.3)

    # o3d.visualization.draw_geometries([pcd_filtered])
    # o3d.io.write_point_cloud("3d_images/Basic_forces/fused_3_filtered.ply", pcd_filtered3)

    return pcd_filtered