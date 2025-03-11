# Module for the manipulations of the point cloud we are analyzing
import numpy as np
import open3d as o3d
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
import scipy.stats
import scipy.spatial
from scipy.spatial import distance
import math
from PIL import Image
from sklearn.base import BaseEstimator, TransformerMixin
# ------------------------------------------------------------------------------


def crop_point_cloud(pcd: o3d.geometry.PointCloud, json_file_name: str) -> o3d.geometry.PointCloud:
    """
    Receive a point cloud and a json file describing a polygon and crop the point cloud according to the json file.
    :param pcd: Original point cloud.
    :param json_file_name: Name of the file that specifies how to crop.
    :return:
    Return the cropped point cloud.
    """
    vol = o3d.visualization.read_selection_polygon_volume(json_file_name)
    cropped_pcd = vol.crop_point_cloud(pcd)
    return cropped_pcd
# ------------------------------------------------------------------------------


def filter_by_color(pcd: o3d.geometry.PointCloud, r_aim: float = 0.7, g_aim: float = 0.1,
                    b_aim: float = 0.1, r_error: float = 0.25, g_error: float = 0.25,
                    b_error: float = 0.25) -> o3d.geometry.PointCloud:
    """
    Receive a point cloud and a range of colors in RGB and filter according to this range.(RGB range between 0 and 1
    as is standard for open3d).
    :param pcd: Original point cloud.
    :param r_aim: Target value for red.
    :param g_aim: Target value for green.
    :param b_aim: Target value for blue.
    :param r_error: Error margin for red.
    :param g_error: Error margin for green.
    :param b_error: Error margin for blue.
    :return:
    Return a point cloud containing just the points with colors in the range specified.
    """
    colors = np.asarray(pcd.colors)
    r_mask = np.array((colors[:, 0] > (r_aim - r_error)) & (colors[:, 0] < (r_aim + r_error)))
    g_mask = np.array((colors[:, 1] > (g_aim - g_aim)) & (colors[:, 1] < (g_aim + g_error)))
    b_mask = np.array((colors[:, 2] > (b_aim - b_aim)) & (colors[:, 2] < (b_aim + b_error)))
    mask = r_mask & g_mask & b_mask
    pcd_filtered = filter_pcd_from_mask(pcd, mask)
    return pcd_filtered
# ------------------------------------------------------------------------------


def filter_by_color_hsv(pcd: o3d.geometry.PointCloud, h_aim: float = 0, s_aim: float = 0.8,
                        v_aim: float = 0.8, h_error: float = 0.1, s_error: float = 0.25,
                        v_error: float = 0.25) -> o3d.geometry.PointCloud:
    """
    Receive a point cloud and a range of colors in HSV and filter for this range.(HSV range between 0 and 1 as is
     standard for open3d).
    :param pcd: Original point cloud.
    :param h_aim: Target value of hue.
    :param s_aim: Target value of saturation.
    :param v_aim: Target value of value.
    :param h_error: Error margin of hue.
    :param s_error: Error margin of saturation.
    :param v_error: Error margin of value.
    :return:
    Return a point cloud containing just the points with colors in the range specified.
    """
    colors_rgb = np.asarray(pcd.colors)
    colors_hsv = mcolors.rgb_to_hsv(colors_rgb)
    h_mask = get_mask_by_hue_hsv(pcd, h_aim, h_error)
    s_mask = np.array((colors_hsv[:, 1] > (s_aim - s_aim)) & (colors_hsv[:, 1] < (s_aim + s_error)))
    v_mask = np.array((colors_hsv[:, 2] > (v_aim - v_aim)) & (colors_hsv[:, 2] < (v_aim + v_error)))
    mask = h_mask & s_mask & v_mask
    pcd_filtered = filter_pcd_from_mask(pcd, mask)
    return pcd_filtered
# ------------------------------------------------------------------------------


def get_mask_by_hue_hsv(pcd: o3d.geometry.PointCloud, h_aim: float = 0,
                        h_error: float = 0.1) -> np.ndarray:
    """
    Receive a point cloud and a range of hue values in HSV to filter.
    :param pcd:
    :param h_aim:
    :param h_error:
    :return:
    Return a mask with 1's in the indexes corresponding to the
    points with colors within that range.
    """
    colors_rgb = np.asarray(pcd.colors)
    colors_hsv = mcolors.rgb_to_hsv(colors_rgb)
    h_low = (h_aim - h_error)
    h_high = (h_aim + h_error)
    if (h_low < 0):
        h_low += 1
    if (h_high > 1):
        h_high -= 1
    if (h_low < h_high):
        h_mask = np.array((colors_hsv[:, 0] > h_low) & (colors_hsv[:, 0] < h_high))
    else:
        h_mask = np.array((colors_hsv[:, 0] > h_low) | (colors_hsv[:, 0] < h_high))
    return h_mask
# ------------------------------------------------------------------------------


def filter_pcd_from_mask(pcd: o3d.geometry.PointCloud, mask: np.ndarray) -> o3d.geometry.PointCloud:
    """
    Receive a point cloud and a mask and apply the mask.
    :param pcd: Original point cloud.
    :param mask: Array of 1s and 0s.
    :return:
    Return a point cloud of just the points agreeing with the nonzero values in the mask.
    """
    indexes = np.where(mask == 1)[0]
    pcd_filtered = pcd.select_by_index(indexes)
    return pcd_filtered
# ------------------------------------------------------------------------------


def filter_n_biggest_labels(pcd: o3d.geometry.PointCloud, labels: np.ndarray, n: int,
                            unwanted_label: int = None) -> o3d.geometry.PointCloud:
    """
    Receive a point cloud, an array of labels for its points, a number, and an optional unwanted label,
    and return a point cloud comprised of the points corresponding to the n most frequent labels.
    :param pcd: Original point cloud.
    :param labels: Array labeling the points with the corresponding cluster number.
    :param n: The number of clusters that we want to keep.
    :param unwanted_label: A label for a cluster that we want to ommit regadless of its size.
    :return:
    Return the filtered point cloud.
    """
    values, counts = np.unique(labels, return_counts=True)
    if values.size <= n:
        return pcd
    ind = np.argpartition(-counts, kth=n)[:n]
    mask = np.zeros(labels.shape)
    if (unwanted_label != None and unwanted_label in values[ind] and counts.size > (n + 1)):
        ind = np.argpartition(-counts, kth=(n + 1))[:(n + 1)]
        mask[[i for i, x in enumerate(labels) if (x in values[ind] and x != unwanted_label)]] = 1
    else:
        mask[[i for i, x in enumerate(labels) if (x in values[ind] and x != unwanted_label)]] = 1
    pcd_filtered = filter_pcd_from_mask(pcd, mask)
    return pcd_filtered
# ------------------------------------------------------------------------------


def find_median_label(labels: np.ndarray) -> int:
    """
    Receive an array of labels and return the label with the median size.
    :param labels: The result of clustering, which is an array containing labels for each point telling to which
    cluster it belongs.
    :return:
    Return the median size of a cluster.
    """
    values, counts = np.unique(labels, return_counts=True)
    median = np.median(counts)
    return median
# ------------------------------------------------------------------------------


def filter_around_median_label(pcd: o3d.geometry.PointCloud, labels: np.ndarray, spread: int = 2000,
                               unwanted_label: int = -1) -> o3d.geometry.PointCloud:
    """
    Receive a point cloud, labels from clustering and a spread and an unwanted label and return the filtered point cloud
    keeping all the labels that are at least one spread away from the median.
    :param pcd: Original point cloud.
    :param labels: The result from a clustering function which is an array containing labels for each point of which
    cluster it belongs to.
    :param spread: How much to deviate from the median size of a label before we refrain from discarding a cluster.
    :param unwanted_label: Label that we want to discard regardless of its size.
    :return:
    Return the filtered point cloud after selecting the right clusters.
    """
    values, counts = np.unique(labels, return_counts=True)
    median = np.median(counts)
    target_values_indeces = [i for i, x in enumerate(values) if (counts[i] > (median + spread) or counts[i] < (median - spread))]
    mask = np.zeros(labels.shape)
    mask[[i for i, x in enumerate(labels) if (x in values[target_values_indeces] and x != unwanted_label)]] = 1
    pcd_filtered = filter_pcd_from_mask(pcd, mask)
    return pcd_filtered
# ------------------------------------------------------------------------------


def dbscan_based_on_distance(pcd: o3d.geometry.PointCloud, epsilon: float = 0.01,
                             min_points: int = 25, number_of_clusters: int = 3) -> o3d.geometry.PointCloud:
    """
    Receive a point cloud, a neighborhood size (epsilon), number of points, and number of clusters
    and return the number_of_clusters largest clusters found using dbscan for the point positions
    with the relevant parameters.
    :param pcd: Original point cloud to be segmented.
    :param epsilon: Epsilon for the dbscan.
    :param min_points: Minimum number of points for a cluster to be considered such.
    :param number_of_clusters: Number of clusters to keep.
    :return:
    Return the filtered point cloud.
    """
    labels = np.array(pcd.cluster_dbscan(epsilon, min_points, True))
    pcd_filtered = filter_n_biggest_labels(pcd, labels, number_of_clusters, -1)
    return pcd_filtered
# ------------------------------------------------------------------------------


def remove_radius_outliers(pcd: o3d.geometry.PointCloud, min_points: int = 20,
                           radius: float = 0.05) -> o3d.geometry.PointCloud:
    """
    Receive a point cloud, a number of points and a radius and return a point cloud containing all the same
    points as the original one except those with less than the number of points in the specified radius.
    :param pcd: Original point cloud.
    :param min_points: Minimum number of points in the vicinity if a point is to be kept.
    :param radius: Radius of the neighborhood we are considering for every point.
    :return:
    Return the filtered point cloud.
    """
    new_pcd, outliers = pcd.remove_radius_outlier(min_points, radius)
    return new_pcd
# ------------------------------------------------------------------------------


def dbscan_based_on_color_and_distance(pcd: o3d.geometry.PointCloud, color_weight: float = 0.7,
                                       distance_weight: float = 1, epsilon: float = 0.08, min_points: int = 20,
                                       number_of_clusters: int = 3, pca_num: int = 0) -> o3d.geometry.PointCloud:
    """
    Receive a point cloud, weight values for colors and coordinates, an epsilon, a min number of points a number
    of clusters and a potential number to do prime component analysis and perform dbscan based on the color and
    position values of the point cloud.
    :param pcd: Original point cloud.
    :param color_weight: Weight given to the colors.
    :param distance_weight: Weight given to the coordinates.
    :param epsilon: Epsilon value for the dbscan.
    :param min_points: Minimum number of points to qualify as a cluster.
    :param number_of_clusters: Number of clusters to keep.
    :param pca_num: Number of primary components if we decide to do pca.
    :return:
    Return the selected clusters.
    """
    normalized_R = np.array(scipy.stats.zscore(np.array(pcd.colors)[:, 0]))
    normalized_G = np.array(scipy.stats.zscore(np.array(pcd.colors)[:, 1]))
    normalized_B = np.array(scipy.stats.zscore(np.array(pcd.colors)[:, 2]))
    normalized_x = np.array(scipy.stats.zscore(np.array(pcd.points)[:, 0]))
    normalized_y = np.array(scipy.stats.zscore(np.array(pcd.points)[:, 1]))
    normalized_z = np.array(scipy.stats.zscore(np.array(pcd.points)[:, 2]))
    features = np.transpose(
        np.vstack((color_weight * normalized_R, color_weight * normalized_G, color_weight * normalized_B,
                   distance_weight * normalized_x, distance_weight * normalized_y, distance_weight * normalized_z)))
    if pca_num > 0:
        pca = PCA(n_components=pca_num)
        pca.fit(features)
        features = pca.transform(features)
    clustering = DBSCAN(eps=epsilon, min_samples=min_points).fit(features)
    labels = np.array(clustering.labels_)
    pcd_filtered = filter_n_biggest_labels(pcd, labels, number_of_clusters, -1)
    return pcd_filtered
# ------------------------------------------------------------------------------


def dbscan_based_on_hsv_color_and_distance(pcd: o3d.geometry.PointCloud, color_weight: float = 0.7,
                                           distance_weight: float = 1, epsilon: float = 0.05,
                                           min_points: int = 25, number_of_clusters: int = 3,
                                           pca_num: int = 0) -> o3d.geometry.PointCloud:
    """
    Receive a point cloud, weight values for colors and coordinates, an epsilon, a min number of points a number
    of clusters and a potential number of variables to do prime component analysis and perform dbscan based on the
    HSV color and position values of the point cloud.
    :param pcd: The point cloud.
    :param color_weight: Weight given to the colors.
    :param distance_weight: Weight given to the coordinates
    :param epsilon: Epsilon for the dbscan algorithm.
    :param min_points: Minimum number of points to qualify as a cluster.
    :param number_of_clusters: Number of clusters to return in the filtered point cloud.
    :param pca_num: Number of primary components if we decide to do pca.
    :return:
    Return the filtered point cloud with the specified number of clusters.
    """
    normalized_x = np.array(scipy.stats.zscore(np.array(pcd.points)[:, 0]))
    normalized_y = np.array(scipy.stats.zscore(np.array(pcd.points)[:, 1]))
    normalized_z = np.array(scipy.stats.zscore(np.array(pcd.points)[:, 2]))
    rgbcolors = pcd.colors
    hsvcolors = mcolors.rgb_to_hsv(rgbcolors)
    normalized_H = np.array(scipy.stats.zscore(np.array(hsvcolors)[:, 0]))
    normalized_S = np.array(scipy.stats.zscore(np.array(hsvcolors)[:, 1]))
    normalized_V = np.array(scipy.stats.zscore(np.array(hsvcolors)[:, 2]))
    features = np.transpose(np.vstack((color_weight * normalized_S, color_weight * normalized_V, color_weight * normalized_H, distance_weight * normalized_x, \
                                       distance_weight * normalized_y, distance_weight * normalized_z)))
    if pca_num > 0:
        pca = PCA(n_components=pca_num)
        pca.fit(features)
        features = pca.transform(features)
    clustering = DBSCAN(eps=epsilon, min_samples=min_points).fit(features)
    labels = np.array(clustering.labels_)
    pcd_filtered = filter_n_biggest_labels(pcd, labels, number_of_clusters, -1)
    return pcd_filtered
# ------------------------------------------------------------------------------


def kmeans_based_on_hsv_color_and_distance(pcd: o3d.geometry.PointCloud, color_weight: float = 0.7,
                                           distance_weight: float = 1, number_of_clusters: int = 3,
                                           number_to_filter: int = 3, pca_num: int = 0) -> None:
    """
    Receive a point cloud, weights, number of clusters and number of clusters to return and a number of variables to
    do prime component analysis in. Cluster the point cloud with k-means
    :param pcd: The point cloud to segment.
    :param color_weight: Weight to give to the colors.
    :param distance_weight: Weight to give to the position values.
    :param number_of_clusters: How many clusters we want k-means to find.
    :param number_to_filter: How many clusters we want to keep.
    :param pca_num: Number of primary components to use if we decide to use pca.
    :return:
    Return the number specified of the biggest clusters found from the original
    point cloud by using the k-means algorithm.
    """
    normalized_x = np.array(scipy.stats.zscore(np.array(pcd.points)[:, 0]))
    normalized_y = np.array(scipy.stats.zscore(np.array(pcd.points)[:, 1]))
    normalized_z = np.array(scipy.stats.zscore(np.array(pcd.points)[:, 2]))
    rgbcolors = pcd.colors
    hsvcolors = mcolors.rgb_to_hsv(rgbcolors)
    normalized_H = np.array(scipy.stats.zscore(np.array(hsvcolors)[:, 0]))
    normalized_S = np.array(scipy.stats.zscore(np.array(hsvcolors)[:, 1]))
    normalized_V = np.array(scipy.stats.zscore(np.array(hsvcolors)[:, 2]))
    features = np.transpose(np.vstack((color_weight * normalized_S, color_weight * normalized_V, color_weight * normalized_H, distance_weight * normalized_x, \
                                       distance_weight * normalized_y, distance_weight * normalized_z)))
    if pca_num > 0:
        pca = PCA(n_components=pca_num)
        pca.fit(features)
        features = pca.transform(features)
    kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(features)
    labels = np.array(kmeans.labels_)
    pcd_filtered = filter_n_biggest_labels(pcd, labels, number_to_filter, -1)
    o3d.visualization.draw_geometries([pcd_filtered])
    return pcd_filtered
# ------------------------------------------------------------------------------

def hsv_h_thresholding(pcd: o3d.geometry.PointCloud, h_aim_1: float = 0.019, h_aim_2: float = 0.736,
                       h_aim_3: float = 0.327, h_error_1: float = 0.02,
                       h_error_2: float = 0.02, h_error_3: float = 0.02) -> o3d.geometry.PointCloud:
    """
    Receive a point cloud, and 3 target values for the Hue in HSV color scheme and 3 error values.
    :param pcd:
    :param h_aim_1:
    :param h_aim_2:
    :param h_aim_3:
    :param h_error_1:
    :param h_error_2:
    :param h_error_3:
    :return:
    Return a point cloud containing only those points whose colors lie within the specified error distance from the
    target values.
    """
    mask1 = get_mask_by_hue_hsv(pcd, h_aim=h_aim_1, h_error=h_error_1)
    mask2 = get_mask_by_hue_hsv(pcd, h_aim=h_aim_2, h_error=h_error_2)
    mask3 = get_mask_by_hue_hsv(pcd, h_aim=h_aim_3, h_error=h_error_3)
    mask = mask1 | mask2 | mask3
    pcd_filtered = filter_pcd_from_mask(pcd, mask)
    return pcd_filtered
# ------------------------------------------------------------------------------


def dbscan_based_on_normals_and_distance(pcd: o3d.geometry.PointCloud, normal_weight: float = 0.7,
                                        distance_weight:float = 1, epsilon: float=0.1, min_points: int = 7,
                                        number_of_clusters: int = 3, pca_num: int = 0) -> o3d.geometry.PointCloud:
    """
    Receive a point cloud, weight values for normals and coordinates, an epsilon, a min number of points a number
    of clusters and a potential number to do prime component analysis and perform dbscan based on the normals and
    position values of the point cloud.
    :param pcd: The point cloud to be filtered
    :param normal_weight: The weight to be given to the normals.
    :param distance_weight: The weight to be given to the coordinates.
    :param epsilon: The epsilon value for the dbscan.
    :param min_points: The minimum number of points to start a cluster.
    :param number_of_clusters: The number of clusters to keep.
    :param pca_num: The number of primary components to consider if we do pca (0 if we don't do pca).
    :return:
    Return the filtered point cloud containing just the clusters.
    """
    normalized_nx = np.array(scipy.stats.zscore(np.array(pcd.normals)[:, 0]))
    normalized_ny = np.array(scipy.stats.zscore(np.array(pcd.normals)[:, 1]))
    normalized_nz = np.array(scipy.stats.zscore(np.array(pcd.normals)[:, 2]))
    normalized_x = np.array(scipy.stats.zscore(np.array(pcd.points)[:, 0]))
    normalized_y = np.array(scipy.stats.zscore(np.array(pcd.points)[:, 1]))
    normalized_z = np.array(scipy.stats.zscore(np.array(pcd.points)[:, 2]))
    features = np.transpose(
        np.vstack((normal_weight * normalized_nx, normal_weight * normalized_ny, normal_weight * normalized_nz,
                   distance_weight * normalized_x, distance_weight * normalized_y, distance_weight * normalized_z)))
    if pca_num > 0:
        pca = PCA(n_components=pca_num)
        pca.fit(features)
        features = pca.transform(features)
    clustering = DBSCAN(eps=epsilon, min_samples=min_points).fit(features)
    labels = np.array(clustering.labels_)
    pcd_filtered = filter_n_biggest_labels(pcd, labels, number_of_clusters, -1)
    return pcd_filtered
# ------------------------------------------------------------------------------


def projected_convex_hull(pcd: o3d.geometry.PointCloud, dim1: int = 0, dim2: int = 1) -> scipy.spatial.ConvexHull:
    """
    Receive a point cloud and two dimensions of choice and project that point cloud into the plane formed by those
    dimensions and calculate the convex hull of the projection.
    :param pcd: The point cloud.
    :param dim1: The first dimension.
    :param dim2: The second dimension.
    :return:
    Return the convex hull of the projection in 2d.
    """
    points = np.array(pcd.points)[:, (dim1, dim2)]
    convex_hull = scipy.spatial.ConvexHull(points)
    return convex_hull
# ------------------------------------------------------------------------------


def compute_distance_between_points(pcd: o3d.geometry.PointCloud, index1: int, index2: int):
    """
    Receive a point cloud and the indexes for two of its points and compute the
    Euclidean distance between the two points.
    :param pcd: The point cloud.
    :param index1: The index of the first point.
    :param index2: The index of the second point.
    :return:
    Return the distance.
    """
    points = np.asarray(pcd.points)
    distance_vector = (points[index1, :] - points[index2, :])
    distance = np.sqrt(distance_vector.dot(distance_vector))
    return distance
# -----------------------------------------------------------------------------


def paint_pcd_based_on_normals(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """
    Receive a point cloud and reassign its colors based on the direction of the normals
    for the points.
    :param pcd: The point cloud to be painted.
    :return:
    Return the point cloud with the new colors.
    """
    normals = np.asarray(pcd.normals)
    colors = (normals[:, :] + 1) / 2
    # colors[:, 1] = 0
    print(colors)
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    return pcd
# -----------------------------------------------------------------------------


def tune_normal_directions(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """
    Receive a point cloud and tune all of its normals in the same orientation of the surface.
    :param pcd: The point cloud to be tuned.
    :return:
    Return the point cloud with the corrected normals.
    """
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    distance_matrix = distance.squareform(distance.pdist(points))
    nearest_neighbors = np.argsort(distance_matrix, axis=1)
    total_points = points.shape[0]
    tuned_point_indeces = np.zeros(total_points)
    current_point = 0
    next_point = nearest_neighbors[current_point, 1]
    num_points_tuned = 1


    while(num_points_tuned < total_points):
        i = 2
        while (next_point in tuned_point_indeces) and (i < total_points):
            next_point = nearest_neighbors[current_point, i]
            i += 1
        added_normals = normals[current_point, :] + normals[next_point, :]
        added_normals_norm = np.sqrt(added_normals.dot(added_normals))
        tuned_point_indeces[num_points_tuned] = next_point
        num_points_tuned += 1
        if added_normals_norm < math.sqrt(2):
            normals[next_point, :] *= -1
        current_point = next_point
        next_point = nearest_neighbors[current_point, 1]
    pcd.normals = o3d.utility.Vector3dVector(normals[:, :3])
    return pcd
# -----------------------------------------------------------------------------


def tune_normal_directions_modified(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """
    Receive a point cloud and tune all of its normals in the same orientation of the surface.
    :param pcd: The point cloud to be tuned.
    :return:
    Return the point cloud with the corrected normals.
    """
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals)
    distance_matrix = distance.squareform(distance.pdist(points))
    nearest_neighbors = np.argsort(distance_matrix, axis=1)
    total_points = points.shape[0]
    tuned_point_indeces = np.zeros(total_points)
    current_point = 0
    next_point = nearest_neighbors[current_point, 1]
    num_points_tuned = 1


    while(num_points_tuned < total_points):
        i = 2
        while (next_point in tuned_point_indeces) and (i < total_points):
            next_point = nearest_neighbors[current_point, i]
            i += 1
        added_normals = normals[current_point, :] + normals[next_point, :]
        added_normals_norm = np.sqrt(added_normals.dot(added_normals))
        tuned_point_indeces[num_points_tuned] = next_point
        num_points_tuned += 1
        if added_normals_norm < math.sqrt(2):
            normals[next_point, :] *= -1
        # if (added_normals_norm < 0.75) or (added_normals_norm > 1.85):
        if (added_normals_norm < 0.9) or (added_normals_norm > 1.5):
            current_point = next_point
            next_point = nearest_neighbors[current_point, 1]
        else:
            next_point = nearest_neighbors[current_point, 1]
    pcd.normals = o3d.utility.Vector3dVector(normals[:, :3])
    return pcd
# -----------------------------------------------------------------------------

def downsample_2d(points, voxel_size):
    """
    Downsample 2D array of points
    """
    points_3d = np.c_[points, np.zeros(len(points))]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    points_3d = np.asarray(pcd.points)
    points = points_3d[:, 0:2]

    return points
# -----------------------------------------------------------------------------

def image_to_point_array(image_path: str) -> np.ndarray:
    """
    Receives the path to an image and returns the corresponding point array.
    For example from an image of the "ideal" leaf you can get array of points in that shape
    :param image_path: The path to the image.
    :return:
    Return the point array.
    """
    # Load the image as a numpy array and normalize its values to [0, 1]
    image = np.asarray(Image.open(image_path).convert("1"))
    img_points = np.where(image == False)
    img_points = np.dstack(img_points)[0]

    # Normalize points
    # scaler = MinMaxScaler((-0.5, 0.5))
    # img_points = scaler.fit_transform(img_points)
    # scaler = PointCloudWidthScaler(target_width=0.5)
    scaler = PointCloudScaler(target_distance=1.0)
    img_points = scaler.fit_transform(img_points)

    # Downsample points with pcd
    img_points = downsample_2d(img_points, voxel_size=0.03)
    return img_points
# -----------------------------------------------------------------------------

def plot_overlapping_2d_point_clouds(point_cloud_1, point_cloud_2):
    plt.scatter(point_cloud_1[:, 0], point_cloud_1[:, 1], color='red', marker='o', label='image points')
    plt.scatter(point_cloud_2[:, 0], point_cloud_2[:, 1], color='blue', marker='x', label='data points')
    plt.axis('equal')
    plt.title("Plot of fitting")
    plt.show()
    return

# =============================================================================
# Scaler class to normalize the point cloud. We will have two separate classes for this.
# The classes are PointCloudScaler and PointCloudWidthScaler. The first one is what we use
# on the point cloud that we flattened using isomap. It will scale the largest distance from
# the center (mean) of the point cloud to one of its points to be 0.5. The second one is what
# we will use on the point cloud created from the picture of the blueprint (the target for)
# fitting. This will turn the width of the point cloud (the leaf) to be 0.5.

class PointCloudScaler(BaseEstimator, TransformerMixin):
    def __init__(self, target_distance=1.0):
        self.scaling_factor = None
        self.mean_position = None
        self.center_position = None
        self.target_distance = target_distance

    # def fit(self, X):
    #     self.mean_position = np.mean(X, axis=0)
    #     centered_points = X - self.mean_position
    #     distances = np.linalg.norm(centered_points, axis=1)
    #     max_distance = np.max(distances)
    #     self.scaling_factor = self.target_distance / max_distance
    #     return self

    def fit(self, X):
        # Calculate the bounding box
        min_coords = np.min(X, axis=0)
        max_coords = np.max(X, axis=0)

        # Find the geometric center of the bounding box
        self.center_position = (max_coords + min_coords) / 2

        # Calculate the width of the bounding box (max x - min x)
        current_width = max_coords[0] - min_coords[0]

        # Determine the scaling factor based on the target width
        self.scaling_factor = self.target_distance / current_width
        return self

    def transform(self, X):
        # centered_points = X - self.mean_position
        centered_points = X - self.center_position
        scaled_points = centered_points * self.scaling_factor
        return scaled_points
# =============================================================================