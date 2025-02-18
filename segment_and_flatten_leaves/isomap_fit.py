import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from scipy.spatial import KDTree
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.neighbors import NearestNeighbors
from sklearn.base import BaseEstimator, TransformerMixin
import cv2
from PIL import Image
import gmmreg
import config
import math
from scipy.spatial.distance import pdist


# =============================================================================
def rot(angle):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the z-axis by the given angle in degrees.
    """
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    return R


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


def sort_hull_simplices(simplices: np.ndarray):
    """
    Receive an array of simplices (for a convex hull) of size n by 2
    and sort it such that the value at [n, 0] equals that at [n-1, 1].
    :param simplices: Thee n by 2 array of indices for the segments.
    :return:
    Return the sorted array of simplices.
    """
    shape = simplices.shape
    size = shape[0]
    new_simplices = np.zeros(shape=shape)
    new_simplices[0, 0] = simplices[0, 0]
    new_simplices[0, 1] = simplices[0, 1]
    simplices[0, 0] = -1
    simplices[0, 1] = -1
    for i in range(1, size):
        j = np.where(simplices == new_simplices[i - 1, 1])[0][0]
        k = np.where(simplices == new_simplices[i - 1, 1])[1][0]
        new_simplices[i, 0] = simplices[j, k].astype(int)
        if (k == 0):
            new_simplices[i, 1] = simplices[j, 1].astype(int)
        else:
            new_simplices[i, 1] = simplices[j, 0].astype(int)
        simplices[j, 0] = -1
        simplices[j, 1] = -1

    return new_simplices


def cartesian_to_polar_2d(points: np.ndarray) -> np.ndarray:
    """
    Receive a set of points in 2d Cartesian coordinates and convert them into polar coordinates.
    :param points: Set of points in the shape (n,2).
    :return:
    Return the set of points in the same order in polar coordinates.
    """
    x = points[:, 0]
    y = points[:, 1]

    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x)

    return np.column_stack((r, theta))


def sort_points_by_theta(points: np.ndarray) -> np.ndarray:
    """
    Get a set of points in 2d Cartesian coordinates and sort their indices by the polar (azimuthal) angle.
    :param points: Set of points in the shape (n,2).
    :return:
    Return the indices of the points sorted by polar order.
    """
    polar_points = cartesian_to_polar_2d(points)
    theta_values = polar_points[:, 1]
    sorted_indices = np.argsort(theta_values)
    return sorted_indices


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


class PointCloudWidthScaler(BaseEstimator, TransformerMixin):
    def __init__(self, target_width=0.5):
        self.scaling_factor = None
        self.mean_position = None
        self.target_width = target_width

    def fit(self, X):
        # Calculate the mean position of the point cloud
        self.mean_position = np.mean(X, axis=0)

        # Center the points by subtracting the mean
        centered_points = X - self.mean_position

        # Calculate the extent (range) of the point cloud along each axis
        min_values = np.min(centered_points, axis=0)
        max_values = np.max(centered_points, axis=0)
        extents = max_values - min_values

        # The width is the smallest extent
        current_width = np.min(extents)

        # Calculate the scaling factor to make the width equal to the target width
        self.scaling_factor = self.target_width / current_width
        return self

    def transform(self, X):
        # Center the points by subtracting the mean
        centered_points = X - self.mean_position

        # Scale the points
        scaled_points = centered_points * self.scaling_factor
        return scaled_points
# =============================================================================


class IsomapFit:
    USE_ISOMAP = config.USE_ISOMAP
    USE_LLE = config.USE_LLE
    """Class for encoding the generic process of fitting the point cloud to its 2d blueprint
    by flattening it with the Isomap algorithm."""

    def __init__(self, blueprint_path: str, n_components: int = 2,
                 n_neighbors: int = 50, mapping=None) -> None:
        # self.points = points
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.blueprint_path = blueprint_path
        if mapping is not None:
            self.mapping = mapping

    # --------------------------------------------------------------------------

    def use_isomap(self, points: np.ndarray) -> np.ndarray:
        """
        Receive a 3d point cloud and a number of components and number of
        nearest neighbors as parameters for isomap and perform an isomap
        dimensional reduction on the 3d points.
        :param points: Set of 3d points.
        :return:
        Return a numpy array of the 2d points data.
        """
        if self.mapping == self.USE_ISOMAP:
            # # Initialize and fit Isomap model
            data = lil_matrix(points)
            isomap = Isomap(n_components=self.n_components, n_neighbors=self.n_neighbors)
            data_2d = isomap.fit_transform(data)
            data_2d = np.array(data_2d)
            # points_2d = (data_2d + 2) * 250
            points_2d = data_2d
            return points_2d
        elif self.mapping == self.USE_LLE:
            params = {
                # "n_neighbors": self.n_neighbors,
                "n_neighbors": 5,
                "n_components": self.n_components,
                "eigen_solver": "auto",
                "random_state": 0,
                "n_jobs": -1
            }
            lle = LocallyLinearEmbedding(method="modified", **params)
            points_2d = lle.fit_transform(points)
            # points_2d = (points_2d + 1) * 1000 - 600
            return points_2d
        else:
            raise Exception("OOf")

    # --------------------------------------------------------------------------

    def turn_2d_point_cloud_to_image(self, points_2d: np.ndarray) -> np.ndarray:
        """
        Receive the 2d point cloud for the surface and turn it into an image.
        :param points_2d: Set of 2d points.
        :return:
        Return the image associated with the point cloud.
        """

        # Create an image with a white background
        image = np.ones((800, 800, 3), dtype=np.uint8) * 255

        # Convert the points to int type
        points = np.int32(points_2d)

        # Draw red pixels at the location of the points in the point cloud
        for point in points:
            cv2.circle(image, tuple(point), 1, (0, 0, 255), -1)

        # # Display the image
        # cv2.imshow("Image with point cloud", image)
        # # cv2.imwrite("2d_images/image_for_dima.jpg", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        return image

    # --------------------------------------------------------------------------

    def perform_closing_on_image(self, image: np.ndarray, kernel_size_close: int,
                                 kernel_size_open: int, kernel_size_close_2: int) -> np.ndarray:
        """
        Perform topological opening and closing on the image in order to connect
        the points as much as possible and get rid of the noise. The order of the
        operations is close, open, close.
        :param kernel_size_close: Size of the kernel for the first round of closing.
        :param kernel_size_open: Size of the kernel for the first round of opening.
        :param kernel_size_close_2: Size of the kernel for the second round of closing.
        :return:
        Return the image that results from these operations.
        """

        grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        (thresh, blackAndWhiteImage) = cv2.threshold(grayImage, 127, 255, cv2.THRESH_BINARY_INV)

        # Define the structuring element for the morphological operation
        kernel_close = np.ones((kernel_size_close, kernel_size_close), np.uint8)
        kernel_close_2 = np.ones((kernel_size_close_2, kernel_size_close_2), np.uint8)
        kernel_open = np.ones((kernel_size_open, kernel_size_open), np.uint8)

        # Perform closing
        closing = cv2.morphologyEx(blackAndWhiteImage, cv2.MORPH_CLOSE, kernel_close)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_open)
        closing_2 = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_close_2)

        return closing

    # --------------------------------------------------------------------------

    # def fit_to_blueprint(self, image1: np.ndarray) -> np.ndarray:
    #     """
    #     Receives the image of the point cloud and the path to the blueprint image
    #     and calculates and returns the affine transformation matrix that takes the
    #     former to the latter.
    #     :param image1: The image that we are trying to fit.
    #     :return:
    #     Return the affine transformation matrix.
    #     """
    #
    #     image2_path = self.blueprint_path
    #     # Load the two images
    #     img1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    #     img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    #
    #     if config.SHOW_ALL_PLOTS == True:
    #         cv2.imshow("Blueprint", img2)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()
    #
    #     # Find the keypoints and descriptors of the images using ORB (Oriented FAST and Rotated BRIEF)
    #     orb = cv2.ORB_create()
    #     kp1, des1 = orb.detectAndCompute(img1, None)
    #     kp2, des2 = orb.detectAndCompute(img2, None)
    #
    #     # Match the keypoints using a brute-force matcher
    #     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    #     matches = bf.match(des1, des2)
    #
    #     # Find the points in the first image that correspond to the points in the second image
    #     src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    #     dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    #
    #     # Find the affine transformation matrix using the RANSAC algorithm
    #     # M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    #     M, mask = cv2.estimateAffinePartial2D(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0)
    #
    #     if config.SHOW_ALL_PLOTS == True:
    #         # Apply the affine transformation to the first image
    #         img1_aligned = cv2.warpAffine(img1, M, (img2.shape[1], img2.shape[0]))
    #
    #         cv2.imshow("Image with point cloud", img1_aligned)
    #         cv2.waitKey(0)
    #         cv2.destroyAllWindows()
    #
    #     # Compare the overlap between the aligned first image and the second image
    #     # overlap = np.sum((img1_aligned == img2) & (img1_aligned == 255)) / np.sum(img1_aligned == 255)
    #
    #     # The affine transformation that leads to the greatest overlap is stored in the variable M
    #     return M

    # --------------------------------------------------------------------------

    def compute_iou_score(self, points1, points2, tolerance=0.01) -> float:
        """
        Function to compute the score of a transformation that is supposed to take
        one point cloud into another. We calculate the score as a function that reflects
        the intersection over the points divided by their union. Then, the larger the score
        is, the better, since it means more intersection and less union.
        :param points1: The first point cloud.
        :param points2: The second point cloud.
        :param tolerance: Threshold to consider the points as "intersecting".
        :return:
        Return the score.
        """
        # Build KD-trees for fast nearest neighbor search
        tree1 = KDTree(points1)
        tree2 = KDTree(points2)

        # Find nearest neighbors within the tolerance
        dist1, ind1 = tree1.query(points2, distance_upper_bound=tolerance)
        dist2, ind2 = tree2.query(points1, distance_upper_bound=tolerance)

        # Count the number of matched points
        num_matched_points = np.sum(dist1 < np.inf) + np.sum(dist2 < np.inf)

        # Compute the union and intersection sizes
        union_size = len(points1) + len(points2) - num_matched_points
        intersection_size = num_matched_points

        # Compute the IoU score
        iou_score = intersection_size / union_size

        return iou_score

    # --------------------------------------------------------------------------

    def image_to_point_array(self, image_path: str) -> np.ndarray:
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

    # --------------------------------------------------------------------------

    def fit_gmmreg(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Receives the source and target point arrays and fits the GMMREG algorithm
        :param source: The source point array.
        :param target: The target point array.
        :return:
        Return the transformed points.
        """
        np.savetxt(".source.txt", source)
        np.savetxt(".target.txt", target)

        _, _, after_transform = gmmreg.run_config("gmmreg.ini")
        return after_transform

    # --------------------------------------------------------------------------

    def fit_gmmreg_all_permutations(self, source: np.ndarray, target: np.ndarray) -> list[np.ndarray]:
        """
        Receives the source and target point arrays and fits the GMMREG algorithm with all possible 90 degree rotations
        and flips.
        :param source: The source point array.
        :param target: The target point array.
        :return:
        Return array of all possible transformed points, out of which you can select the best one.
        """
        # Normalize points around center (useful for rotations and flips)
        # scaler = MinMaxScaler((-0.5, 0.5))
        # source = scaler.fit_transform(source)
        # target = scaler.fit_transform(target)
        scaler = PointCloudScaler()
        # width_scaler = PointCloudWidthScaler(target_width=0.5)
        width_scaler = PointCloudScaler(target_distance=1.0)
        source = scaler.fit_transform(source)
        target = width_scaler.fit_transform(target)

        transformed = []
        # angle in [0,90,180,270]
        div = 90
        for angle in [div * i for i in range(360 // div)]:
            for flip in [False, True]:
                if flip:
                    source_flipped = source.dot(np.array([[1, 0], [0, -1]]).T)
                else:
                    source_flipped = source

                # Rotate
                source_flipped_rotated = source_flipped.dot(rot(angle))

                # New line might be problematic.
                source_flipped_rotated = scaler.fit_transform(source_flipped_rotated)

                after_transfrom = self.fit_gmmreg(source_flipped_rotated, target)
                transformed.append(after_transfrom)

        return transformed

    # --------------------------------------------------------------------------

    def fit_gmmreg_selected_ic(self, source: np.ndarray, target: np.ndarray, case_number: int = 0) -> np.ndarray:
        """
        Receives the source and target point arrays and fits the GMMREG algorithm the specified initial condition from
        within the 90 degree rotations plus reflections.
        :param source: The source point array.
        :param target: The target point array.
        :return:
        Return array of all possible transformed points, out of which you can select the best one.
        """

        scaler = PointCloudScaler()
        width_scaler = PointCloudScaler(target_distance=1.0)
        source = scaler.fit_transform(source)
        target = width_scaler.fit_transform(target)

        i = math.floor(case_number / 2)
        flip = case_number % 2
        div = 90
        angle = i * div
        if flip == 1:
            source = source.dot(np.array([[1, 0], [0, -1]]).T)

        # Rotate
        source = source.dot(rot(angle))
        transformed = self.fit_gmmreg(source, target)

        return transformed

    # --------------------------------------------------------------------------

    def fit_gmmreg_best(self, source: np.ndarray, target: np.ndarray, case_number: int = None) -> tuple[np.ndarray, list]:
        """
        Receives the source and target point arrays and fits the GMMREG algorithm with all possible 90 degree rotations
        and flips. Then finds the best transformation out of these and returns it
        :param source: The source point array.
        :param target: The target point array.
        :return:
        Return best mapped points from source to target, the scores of each of the initial conditions, and the number of
        the round with the best score.
        """
        if config.SHOW_ALL_PLOTS == True:
            plt.scatter(source[:, 0], source[:, 1])
            plt.title("fitting source")
            plt.axis('equal')
            plt.show()
            plt.scatter(target[:, 0], target[:, 1])
            plt.title("fitting target")
            plt.axis('equal')
            plt.show()

        if case_number == None:
            # Get all possible fits (i.e. rotation/flip + GMMREG fit)
            all_tranformations = self.fit_gmmreg_all_permutations(source, target)

            # Loop over all transformations and find the best one using the k-nearest neighbors
            scores = []
            i = 1
            best_score_index = None
            for transformation in all_tranformations:
                if config.SCORING_METHOD == 0:
                    # For each target point, find it's k nearest neighbours in the source points, this is what we minimize
                    nbrs = NearestNeighbors(n_neighbors=20, algorithm='ball_tree').fit(target)
                    distances, _ = nbrs.kneighbors(transformation)
                    score = np.sum(distances)
                    scores.append(score)
                    best_score_index = np.argmin(scores)
                else:
                    score = self.compute_iou_score(target, transformation,
                                                   tolerance=0.01)  # tolerance=0.005) #alternative scoring based on iou.
                    scores.append(score)
                    best_score_index = np.argmax(
                        scores)  # ---> when using iou, the best score is the highest to we use this.

                if config.SHOW_ALL_PLOTS == True:
                    plt.scatter(target[:, 0], target[:, 1], color='red', marker='o', label='image points')
                    plt.scatter(transformation[:, 0], transformation[:, 1], color='blue', marker='x',
                                label='data points')
                    plt.axis('equal')
                    plt.title(f"Transformation {i} -- Score = {score}")
                    plt.show()
                i += 1

            print(f"best score index = {best_score_index}")
            best_transformation = all_tranformations[best_score_index]
        else:
            scores = None
            best_score_index = None
            best_transformation = self.fit_gmmreg_selected_ic(source, target, case_number=case_number)

        return best_transformation, scores

    # --------------------------------------------------------------------------

    def affine_transform_2d_point_cloud(self, matrix: np.ndarray, points_2d: np.ndarray) -> np.ndarray:
        """
        Receive an affine transformation matrix and a set of 2d points and apply the
        transformation on the points.
        :param matrix: An affine matrix with translations.
        :param points_2d: The set of 2d points.
        :return:
        Return the set of points after the transformation.
        """
        matrix_linear = matrix[:, 0:2]
        matrix_linear_T = matrix_linear.T

        translation = matrix[:, 2]

        multiplication = np.matmul(points_2d, matrix_linear_T)
        transformed_points = multiplication + translation

        return transformed_points
    # --------------------------------------------------------------------------

    def flatten_scale_and_save(self, points: np.ndarray, target_path_2d: str, target_path_3d: str,
                               target_path_3d_original: str, gmmreg_case_number: int = None):
        """
        An alternative method to fit. It does the same process, but instead of fitting the point cloud with splines,
        it saves the normalized, 2d and 3d point clouds, so they can be fitted later by an energy minimization scheme.
        :param points: Set of 3d points.
        :return:
        Return the splines for each of the three coordinates.
        """
        # Use isomap to reduce the dimension of the point cloud.
        points_3d_original = points
        points_2d_original = self.use_isomap(points_3d_original)

        if config.SHOW_ALL_PLOTS == True:
            plt.scatter(points_2d_original[:, 0], points_2d_original[:, 1])
            plt.axis('equal')
            plt.title("Plot flattened point cloud")
            plt.show()

        # Normalize points
        # scaler = MinMaxScaler()
        scaler = PointCloudScaler()
        points_2d = scaler.fit_transform(points_2d_original)

        image = self.turn_2d_point_cloud_to_image(points_2d)
        image_with_closing = self.perform_closing_on_image(image, 12, 2, 15)

        # Fit to ideal image
        image_points = self.image_to_point_array(self.blueprint_path)
        if config.GMMREG_FITTING == True:
            transformed_points, scores = self.fit_gmmreg_best(points_2d, image_points, case_number=gmmreg_case_number)
            points_2d = transformed_points
        else:
            scaler = PointCloudScaler()
            transformed_points = scaler.fit_transform(points_2d)
            points_2d = transformed_points
            scores = np.array([])

        # Scale the 3d point cloud so it is the same size as the 2d one.
        distances_original = np.linalg.norm(points_2d_original, axis=1)
        max_distance_original = np.max(distances_original)
        distances = np.linalg.norm(points_2d, axis=1)
        max_distance = np.max(distances)
        scaling_factor = max_distance / max_distance_original

        points_3d = points_3d_original * scaling_factor

        # Save the 2d point cloud to a file.
        # Define the file path where you want to save the CSV file
        file_path_2d = target_path_2d
        file_path_3d = target_path_3d
        file_path_3d_original = target_path_3d_original
        # Save the point cloud as a CSV file
        np.savetxt(file_path_2d, points_2d, delimiter=',', header='X,Y', comments='')
        np.savetxt(file_path_3d, points_3d, delimiter=',', header='X,Y,Z', comments='')
        np.savetxt(file_path_3d_original, points_3d_original, delimiter=',', header='X,Y,Z', comments='')
        # --------------------------- display to check -----------------
        if config.SHOW_ALL_PLOTS == True:

            plt.scatter(image_points[:, 0], image_points[:, 1], color='red', marker='o', label='image points')
            plt.scatter(points_2d[:, 0], points_2d[:, 1], color='blue', marker='x', label='data points')
            plt.axis('equal')
            plt.title("Plot of fitting")
            plt.show()

            base = cv2.imread(config.BLUEPRINT_PATH, cv2.IMREAD_GRAYSCALE)

            # Convert the points to int type
            points = np.int32(transformed_points)
            image_points2 = np.int32(image_points)

            # Draw red pixels at the location of the points in the point cloud
            for point in points:
                cv2.circle(base, tuple(point), 1, (0, 0, 255), -1)
            for point in image_points2:
                cv2.circle(base, tuple(point), 1, (255, 0, 0), -1)

            # Display the image
            cv2.imshow("Image with point cloud", base)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # --------------------------- end display to check -----------------
        return scores

    # -------------------------------------------------------------------------

    def flatten_and_save(self, points: np.ndarray, target_path_2d: str, target_path_3d: str,
                         target_path_3d_original: str):
        """
        Flatten a 3d point cloud using isomap, and scale it so it is on a size comparable
        with that of the blueprint. Also scale the 3d point cloud by the same amount so that
        later, when fitting using energy minimization we don't get any stretching. Save the
        2d point cloud, the original segmented 3d point cloud and the scaled segmented 3d point
        cloud in the addresses provided.

        :return:

        """

        # Use isomap to reduce the dimension of the point cloud.
        points_3d_original = points
        points_2d_original = self.use_isomap(points_3d_original)

        if config.SHOW_ALL_PLOTS == True:
            plt.scatter(points_2d_original[:, 0], points_2d_original[:, 1])
            plt.axis('equal')
            plt.title("Plot flattened point cloud")
            plt.show()

        # Normalize points
        # scaler = MinMaxScaler()
        scaler = PointCloudScaler()
        points_2d = scaler.fit_transform(points_2d_original)

        # Fit to ideal image
        image_points = self.image_to_point_array(self.blueprint_path)

        # Scale the 3d point cloud so it is the same size as the 2d one.
        distances_original = np.linalg.norm(points_2d_original, axis=1)
        max_distance_original = np.max(distances_original)
        distances = np.linalg.norm(points_2d, axis=1)
        max_distance = np.max(distances)
        scaling_factor = max_distance / max_distance_original

        points_3d = points_3d_original * scaling_factor

        # Save the 2d point cloud to a file.
        # Define the file path where you want to save the CSV file
        file_path_2d = target_path_2d
        file_path_3d = target_path_3d
        file_path_3d_original = target_path_3d_original
        # Save the point cloud as a CSV file
        np.savetxt(file_path_2d, points_2d, delimiter=',', header='X,Y', comments='')
        np.savetxt(file_path_3d, points_3d, delimiter=',', header='X,Y,Z', comments='')
        np.savetxt(file_path_3d_original, points_3d_original, delimiter=',', header='X,Y,Z', comments='')
        # --------------------------- display to check -----------------
        if config.SHOW_ALL_PLOTS == True:

            plt.scatter(image_points[:, 0], image_points[:, 1], color='red', marker='o', label='image points')
            plt.scatter(points_2d[:, 0], points_2d[:, 1], color='blue', marker='x', label='data points')
            plt.axis('equal')
            plt.title("Plot of fitting")
            plt.show()

            base = cv2.imread(config.BLUEPRINT_PATH, cv2.IMREAD_GRAYSCALE)

            # Convert the points to int type
            points = np.int32(points_2d)
            image_points2 = np.int32(image_points)

            # Draw red pixels at the location of the points in the point cloud
            for point in points:
                cv2.circle(base, tuple(point), 1, (0, 0, 255), -1)
            for point in image_points2:
                cv2.circle(base, tuple(point), 1, (255, 0, 0), -1)

            # Display the image
            cv2.imshow("Image with point cloud", base)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # --------------------------- end display to check -----------------
        return

    # -------------------------------------------------------------------------

    def scale_3d_point_cloud(self, points_3d: np.ndarray, points_2d: np.ndarray, target_path_3d: str):
        """
        Flatten a 3d point cloud using isomap, compare its size with that of the already fitted 2d point cloud
        provided. Scale the 3d point cloud by the right factor so that it is of the same size as the 2d point cloud.
        This should work so later, when fitting using energy minimization we don't get any stretching. Save the
        2d point cloud, the original segmented 3d point cloud and the scaled segmented 3d point
        cloud in the addresses provided.

        :return:

        """

        # Use isomap to reduce the dimension of the point cloud.
        points_3d_original = points_3d
        points_2d_original = points_2d
        points_3d_flattened = self.use_isomap(points_3d_original)

        # Compute the pairwise distances for the original 2D point cloud
        pairwise_distances_flattened = pdist(points_3d_flattened)  # Flattened pairwise distances
        max_distance_original = np.max(pairwise_distances_flattened)  # Largest pairwise distance (diameter)

        # Compute the pairwise distances for the modified 2D point cloud
        pairwise_distances_target = pdist(points_2d_original)  # Flattened pairwise distances
        max_distance_target = np.max(pairwise_distances_target)  # Largest pairwise distance (diameter)

        # Compute the scaling factor based on the diameters
        scaling_factor = max_distance_target / max_distance_original

        # Scale the original 3D point cloud
        points_3d = points_3d_original * scaling_factor

        # Save the 3d point cloud to a file.
        # Define the file path where you want to save the CSV file
        file_path_3d = target_path_3d
        # Save the point cloud as a CSV file
        np.savetxt(file_path_3d, points_3d, delimiter=',', header='X,Y,Z', comments='')

        return scaling_factor

    # -------------------------------------------------------------------------

    def plot_of_fitting(self, points_2d: np.ndarray):
        """
        Plot the point cloud given as a parameter overlayed with the point cloud of the blueprint
        we are trying to fit in 2d to find local coordinates.
        :param points_2d: The point cloud to be plotted.
        """
        image_points = self.image_to_point_array(self.blueprint_path)
        plt.scatter(image_points[:, 0], image_points[:, 1], color='red', marker='o', label='image points')
        plt.scatter(points_2d[:, 0], points_2d[:, 1], color='blue', marker='x', label='data points')
        plt.axis('equal')
        plt.title("Plot of fitting")
        plt.show()

        return

# =============================================================================