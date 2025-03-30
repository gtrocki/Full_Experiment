import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.widgets import Button, TextBox
from general_functions.point_cloud_manipulations import image_to_point_array


def read_point_cloud(file_path):
    return np.loadtxt(file_path, delimiter=',', skiprows=1)
# -----------------------------------------------------------------------------
def mirror_point_cloud(points):
    return np.array([points[:, 0], -points[:, 1]]).T
# -----------------------------------------------------------------------------

def scale_point_cloud(points, scale_factor):
    """Scales the point cloud around its center by the specified scale factor."""
    # Calculate the center of the point cloud
    center = np.mean(points, axis=0)

    # Translate the point cloud to the origin
    translated_points = points - center

    # Define the scaling matrix
    scaling_matrix = np.array([
        [scale_factor, 0],
        [0, scale_factor]
    ])

    # Apply the scaling
    scaled_points = np.dot(translated_points, scaling_matrix)

    # Translate back to the original center
    scaled_points += center

    return scaled_points
# -----------------------------------------------------------------------------

def translate_point_cloud(points, translation_vector):
    """Translates the point cloud by the specified translation vector [dx, dy]."""
    # Ensure the translation vector is in the correct format
    translation_vector = np.array(translation_vector)

    # Translate each point by the translation vector
    translated_points = points + translation_vector

    return translated_points
# -----------------------------------------------------------------------------

def rotate_point_cloud(points, angle):
    """Rotates the point cloud around its bounding box center by the specified angle in degrees."""
    # Convert angle from degrees to radians
    radians = np.radians(angle)

    # Calculate the bounding box center
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    center = (max_coords + min_coords) / 2

    # Translate the point cloud to the origin
    translated_points = points - center

    # Define the general rotation matrix
    rotation_matrix = np.array([
        [np.cos(radians), np.sin(radians)],
        [-np.sin(radians), np.cos(radians)]
    ])

    # Apply the rotation
    rotated_points = np.dot(translated_points, rotation_matrix)

    # Translate back to the original center
    rotated_points += center

    return rotated_points
# -----------------------------------------------------------------------------

def save_point_cloud(points, output_path):
    """Saves the transformed point cloud to a CSV file."""
    np.savetxt(output_path, points, delimiter=',', header='x,y', comments='')
# -----------------------------------------------------------------------------

# def plot_point_cloud(ax, original, transformed, title):
#     ax.clear()
#     ax.scatter(original[:, 0], original[:, 1], color='blue', label='Original', alpha=0.5)
#     ax.scatter(transformed[:, 0], transformed[:, 1], color='red', label='Transformed', alpha=0.7)
#     ax.set_title(title)
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.axis('equal')
#     ax.legend()
#     ax.grid(True)
#     plt.draw()

def plot_point_cloud(ax, original, transformed, title):
    ax.clear()
    ax.scatter(original[:, 0], original[:, 1], color='red', marker='o', label='image points')
    ax.scatter(transformed[:, 0], transformed[:, 1], color='blue', marker='x', label='data points')
    ax.set_title(title)
    ax.axis('equal')
    ax.legend()
    plt.draw()

# -----------------------------------------------------------------------------

def interactive_transform(file_path, output_path, blueprint_path):
    points = read_point_cloud(file_path)
    transformed_points = points.copy()
    points = image_to_point_array(blueprint_path)

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(left=0.3, bottom=0.25)

    plot_point_cloud(ax, points, transformed_points, "Interactive Point Cloud Transformer")

    def update():
        plot_point_cloud(ax, points, transformed_points, "Interactive Point Cloud Transformer")

    def on_mirror(event):
        nonlocal transformed_points
        transformed_points = mirror_point_cloud(transformed_points)
        update()

    def on_scale(event):
        nonlocal transformed_points
        try:
            scale_factor = float(scale_text_box.text)
            transformed_points = scale_point_cloud(transformed_points, scale_factor)
            update()
        except ValueError:
            print("Invalid scale factor")

    def on_translate(event):
        nonlocal transformed_points
        try:
            dx, dy = map(float, translate_text_box.text.split(','))
            transformed_points = translate_point_cloud(transformed_points, [dx, dy])
            update()
        except ValueError:
            print("Invalid translation values")

    def on_rotate(event):
        nonlocal transformed_points
        try:
            angle = float(rotate_text_box.text)
            transformed_points = rotate_point_cloud(transformed_points, angle)
            update()
        except ValueError:
            print("Invalid rotation angle")

    def on_save(event):
        save_point_cloud(transformed_points, output_path)
        print(f"Saved to {output_path}")

    mirror_button_ax = plt.axes([0.05, 0.7, 0.15, 0.05])
    mirror_button = Button(mirror_button_ax, 'Mirror')
    mirror_button.on_clicked(on_mirror)

    save_button_ax = plt.axes([0.05, 0.05, 0.15, 0.05])
    save_button = Button(save_button_ax, 'Save')
    save_button.on_clicked(on_save)

    scale_text_box_ax = plt.axes([0.05, 0.6, 0.1, 0.05])
    scale_text_box = TextBox(scale_text_box_ax, 'Scale', initial="1.0")
    scale_button_ax = plt.axes([0.16, 0.6, 0.1, 0.05])
    scale_button = Button(scale_button_ax, 'Apply')
    scale_button.on_clicked(on_scale)

    translate_text_box_ax = plt.axes([0.05, 0.45, 0.1, 0.05])
    translate_text_box = TextBox(translate_text_box_ax, 'Translate (x,y)', initial="0,0")
    translate_button_ax = plt.axes([0.16, 0.45, 0.1, 0.05])
    translate_button = Button(translate_button_ax, 'Apply')
    translate_button.on_clicked(on_translate)

    rotate_text_box_ax = plt.axes([0.05, 0.3, 0.1, 0.05])
    rotate_text_box = TextBox(rotate_text_box_ax, 'Rotate (°)', initial="0")
    rotate_button_ax = plt.axes([0.16, 0.3, 0.1, 0.05])
    rotate_button = Button(rotate_button_ax, 'Apply')
    rotate_button.on_clicked(on_rotate)

    plt.show()
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    drive = "C:"
    base_path= os.path.join(drive, "\\", "Users", "michalro", "PycharmProjects", "Full_Experiment", "data",
                             "2024-06-25-14-58-28")

    blueprint_path = os.path.join("../blueprints", "turquoise" + "_leaf_simple.jpg")

    input_file = os.path.join(base_path, "Session_710", "pcd_2d_turquoise.csv")
    output_file = os.path.join(base_path, "Session_710", "pcd_2d_turquoise.csv")

    interactive_transform(input_file, output_file, blueprint_path)