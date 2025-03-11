import matplotlib.pyplot as plt
import os
from transform_2d_point_cloud_interactive import (read_point_cloud, mirror_point_cloud, scale_point_cloud,
                                                  translate_point_cloud, rotate_point_cloud, save_point_cloud)

def plot_point_cloud(points, title):
    """Plots the point cloud for visualization."""
    plt.scatter(points[:, 0], points[:, 1])
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis('equal')
    plt.grid(True)
    plt.show()


def main(file_path, output_path, operation, modifier=None):
    points = read_point_cloud(file_path)

    if operation == "mirror":
        transformed_points = mirror_point_cloud(points)
        # plot_point_cloud(transformed_points, "Mirrored Point Cloud")

    elif operation == "rotate":
        transformed_points = rotate_point_cloud(points, modifier)
        # plot_point_cloud(transformed_points, f"Rotated Point Cloud by {modifier} Degrees")

    elif operation == "scale":
        transformed_points = scale_point_cloud(points, modifier)
        # plot_point_cloud(transformed_points, f"Scaled Point Cloud by factor of {modifier}")

    elif operation == "translate":
        transformed_points = translate_point_cloud(points, modifier)
        # plot_point_cloud(transformed_points, f"Translated Point Cloud by {modifier}")
    else:
        print("Invalid operation.")
        transformed_points = points

    save_point_cloud(transformed_points, output_path)
    print(f"Transformed point cloud saved to {output_path}")


if __name__ == "__main__":
    # Specify the input CSV file path
    drive = "C:"
    base_path = os.path.join(drive, "\\", "Users", "michalro", "PycharmProjects", "Full_Experiment", "data",
                             "2024-06-25-14-58-28")

    # Arrangement for full experiment
    for i in range(717, 718):
        input_file = os.path.join(base_path, "Session_" + str(i), "pcd_2d_turquoise.csv")

        # Specify the output CSV file path
        output_file = os.path.join(base_path, "Session_" + str(i), "pcd_2d_turquoise.csv")

        # Choose the operation: "mirror" or "rotate" or "scale" or "translate"
        operation_choice1 = "scale"
        operation_choice2 = "mirror"
        operation_choice3 = "translate"

        # Specify the modifier if the chosen operation requires one.
        # e.g.: rotation angle, translation vector, or scaling factor.
        modifier1 = 0.96
        modifier2 = None #[0.01, 0.02]
        modifier3 = [0, 0.01]


        # main(input_file, output_file, operation_choice1, modifier=modifier1)
        # main(input_file, output_file, operation_choice2, modifier=modifier2)
        main(input_file, output_file, operation_choice3, modifier=modifier3)