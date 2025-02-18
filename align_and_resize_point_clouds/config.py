import os
import numpy as np

# define base path containing the point clouds.
drive1 = "Y:"
drive2 = "C:"
base_path = os.path.join(drive2, "\\", "Users", "michalro", "PycharmProjects", "Full_Experiment", "data", "2024-06-25-14-58-28")

# Define the cropping box.
min_bound = np.array([-2.0, -2.0, 0.0])
max_bound = np.array([2.0, 3.0, 3.5])

# Define the size we want the radius of the red circle to have once we normalize the size of the point cloud.
red_circle_radius_scale = 1.5

SHOW_ALL_PLOTS = False

number_of_frames = 529
starting_frame = 188