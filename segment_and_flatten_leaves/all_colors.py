"""
This file is a variation on the file full_experiment.py. It is intended to do the
same thing but for all colors in a loop.
Make sure to set all the parameters properly as is expected in run_full_experiment, and also that the color cases
chosen correspond to the colors of the leaves used in the actual experiment as per explained in config.py.
"""
import config
import full_experiment
import numpy as np
import os

if __name__ == "__main__":

    for color_case in [0, 1, 2, 4]:
        if color_case == 0:
            # Values for red leaf.
            trapezoid = np.array([
                [-0.348, -0.099],  # Point A
                [-0.148, -0.028],  # Point B
                [-0.121, -0.122],  # Point C
                [-0.348, -0.2]  # Point D
            ])
            blueprint_path = os.path.join("../blueprints", "red" + "_leaf_simple.jpg")
            full_experiment.run_full_experiment(config.BASE_EXPERIMENT_PATH, 1, 2,
                                                color_range=trapezoid, extra_color=1, extra_color_range=(0, 0.68),
                                                result_option=3, color="red", blueprint_path=blueprint_path)
        elif color_case == 1:
            # Values for pink leaf.
            trapezoid = np.array([
                [-0.086, -0.009],  # Point A
                [-0.238, 0.124],  # Point B
                [-0.160, 0.168],  # Point C
                [-0.038, 0.034]  # Point D
            ])
            blueprint_path = os.path.join("../blueprints", "pink" + "_leaf_simple.jpg")
            full_experiment.run_full_experiment(config.BASE_EXPERIMENT_PATH, 1, 2,
                                                color_range=trapezoid, extra_color=1, extra_color_range=(0, 0.68),
                                                result_option=3, color="pink", blueprint_path=blueprint_path)
        elif color_case == 2:
            # Values for blue leaf.
            trapezoid = np.array([
                [-0.020, 0.030],  # Point A
                [0.053, 0.167],  # Point B
                [0.201, 0.164],  # Point C
                [0.062, 0.012]  # Point D
            ])
            blueprint_path = os.path.join("../blueprints", "blue" + "_leaf_simple.jpg")
            full_experiment.run_full_experiment(config.BASE_EXPERIMENT_PATH, 1, 2,
                                                color_range=trapezoid, extra_color=1, extra_color_range=(0, 0.68),
                                                result_option=3, color="blue", blueprint_path=blueprint_path)
        elif color_case == 3:
            # Values for green leaf.
            trapezoid = np.array([
                [-0.057, 0.02],  # Point A
                [-0.032, 0.174],  # Point B
                [0.164, 0.174],  # Point C
                [0.02, 0.038]  # Point D
            ])
            blueprint_path = os.path.join("../blueprints", "green" + "_leaf_simple.jpg")
            full_experiment.run_full_experiment(config.BASE_EXPERIMENT_PATH, 2, 1,
                                                color_range=trapezoid, extra_color=0, extra_color_range=(0, 0.5),
                                                result_option=3, color="green", blueprint_path=blueprint_path)
        elif color_case == 4:
            # Values for turquoise leaf.
            trapezoid = np.array([
                [0.038, -0.002],  # Point A
                [0.141, 0.022],  # Point B
                [0.167, 0. - 60],  # Point C
                [0.028, -0.061]  # Point D
            ])
            blueprint_path = os.path.join("../blueprints", "turquoise" + "_leaf_simple.jpg")
            full_experiment.run_full_experiment(config.BASE_EXPERIMENT_PATH, 1, 2,
                                                color_range=trapezoid, extra_color=1, extra_color_range=(0, 0.68),
                                                result_option=3, color="turquoise", blueprint_path=blueprint_path)
