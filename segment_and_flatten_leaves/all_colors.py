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
                [-0.46, -0.012],  # Point A
                [-0.187, -0.012],  # Point B
                [-0.198, -0.109],  # Point C
                [-0.46, -0.1]  # Point D
            ])
            blueprint_path = os.path.join("blueprints", "red" + "_leaf_simple.jpg")
            full_experiment.run_full_experiment(config.BASE_EXPERIMENT_PATH, 187, 531,
                                                color_range=trapezoid, extra_color=1, extra_color_range=(0, 0.68),
                                                result_option=1, blueprint_path=blueprint_path)
        elif color_case == 1:
            # Values for pink leaf.
            trapezoid = np.array([
                [-0.350, 0.106],  # Point A
                [-0.250, 0.194],  # Point B
                [-0.054, 0.031],  # Point C
                [-0.113, -0.011]  # Point D
            ])
            blueprint_path = os.path.join("blueprints", "pink" + "_leaf_simple.jpg")
            full_experiment.run_full_experiment(config.BASE_EXPERIMENT_PATH, 187, 531,
                                                color_range=trapezoid, extra_color=1, extra_color_range=(0, 0.68),
                                                result_option=1, blueprint_path=blueprint_path)
        elif color_case == 2:
            # Values for blue leaf.
            trapezoid = np.array([
                [-0.040, 0.02],  # Point A
                [-0.038, 0.215],  # Point B
                [0.170, 0.219],  # Point C
                [0.023, 0.038]  # Point D
            ])
            blueprint_path = os.path.join("blueprints", "blue" + "_leaf_simple.jpg")
            full_experiment.run_full_experiment(config.BASE_EXPERIMENT_PATH, 187, 531,
                                                color_range=trapezoid, extra_color=1, extra_color_range=(0, 0.68),
                                                result_option=1, blueprint_path=blueprint_path)
        elif color_case == 3:
            # Values for green leaf.
            trapezoid = np.array([
                [-0.057, 0.02],  # Point A
                [-0.032, 0.174],  # Point B
                [0.164, 0.174],  # Point C
                [0.02, 0.038]  # Point D
            ])
            blueprint_path = os.path.join("blueprints", "green" + "_leaf_simple.jpg")
            full_experiment.run_full_experiment(config.BASE_EXPERIMENT_PATH, 187, 531,
                                                color_range=trapezoid, extra_color=0, extra_color_range=(0, 0.5),
                                                result_option=1, blueprint_path=blueprint_path)
        elif color_case == 4:
            # Values for turquoise leaf.
            trapezoid = np.array([
                [0.02, 0.02],  # Point A
                [0.230, 0.170],  # Point B
                [0.320, -0.040],  # Point C
                [0.027, -0.061]  # Point D
            ])
            blueprint_path = os.path.join("blueprints", "turquoise" + "_leaf_simple.jpg")
            full_experiment.run_full_experiment(config.BASE_EXPERIMENT_PATH, 187, 531,
                                                color_range=trapezoid, extra_color=1, extra_color_range=(0, 0.68),
                                                result_option=1, blueprint_path=blueprint_path)
