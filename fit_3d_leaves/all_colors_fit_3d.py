"""
This file is a variation on the file full_experiment_fit_3d.py. It is intended to do the
same thing but for all colors in a loop.
Make sure to set all the parameters properly as is expected in run_full_experiment, and also that the color cases
chosen correspond to the colors of the leaves used in the actual experiment as per explained in config.py.
"""
import config
import full_experiment_fit_3d

if __name__ == "__main__":

    base_path = config.BASE_PATH
    number_of_frames = config.NUMBER_OF_FRAMES
    offset = config.INITIAL_FRAME
    leaf_svg_path = config.LEAF_SVG_PATH

    for color_case in [0, 1, 2, 4]:
        if color_case == 0:
            # Values for red leaf.
            full_experiment_fit_3d.run_full_experiment(config.BASE_PATH, offset, number_of_frames, result_option=1,
                                                       color='red', leaf_svg_path=leaf_svg_path, scaling=1)
        elif color_case == 1:
            # Values for pink leaf.
            full_experiment_fit_3d.run_full_experiment(config.BASE_PATH, offset, number_of_frames,  result_option=1,
                                                       color='pink', leaf_svg_path=leaf_svg_path, scaling=1)
        elif color_case == 2:
            # Values for blue leaf.
            full_experiment_fit_3d.run_full_experiment(config.BASE_PATH, offset, number_of_frames, result_option=1,
                                                       color='blue', leaf_svg_path=leaf_svg_path, scaling=1)
        elif color_case == 3:
            # Values for green leaf.
            full_experiment_fit_3d.run_full_experiment(config.BASE_PATH, offset, number_of_frames, result_option=1,
                                                       color='green', leaf_svg_path=leaf_svg_path, scaling=1)
        elif color_case == 4:
            # Values for turquoise leaf.
            full_experiment_fit_3d.run_full_experiment(config.BASE_PATH, offset, number_of_frames, result_option=1,
                                                       color='turquoise', leaf_svg_path=leaf_svg_path, scaling=1)
