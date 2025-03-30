import os

COLOR_CASE = 0 # options are: 0 red, 1 pink, 2 blue, 3 green

if COLOR_CASE == 0:
    COLOR = "red"
    BLUEPRINT_COLOR = "red"
elif COLOR_CASE == 1:
    COLOR = "pink"
    BLUEPRINT_COLOR = "pink"
elif COLOR_CASE == 2:
    COLOR = "blue"
    BLUEPRINT_COLOR = "green"
elif COLOR_CASE == 3:
    COLOR = "green"
    BLUEPRINT_COLOR = "green"
elif COLOR_CASE == 4:
    COLOR = "turquoise"
    BLUEPRINT_COLOR = "turquoise"

# The path of the file containing the 2d image that represents the leaf blueprint.
BLUEPRINT_PATH = os.path.join("../blueprints", BLUEPRINT_COLOR + "_leaf_simple.jpg")

LEAF_SVG_PATH = """
m 0,0 c 0,-82.369 -54.235,-151.428 -70.134,-169.903 -2.242,-2.605
-6.419,-2.605 -8.661,0 -15.899,18.475 -70.134,87.534
-70.134,169.903 0,75.198 45.202,139.302 65.171,163.952 4.697,5.798
13.89,5.798 18.587,0 C -45.202,139.302 0,75.198 0,0 Z
""".strip("\n").replace("\n", " ")

drive1 = "C:"
drive2 = "Y:"

# Files containing the point clouds to be fitted.
# BASE_PATH = os.path.join(drive1, "\\", "Users", "michalro", "PycharmProjects", "Full_Experiment", "data", "2024-05-29-11-18-42")
# BASE_PATH = os.path.join(drive1, "\\", "Users", "michalro", "PycharmProjects", "Full_Experiment", "data", "2024-06-25-14-58-28")
BASE_PATH = os.path.join(drive1, "\\", "Users", "michalro", "PycharmProjects", "Full_Experiment", "data", "2024-06-09-12-12-23")


SCALING_FACTOR = 0.6322986525350575
FACTOR = 0.01
FITTING_TO_ENERGY_RATIO = 10000
INITIAL_HOLDER_ANGLE = 0 # in radians


NUMBER_OF_FRAMES = 123
INITIAL_FRAME = 203

SHOW_PLOTS = True
# SHOW_PLOTS = False