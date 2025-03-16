import os

# File for configuring the runtime parameters and options of the program.
USE_ISOMAP = 1
USE_LLE = 2

COLOR_CASE = 4 # options are: 0 red, 1 pink, 2 blue, 3 green

if COLOR_CASE == 0:
    COLOR = "red"
    BLUEPRINT_COLOR = "red"
elif COLOR_CASE == 1:
    COLOR = "pink"
    BLUEPRINT_COLOR = "pink"
elif COLOR_CASE == 2:
    COLOR = "blue"
    BLUEPRINT_COLOR = "blue"
elif COLOR_CASE == 3:
    COLOR = "green"
    BLUEPRINT_COLOR = "green"
elif COLOR_CASE == 4:
    COLOR = "turquoise"
    BLUEPRINT_COLOR = "turquoise"

# Configuration constants
SHOW_ALL_PLOTS = False # Toggle whether or not to show the plots when we run the program.

CROPPING_MIN_BOUND = [-1, -1, 0.3] #[-1.5, -1.5, -2.5] # We specify a box for cropping, where these are the min bounds of the box.
CROPPING_MAX_BOUND = [1, 1, 2] #[1.5, 1.5, 1]

# The base path where the reconstructions of the experiment are saved.
drive1 = "Y:"
drive2 = "C:"
BASE_EXPERIMENT_PATH = os.path.join(drive2, "\\", "Users", "michalro", "PycharmProjects", "Full_Experiment", "data", "2024-05-29-11-18-42")

# The path of the file containing the 2d image that represents the leaf blueprint.
BLUEPRINT_PATH = os.path.join("../blueprints", BLUEPRINT_COLOR + "_leaf_simple.jpg")

FIT_TYPE = USE_ISOMAP # Choose the tipe of fitting algorithm between isomap and locally linear embedding. Options of values
# are: IsomapFit.USE_ISOMAP or IsomapFit.USE_LLE.

GMMREG_FITTING = True # Toggle whether we use gmmreg fitting (which is slow) or simply use the normalization and leave it like that.
GMMREG_IC_PERIOD = 3 # Defines how often to do the full round of GMMREG for all possible initial conditions.
SCORING_METHOD = 1 # the method for scoring the agreement of the gmmreg fits. 0 is for nearest neighbors, and 1 for iou (intersection over union).
