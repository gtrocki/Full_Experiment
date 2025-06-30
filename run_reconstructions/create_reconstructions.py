import subprocess
import os.path
from os import path

# --------------------------------------------------------
# UPDATE WITH NEEDED PATH HERE
original_workspace_path = os.path.join("Y:", "COLMAP_reconstructions", "2025-04-30-14-40-16")
original_image_path = os.path.join("Y:", "results", "2025-04-30-14-40-16")
# --------------------------------------------------------

i = 1
final_frame = 3

workspace_path = os.path.join(original_workspace_path, "Session_" + str(i))
image_path = os.path.join(original_image_path, "Session_" + str(i))

# --------------------------------------------------------
# UPDATE WITH NEEDED PATH HERE
script_path = os.path.join("Y:", "results", "Gadi_Tests", "Automated_tests", "15th_colony", "final_structure_120421",
                           "colmap_script_for_python.bat")
# --------------------------------------------------------

while (path.exists(image_path)):

    # call the batch file to execute colmap passing the parameters for the paths
    p = subprocess.Popen([script_path, original_workspace_path, image_path, workspace_path],
                         creationflags=subprocess.CREATE_NEW_CONSOLE)
    # wait for the process (batch file) to terminate before moving to the next iteration.
    p.wait()

    # update variables of path and dummy variable for next loop iteration.
    i += 1

    workspace_path = os.path.join(original_workspace_path, "Session_" + str(i))
    image_path = os.path.join(original_image_path, "Session_" + str(i))
    # To have only the first 25 sessions
    if i > final_frame:
        break
