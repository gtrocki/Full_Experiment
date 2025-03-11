# Full Experiment 

## Description
The objective of this project is to combine the analysis of a full experiment into a single well organized project with
streamlined code and clear instructions on how to run it. 

The project should be able to make a local copy of the point clouds of an experiment, align and scale them consistently,
segment the leaves, flatten the leaves and fit local coordinates on them, find the function that maps the leaves back
from 2d to 3d, and perform different types of measurements using this mapping.

## Installation

1. Clone the repository:WRITE THIS LATER
    
2. Install dependencies: 

    ```bash
   pip install -r requirements.txt
   ```

## Usage

Here is a list of the running order for the analysis of a full experiment.

### Copy reconstructions for the full experiment.

In this step, we copy the reconstructions (the 3d point clouds) of the relevant frames in the experiment
into a local folder for ease of analysis. In order to do this, go into the folder "copy_experiment_reconstructions", update the local config file with the correct paths and 
run the following file: 

```bash
copy_reconstructions.py
```

### Align and resize point clouds for global coordinates.

In this step we assign consistent global coordinates to all the frames of an experiment. We rotate the point clouds to
fit the rim of the arena in the xy plane and resize the radius to agree across frames. 

In order to do this, we need to set the correct values in the config file of the "align_and_resize_point_clouds" folder, 
and then run the file:

```bash
align_and_resize_point_clouds.py
```

### Segment and flatten leaves (and assign local coordinates).

In this step we isolate leaves from the rest of the point cloud, we flatten them using the isomap algorithm, 
and then fit them to the blueprint of the leaf in order to obtain consistent local coordinates throughout the leaves.

In order to do this, we need to set the correct limits for the segmentation trapezoids as well as the choose the 
correct files and modes in the config file of the "segment_and_flatten_leaves" folder, and make sure the correct 
leaf blueprints are located in the "blueprints" subfolder and referenced. Afterwards we can run the file:

```bash
full_experiment.py
```

Alternatively, we can run all the colors in a loop, by running the file 

```bash
all_colors.py
```

which really contains the same as "full_experiment" only looped over the relevant colors. Of course, before doing 
this the correct limits and options have to be set.

### Manually fit the 2d leaves that weren't fitted properly.

While the automatic point set registration algorithm does a very decent job
of fitting the leaves to the blueprint, there may be some frames that are not properly fitted. This normally occurs as 
a result of poor segmentation or detection by colmap. As the experiment progresses, some of the leaves may be occluded 
or their colors affected by shade, leading to an incomplete leaf reconstruction that gmmreg doesn't fit properly to 
the blueprint. In those cases, it is necessary to identify the offending frames and fit them by manually.

In order to do this, we first need to run option 3 in 

```bash
full_experiment.py
```

in the folder "segment_and_flatten_leaves" in order to identify the problematic frames. Then we run option 2 of the same
file on the same frames. And then we can run the file

```bash
transform_2d_point_cloud_interactive.py
```

if we want to transform leaves individually in an interactive fashion, or if we already know the transformation and 
want to apply it in bulk, we can run

```bash
transform_2d_point_cloud_bulk.py
```

Of course, we need to set the right parameters beforehand for either case.

### Fit 3d leaves (and possibly calculate values).

In this step we obtain the mapping from the 2d surface to the 3d one. We begin by creating an almost uniform point 
cloud in 2d with points that fall within the contour of the leaf. We add to that point cloud the points from the point 
cloud we got in the previous step, and we know what those should map to in 3d (namely the 3d point cloud). We obtain the
mapping of the rest of the points by interpolation, finding the locations that minimize the elastic energy.

In order to do this, we need to set the correct running parameters (folder paths for the point clouds, blueprint path, 
range of frames) in the config file of the "fit_3d_leaves" folder, and make sure that we set up the modes we want, then
run the file:

```bash
full_experiment_fit_3d.py
```

Alternatively, we can run all the colors in a loop by running the file:

```bash
all_colors_fit_3d.py
```

which really contains the same as "full_experiment_fit_3d" only looped over the relevant colors. Of course, before doing 
this the correct limits and options have to be set.

