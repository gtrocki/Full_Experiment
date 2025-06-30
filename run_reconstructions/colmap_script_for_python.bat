REM ECHO OFF
REM Set the desired shell variables containing the relevant paths for taking the images and executing the analysis.
SET ORIGINAL_WORKSPACE_PATH=%1
SET IMAGE_PATH=%2
SET WORKSPACE_PATH=%3

REM Change the directory to the directory where colmap is located in order to run it.
C:
cd C:\Users\michalro\Downloads\COLMAP-3.6-windows-cuda
dir

REM LIMIT THE SIZE OF THE BUFFER OF COMMAND INPUTS. (MIGHT SOLVE THE ISSUE I'M HAVING)
doskey /listsize=4


REM Create a folder for the workspace of the trial in question.
IF NOT EXIST %WORKSPACE_PATH% call MKDIR %WORKSPACE_PATH%
::SET WORKSPACE_PATH=!ORIGINAL_WORKSPACE_PATH!\Trial!i!


REM Begin execution of colmap with set parameters.

call ECHO feature extractor >> %WORKSPACE_PATH%\Log.txt

call ECHO %WORKSPACE_PATH%\sparce

call colmap feature_extractor ^
    --database_path %WORKSPACE_PATH%\database.db ^
    --image_path %IMAGE_PATH%

call ECHO exhaustive matcher >> %WORKSPACE_PATH%\Log.txt

call colmap exhaustive_matcher ^
   --database_path %WORKSPACE_PATH%\database.db

call ECHO sparse directory >> %WORKSPACE_PATH%/Log.txt

IF NOT EXIST %WORKSPACE_PATH%\sparse\ call MKDIR %WORKSPACE_PATH%\sparse

call ECHO mapper >> %WORKSPACE_PATH%\Log.txt

call colmap mapper ^
    --database_path %WORKSPACE_PATH%\database.db ^
    --image_path %IMAGE_PATH% ^
    --output_path %WORKSPACE_PATH%\sparse

call ECHO DENSE DIRECTORY  >> %WORKSPACE_PATH%\Log.txt  
     
IF NOT EXIST %WORKSPACE_PATH%\dense\ call MKDIR %WORKSPACE_PATH%\dense    

call ECHO IMAGE UNIDISTORTER >> %WORKSPACE_PATH%\Log.txt

call colmap image_undistorter ^
    --image_path %IMAGE_PATH% ^
    --input_path %WORKSPACE_PATH%\sparse\0 ^
    --output_path %WORKSPACE_PATH%\dense ^
    --output_type COLMAP ^
    --max_image_size 2000

call ECHO PATCH MATCH STEREO >> %WORKSPACE_PATH%\Log.txt

call colmap patch_match_stereo ^
    --workspace_path %WORKSPACE_PATH%\dense ^
    --workspace_format COLMAP ^
    --PatchMatchStereo.geom_consistency true ^
    --PatchMatchStereo.cache_size 64.0 ^
    --PatchMatchStereo.filter true ^
    --PatchMatchStereo.max_image_size 1000 ^
    --PatchMatchStereo.window_step 2 ^
    --PatchMatchStereo.window_radius 3 ^
    --PatchMatchStereo.num_iterations 3 ^
    --PatchMatchStereo.num_samples 10

call ECHO STEREO FUSION >> %WORKSPACE_PATH%\Log.txt   

call colmap stereo_fusion ^
    --workspace_path %WORKSPACE_PATH%\dense ^
    --workspace_format COLMAP ^
    --input_type geometric ^
    --output_path %WORKSPACE_PATH%\dense\fused.ply ^
    --StereoFusion.cache_size 64.0  ^
    --StereoFusion.max_image_size 1000

call ECHO POISSON MESHER >> %WORKSPACE_PATH%\Log.txt      

call colmap poisson_mesher ^
    --input_path %WORKSPACE_PATH%\dense\fused.ply ^
    --output_path %WORKSPACE_PATH%\dense\meshed-poisson.ply   

call ECHO DELAUNAY MESHER >> %WORKSPACE_PATH%\Log.txt

call colmap delaunay_mesher ^
    --input_path %WORKSPACE_PATH%\dense ^
    --output_path %WORKSPACE_PATH%\dense\meshed-delaunay.ply

call ECHO DELAUNAY MESHER WORKED >> %WORKSPACE_PATH%\Log.txt

ECHO %WORKSPACE_PATH%
ECHO %IMAGE_PATH%