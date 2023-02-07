# --------------------------------------------------------------------------- #
# 08/23/2022: use this for the rebuttal to improve GIFs on the project website?
# I think we can save GIFs and then convert them to videos for the website.
# --------------------------------------------------------------------------- #
# We are _normally_ using 128x128 images. But our segmenented point cloud figure
# creates (600,600) images so maybe we actually want to try 600x600 images?
# BUT this is going to annoyingly change our segmentation because it means there
# are more pixels for tool points. So, I propose we run this ONCE at the start
# to get a 600x600 RGB set of frames. Then we redo it with 128x128 to get the
# 600x600 _segmented_point_cloud_ GIF and the _flow_ GIF. Then we manually merge
# the RGB and segmented point cloud GIFs using GIF optimizer online. Then we
# can figure out how to merge that with the flow vectors. I will also use a
# 600x600 image for the flow visualizations.
# Edit: for scooping I had to keep IS at 600, for pouring I used 600 for the
# RGB GIF, and then switched to 128 for flow and PCL visuals.
# --------------------------------------------------------------------------- #
IS=600  # can be slow for scooping

# --------------------------------------------------------------------------- #
# Use to get segmented PCL and flow visualizations for the project website.
# SCOOPING
# --------------------------------------------------------------------------- #
NVARS_F=1200
ALG=ladle_6dof_rotations_scoop
ACT=translation_axis_angle
OBS=combo

# Single-sphere.
python examples/demonstrator.py --num_variations $NVARS_F --use_cached_states \
        --env_name MMOneSphere --render_mode fluid --headless 1 --record_continuous_video \
        --alg_policy $ALG --camera_width $IS --camera_height $IS --img_size $IS \
        --obs_mode $OBS --act_mode $ACT --filtered  --tool_data 4 --action_repeat 8
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Use to get segmented PCL and flow visualizations for the project website.
# POURING
# --------------------------------------------------------------------------- #
NVARS_F=2000
ALG=pw_algo_v02
ACT=translation_axis_angle
OBS=combo
CAM=default_camera

python examples/demonstrator.py --num_variations $NVARS_F --use_cached_states \
        --camera_name $CAM --action_repeat 8 --env_name PourWater6D \
        --render_mode fluid --headless 1 --record_continuous_video \
        --alg_policy $ALG --camera_width $IS --camera_height $IS --img_size $IS \
        --obs_mode $OBS --act_mode $ACT --filtered
# --------------------------------------------------------------------------- #




# # --------------------------------------------------------------------------- #
# # --------------------------------------------------------------------------- #
# I think this is what I used to generate the teaser figure for CoRL 2022 subm.?
# This is the one that uses pouring and shows flow vectors from that.
# # --------------------------------------------------------------------------- #
# For generating _aligned_ images, point clouds, and flow visualizations.
# Use --collect_visuals argument.
# Updated (06/15/2022). Note: for PourWater I ended up just using a new cached
# config, for the MMSphere I stuck with the existing cached data.
# # --------------------------------------------------------------------------- #
# # --------------------------------------------------------------------------- #
# # For a paper might as well crank up the figure size.
# IS=512
#
# # --------------------------------------------------------------------------- #
# # PourWater
# # --------------------------------------------------------------------------- #
# ALG=pw_algo_v02
# ACT=translation_axis_angle
# CAM=default_camera
# OBS=combo
#
# python examples/demonstrator.py --num_variations 1   \
#         --camera_name $CAM --action_repeat 8 --env_name PourWater \
#         --render_mode fluid --headless 0 --record_continuous_video \
#         --alg_policy $ALG --camera_width $IS --camera_height $IS --img_size $IS \
#         --obs_mode $OBS --act_mode $ACT --filtered --collect_visuals
# # --------------------------------------------------------------------------- #
#
# # --------------------------------------------------------------------------- #
# # MMOneSphere
# # --------------------------------------------------------------------------- #
# NVARS_F=2000
# ALG=ladle_algorithmic_v04
# ACT=translation_axis_angle
# CAM=top_down
# OBS=combo
#
# python examples/demonstrator.py --num_variations $NVARS_F --use_cached_states \
#         --camera_name $CAM --action_repeat 8 --env_name MMOneSphere \
#         --render_mode fluid --headless 0 --record_continuous_video \
#         --alg_policy $ALG --camera_width $IS --camera_height $IS --img_size $IS \
#         --obs_mode $OBS --act_mode $ACT --filtered --collect_visuals
# # --------------------------------------------------------------------------- #