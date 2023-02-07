# --------------------------------------------------------------------------- #
# This is the second major step in behavioral cloning: generating the data.
# We should have a set of cached initial configurations already set!
#
# 05/30/2022: supporting pour water.
# 07/20/2022: renaming script.
# --------------------------------------------------------------------------- #

# Note the `--save_data_bc` and `--obs_mode combo` arguments.
# For now just use filtered data (probably don't need unfiltered).

# We are using 128x128 images.
NVARS_F=2000
IS=128

# --------------------------------------------------------------------------- #
# v01 means rotation_bottom
# v02 is the same but with translation_axis_angle so the action is really 6D.
# Can remove --save_data_bc to get flow visualizations.

#ALG=pw_algo_v01
#ACT=rotation_bottom
ALG=pw_algo_v02
ACT=translation_axis_angle
OBS=combo
CAM=default_camera

python examples/demonstrator.py --num_variations $NVARS_F --use_cached_states \
        --camera_name $CAM --action_repeat 8 --env_name PourWater6D \
        --render_mode fluid --headless 1 --record_continuous_video \
        --alg_policy $ALG --camera_width $IS --camera_height $IS --img_size $IS \
        --obs_mode $OBS --act_mode $ACT --filtered  --save_data_bc
# --------------------------------------------------------------------------- #
