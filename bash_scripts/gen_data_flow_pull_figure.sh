# --------------------------------------------------------------------------- #
# 11/09/2022: use this for generating the pull figure and modifying it, etc.
# To make it easier, I'm sticking with `pull_figure` as a camera. I just use
# seed 0, so we don't use a cached figure but just run it. Then make sure to
# use `--collect_visuals`. Might need to run multiple times with different
# downsampling and scaling factors.
# --------------------------------------------------------------------------- #
IS=512

# --------------------------------------------------------------------------- #
# PourWater -- this is what we are using.
# --------------------------------------------------------------------------- #
ALG=pw_algo_v02
ACT=translation_axis_angle
CAM=pull_figure
OBS=combo

python examples/demonstrator.py --num_variations 1   \
        --camera_name $CAM --action_repeat 8 --env_name PourWater \
        --render_mode fluid --headless 0 --record_continuous_video \
        --alg_policy $ALG --camera_width $IS --camera_height $IS --img_size $IS \
        --obs_mode $OBS --act_mode $ACT --filtered --collect_visuals
# --------------------------------------------------------------------------- #