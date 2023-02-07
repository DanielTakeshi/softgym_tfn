# --------------------------------------------------------------------------- #
# Test PourWater (3DoF).
# --------------------------------------------------------------------------- #

# Can change the image size. We used 128 for the paper, but you can increase it
# to a value like 256 or 512 for visualization purposes.
IS=256

python examples/demonstrator.py \
        --num_variations 1 \
        --camera_name default_camera \
        --action_repeat 8 \
        --env_name PourWater \
        --render_mode fluid \
        --headless 0 \
        --record_continuous_video \
        --alg_policy pw_algo_v02 \
        --camera_width $IS \
        --camera_height $IS \
        --img_size $IS \
        --obs_mode cam_rgb \
        --act_mode translation_axis_angle
# --------------------------------------------------------------------------- #
