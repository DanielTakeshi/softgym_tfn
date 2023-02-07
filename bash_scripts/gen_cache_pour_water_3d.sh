# ---------------------------------------------------------------------- #
# For the PourWater env. We can use 1500 cached configs, this should
# be more than enough for us. Run ONCE, then generate data for IL later.
# ---------------------------------------------------------------------- #
# Reminder I: DO NOT override existing cached files!
# Reminder II: if we want num variations != 1000, we change the cached names.
# Reminder III: check the env's `pw_env_version`.

# NOTE! Default camera adjusted 05/31/2022.
# NOTE! It's OK if we dont' use `rotation_bottom` later.
python examples/demonstrator.py --env_name PourWater \
    --act_mode rotation_bottom  --camera_name default_camera \
    --num_variations 1500  --save_cached_states  --headless 1
