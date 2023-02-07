import os
import copy
from gym import error
import numpy as np
import gym
# from softgym.utils.visualization import save_numpy_as_gif
import cv2
import os.path as osp

# NOTE(daniel): temporary I think Yufei stored PourWater6D using Python 3.8, since I
# was getting ... ValueError: unsupported pickle protocol: 5.
# https://pypi.org/project/pickle5/
#import pickle
import pickle5 as pickle

try:
    import pyflex
except ImportError as e:
    raise error.DependencyNotInstalled("{}. (You need to first compile the python binding)".format(e))


class FlexEnv(gym.Env):
    def __init__(self,
                 device_id=-1,
                 headless=False,
                 render=True,
                 horizon=100,
                 camera_width=720,
                 camera_height=720,
                 num_variations=1,
                 action_repeat=8,
                 camera_name='default_camera',
                 deterministic=True,
                 use_cached_states=True,
                 save_cached_states=True, **kwargs):
        self.camera_params, self.camera_width, self.camera_height, self.camera_name = {}, camera_width, camera_height, camera_name
        pyflex.init(headless, render, camera_width, camera_height)

        self.record_video, self.video_path, self.video_name = False, None, None

        self.metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': int(np.round(1.0 / self.dt))
        }

        if device_id == -1 and 'gpu_id' in os.environ:
            device_id = int(os.environ['gpu_id'])
        self.device_id = device_id

        # By default SoftGym envs do NOT terminate early (they go for fixed # of steps).
        self.terminate_early = False
        self.horizon = horizon
        self.time_step = 0
        self.action_repeat = action_repeat
        self.recording = False
        self.prev_reward = None
        self.deterministic = deterministic
        self.use_cached_states = use_cached_states
        self.save_cached_states = save_cached_states
        self.current_config = self.get_default_config()
        self.current_config_id = None
        self.cached_configs, self.cached_init_states = None, None
        self.num_variations = num_variations

        self.dim_position = 4
        self.dim_velocity = 3
        self.dim_shape_state = 14
        self.particle_num = 0
        self.eval_flag = False
        self._filtered_configs = False

        # version 1 does not support robot, while version 2 does.
        # (Daniel) see https://github.com/Xingyu-Lin/softgym_rpad/issues/3
        pyflex_root = os.environ['PYFLEXROOT']
        if 'Robotics' in pyflex_root:
            self.version = 2
        else:
            self.version = 1
        print('using Flex version {}'.format(self.version))
        self.render_img = render

    def get_cached_configs_and_states(self, cached_states_path, num_variations):
        """If the path exists, load from it. Should be a list of (config, states)

        Useful to have a fixed set of configs so that we don't have to generate them
        each time (since in some envs, sampling starting states takes a long time).
        Remember, it's (config, states) so the former is a dict of dicts with params,
        the latter is the state so it includes `particle_pos`, `particle_vel`, etc.,
        letting us reconstruct the scene.

        The public SoftGym code doesn't update the camera (in contrast to internal). I
        removed the camera updating part and simplified it to just overriding the camera
        name. An env class's `initialize_camera()` will get called during `set_scene()`
        and we can update which camera we'd use.
        """
        if self.cached_configs is not None and self.cached_init_states is not None and len(self.cached_configs) == num_variations:
            return self.cached_configs, self.cached_init_states
        if not cached_states_path.startswith('/'):
            cur_dir = osp.dirname(osp.abspath(__file__))
            cached_states_path = osp.join(cur_dir, '../cached_initial_states', cached_states_path)

        if self.use_cached_states and osp.exists(cached_states_path):
            with open(cached_states_path, "rb") as handle:
                self.cached_configs, self.cached_init_states = pickle.load(handle)
            print('{} config and state pairs loaded from {}'.format(
                len(self.cached_init_states), cached_states_path))

            ## Optionally override default cameras; make sure the env has this camera
            ## option implemented! Comment this out if we just want the default camera.
            #for i in range(len(self.cached_configs)):
            #    self.cached_configs[i]['camera_name'] = 'top_down'

            # For simplicity, assume we load exactly the right amount.
            if len(self.cached_configs) == num_variations:
                return self.cached_configs, self.cached_init_states
            else:
                ValueError(f'{len(self.cached_configs)} vs {num_variations}')

        # Call env subclass and PyFlex/C++ files to initialize; potentially save cache.
        self.cached_configs, self.cached_init_states = self.generate_env_variation(num_variations)
        if self.save_cached_states:
            with open(cached_states_path, 'wb') as handle:
                pickle.dump((self.cached_configs, self.cached_init_states),
                            handle, protocol=pickle.HIGHEST_PROTOCOL)
            print('{} config and state pairs generated and saved to {}'.format(
                    len(self.cached_init_states), cached_states_path))
        return self.cached_configs, self.cached_init_states

    def get_current_config(self):
        return self.current_config

    def get_current_config_id(self):
        return self.current_config_id

    def get_filtered_configs_valid_list(self):
        return self._filtered_configs_valid

    def update_camera(self, camera_name, camera_param=None):
        """
        :param camera_name: The camera_name to switch to
        :param camera_param: None if only switching cameras. Otherwise, should be a dictionary
        :return:
        """
        if camera_param is not None:
            self.camera_params[camera_name] = camera_param
        else:
            camera_param = self.camera_params[camera_name]
        pyflex.set_camera_params(
            np.array([*camera_param['pos'], *camera_param['angle'], camera_param['width'], camera_param['height']]))

    @property
    def dt(self):
        # TODO get actual dt from the environment
        # TODO(daniel) is this ever used other than self.metadata?
        # This is normally 1/100, not 1/50 ...
        return 1 / 50.

    def get_state(self):
        pos = pyflex.get_positions()
        vel = pyflex.get_velocities()
        shape_pos = pyflex.get_shape_states()
        phase = pyflex.get_phases()
        camera_params = copy.deepcopy(self.camera_params)
        return {'particle_pos': pos, 'particle_vel': vel, 'shape_pos': shape_pos, 'phase': phase, 'camera_params': camera_params,
                'config_id': self.current_config_id}

    def set_state(self, state_dict):
        pyflex.set_positions(state_dict['particle_pos'])
        pyflex.set_velocities(state_dict['particle_vel'])
        pyflex.set_shape_states(state_dict['shape_pos'])
        pyflex.set_phases(state_dict['phase'])
        # self.camera_params = copy.deepcopy(state_dict['camera_params'])
        self.update_camera(self.camera_name)

    def close(self):
        pyflex.clean()

    def get_colors(self):
        '''
        Overload the group parameters as colors also
        '''
        groups = pyflex.get_groups()
        return groups

    def set_colors(self, colors):
        pyflex.set_groups(colors)

    def start_record(self):
        self.video_frames = []
        self.recording = True

    def end_record(self, video_path=None, **kwargs):
        if not self.recording:
            print('function end_record: Error! Not recording video')
        self.recording = False
        if video_path is not None:
            save_numpy_as_gif(np.array(self.video_frames), video_path, **kwargs)
        del self.video_frames

    def set_env_eval_flag(self, val):
        """NOTE(daniel) added this for SoftAgent code (SAC, BC, etc.)."""
        assert val in [True, False], val
        self.eval_flag = val

    def set_filtered_config_ranges(self, config_idxs_train, config_idxs_valid):
        """NOTE(daniel) added this for handling filtered configs.

        These are indices within the overall env's cached configs, since otherwise we
        would have to change a lot more code. For 1000 filtered configs, I used 2000
        total configs, so these indices go from 0 to 2000, but usually end much ealier.
        We also assume valid comes AFTER train.
        """
        self._filtered_configs_train = config_idxs_train
        self._filtered_configs_valid = config_idxs_valid
        #print(config_idxs_train, config_idxs_valid)
        self._filtered_configs = True

    def reset(self, config=None, initial_state=None, config_id=None):
        """Reset env and call pyflex code.

        Sets the scene based on a configuration file which has to be pre-generated.
        In real experiments, we should have hundreds (or thousands) of cached_configs,
        and we'd simply sample one of them (using either train or eval indices as
        appropriate) or we pre-provide a config_id. The former is helpful for training,
        the latter for validation (so we can go through test configs exactly once).

        Also, this is why there's no notion of a random seed -- the random seed comes
        from the initial configuration, which is pre-generated.

        Note that `self.set_scene()` happens BEFORE `self._reset()`!!
        04/10/2022: adding another case for `self._filtered configs`.
        """
        #print(f'FlexEnv.reset(config_id={config_id}), eval: {self.eval_flag}, {len(self.cached_configs)}')
        if config is None:
            if config_id is None:
                if self.eval_flag:
                    if self._filtered_configs:
                        if self.deterministic:
                            config_id = self.filtered_configs_valid[0]
                        else:
                            _id = np.random.randint(low=0, high=len(self._filtered_configs_valid))
                            config_id = self._filtered_configs_valid[_id]
                    else:
                        eval_beg = int(0.8 * len(self.cached_configs))
                        config_id = np.random.randint(low=eval_beg, high=len(self.cached_configs)) if not self.deterministic else eval_beg
                else:
                    if self._filtered_configs:
                        if self.deterministic:
                            config_id = 0
                        else:
                            _id = np.random.randint(low=0, high=len(self._filtered_configs_train))
                            config_id = self._filtered_configs_train[_id]
                    else:
                        train_high = int(0.8 * len(self.cached_configs))
                        config_id = np.random.randint(low=0, high=max(train_high, 1)) if not self.deterministic else 0
            #print(f'  FlexEnv.reset(), now config_id={config_id}), det.: {self.deterministic}')
            self.current_config = self.cached_configs[config_id]
            self.current_config_id = config_id
            self.set_scene(self.cached_configs[config_id], self.cached_init_states[config_id])
        else:
            self.current_config = config
            self.set_scene(config, initial_state)
        self.particle_num = pyflex.get_n_particles()
        self.prev_reward = 0.
        self.time_step = 0
        obs = self._reset()
        if self.recording:
            self.video_frames.append(self.render(mode='rgb_array'))
        return obs

    def step(self, action, record_continuous_video=False, img_size=None):
        """If record_continuous_video is set to True, records image for each sub-step.

        Daniel notes:
        - The `done=False` always, so episodes end based on horizon count alone. In external
        code, we allow for early termination by recording `done` in `_get_info()`.
        - Some envs have action repeat, in which case the given `step()` runs the env's own
        `_step()` multiple times with the _same_ action as input. If recording, we record
        every other frame, and put in the `info` dict.
        - There is also a `self.video_frames` which records 1 image per time step, and must be
        activated with `self.start_record()`. This can serve the same purpose as the frames
        and they are equivalent if we only record 1 frame among all action_repeats.
        """
        frames = []
        for i in range(self.action_repeat):
            self._step(action)
            if record_continuous_video and i % 4 == 0:  # No need to record each step
                frames.append(self.get_image(img_size, img_size))
        obs = self._get_obs()
        reward = self.compute_reward(action, obs, set_prev_reward=True)
        info = self._get_info()

        if self.recording:
            self.video_frames.append(self.render(mode='rgb_array'))
        self.time_step += 1

        done = False
        if self.time_step >= self.horizon:
            done = True
        if self.terminate_early and info['done']:
            done = True
        if record_continuous_video:
            info['flex_env_recorded_frames'] = frames
        return obs, reward, done, info

    def initialize_camera(self):
        """
        This function sets the postion and orientation of the camera
        camera_pos: np.ndarray (3x1). (x,y,z) coordinate of the camera
        camera_angle: np.ndarray (3x1). (x,y,z) angle of the camera (in degree).

        Note: to set camera, you need
        1) implement this function in your environement, set value of self.camera_pos and self.camera_angle.
        2) add the self.camera_pos and self.camera_angle to your scene parameters,
            and pass it when initializing your scene.
        3) implement the CenterCamera function in your scene.h file.
        Pls see a sample usage in pour_water.py and softgym_PourWater.h

        if you do not want to set the camera, you can just not implement CenterCamera in your scene.h file,
        and pass no camera params to your scene.
        """
        raise NotImplementedError

    def render(self, mode='rgb_array'):
        if mode == 'rgb_array':
            img = pyflex.render()
            width, height = self.camera_params[self.camera_name]['width'], self.camera_params[self.camera_name]['height']
            img = img.reshape(height, width, 4)[::-1, :, :3]  # Need to reverse the height dimension
            return img
        elif mode == 'human':
            raise NotImplementedError

    def get_image(self, width=720, height=720):
        """ use pyflex.render to get a rendered image. """
        img = self.render(mode='rgb_array')
        img = img.astype(np.uint8)
        if width != img.shape[0] or height != img.shape[1]:
            img = cv2.resize(img, (width, height))
        return img

    def set_scene(self, config, state=None):
        """ Set up the flex scene """
        raise NotImplementedError

    def get_default_config(self):
        """ Generate the default config of the environment scenes"""
        raise NotImplementedError

    def generate_env_variation(self, num_variations, **kwargs):
        """
        Generate a list of configs and states
        :return:
        """
        raise NotImplementedError

    def compute_reward(self, action=None, obs=None, set_prev_reward=False):
        """ set_prev_reward is used for calculate delta rewards"""
        raise NotImplementedError

    def _get_obs(self):
        raise NotImplementedError

    def _get_info(self):
        raise NotImplementedError

    def _reset(self):
        raise NotImplementedError

    def _step(self, action):
        raise NotImplementedError

    def _seed(self):
        pass
