import numpy as np
from gym.spaces import Box
import pyflex
from softgym.envs.flex_env import FlexEnv
from softgym.action_space.action_space import ParallelGripper, Picker, PickerPickPlace
from softgym.action_space.robot_env import RobotBase
from copy import deepcopy
from scipy.spatial.transform import Rotation as R
import os
from softgym.utils.pyflex_utils import center_object


class ClothFromPC(FlexEnv):
    def __init__(self, observation_mode, action_mode, num_picker=2, picker_radius=0.0216, render_mode='particle',
        particle_radius=0.0216, picker_threshold=0.025, nodes=None, edges=None, **kwargs):

        self.action_mode = action_mode
        self.cloth_particle_radius = particle_radius
        self.render_mode = render_mode
        self.nodes = nodes
        self.edges = edges
        super().__init__(**kwargs)

        # assert observation_mode in ['2d_obs','3d_obs','simple']
        assert action_mode in ['sphere', 'picker', 'pickerpickplace', 'sawyer', 'franka', 'picker2dpp']
        self.observation_mode = observation_mode

        if action_mode.startswith('key_point'):
            space_low = np.array([0, -0.1, -0.1, -0.1] * 2)
            space_high = np.array([3.9, 0.1, 0.1, 0.1] * 2)
            self.action_space = Box(space_low, space_high, dtype=np.float32)
        elif action_mode.startswith('sphere'):
            self.action_tool = ParallelGripper(gripper_type='sphere')
            self.action_space = self.action_tool.action_space
        elif action_mode == 'picker':
            self.action_tool = Picker(num_picker, picker_radius=picker_radius, particle_radius=particle_radius,
                                      picker_low=(-0.3, 0.0125, -0.3), picker_high=(0.3, 0.3, 0.3))
            self.action_space = self.action_tool.action_space
            self.action_dim = num_picker * 4
        elif action_mode == 'pickerpickplace':
            self.action_tool = PickerPickPlace(num_picker=num_picker, picker_radius=picker_radius, particle_radius=particle_radius, env=self,
                                               picker_low=(-0.3, 0.0125, -0.3), picker_high=(0.3, 0.3, 0.3), collect_steps=collect_steps)
            self.action_space = self.action_tool.action_space
            self.action_dim = num_picker * 4
            assert self.action_repeat == 1
        elif action_mode in ['sawyer', 'franka']:
            self.action_tool = RobotBase(action_mode)
            self.action_space = self.action_tool.action_space

        self.state_dim = num_picker * 4
        print("simple: ",self.state_dim)
        self.observation_space = Box(low=0., high=1., shape=(self.state_dim,), dtype=np.float32)

        self.default_pos = None


    def move_to_pos(self,new_pos):
        # TODO
        pos = pyflex.get_positions().reshape(-1, 4)
        center = np.mean(pos, axis=0)
        pos[:,:3] -= center[:3]
        pos[:,:3] += np.asarray(new_pos)
        pyflex.set_positions(pos)


    def apply_rotation(self,euler):
        r = R.from_euler('zyx', euler, degrees=True)
        pos = pyflex.get_positions().reshape(-1, 4)
        center = np.mean(pos, axis=0)
        pos -= center
        new_pos = pos.copy()[:,:3]
        new_pos = r.apply(new_pos)
        new_pos = np.column_stack([new_pos,pos[:,3]])
        new_pos += center
        pyflex.set_positions(new_pos)


    def _set_to_flat(self):
        pyflex.set_positions(self.default_pos)
        self.apply_rotation([0,0,90])
        pyflex.step()


    def get_default_config(self):
        """ Set the default config of the environment and load it to self.config """
        if self.action_mode in ['sawyer', 'franka']:
            cam_pos, cam_angle = np.array([0.0, 1.62576, 1.04091]), np.array([0.0, -0.844739, 0])
        else:
            cam_pos, cam_angle = np.array([-0.0, 0.82, 0.82]), np.array([0, -45 / 180. * np.pi, 0.])

        config = {
            'pos': [0.0, 0.0, 0.0],
            'scale': 1,
            'rot': 0.0,
            'vel': [0., 0., 0.],
            'stiff': 0.9,
            'mass': 1,
            'radius': self.cloth_particle_radius,
            'camera_name': 'default_camera',
            'camera_params': {'default_camera':
                                  {'pos': cam_pos,
                                   'angle': cam_angle,
                                   'width': self.camera_width,
                                   'height': self.camera_height}},
        }

        return config


    def get_rgbd(self):
        rgbd = pyflex.render_sensor()
        rgbd = np.array(rgbd).reshape(self.camera_height, self.camera_width, 4)
        rgbd = rgbd[::-1, :, :]
        rgb = rgbd[:, :, :3]
        depth = rgbd[:, :, 3]

        return rgb, depth


    def _get_obs(self, fix_debug=False):

        if self.action_mode in ['picker','pickerpickplace','picker2dpp']:
            picker_pos, particle_pos = self.action_tool._get_pos()
            picker_open = [1 if self.action_tool.picked_particles[i] == None else 0 for i in range(self.action_tool.num_picker)]
            picker_pos = np.concatenate([picker_pos.flatten(),picker_open])

        return self.get_image(self.camera_width, self.camera_height)


    """
    There's always the same parameters that you can set 
    """

    def set_scene(self, config, state=None):
        camera_params = config['camera_params'][config['camera_name']]
        # TODO: check the environment idx
        env_idx = 6 if 'env_idx' not in config else config['env_idx']
        if self.render_mode == 'particle':
            render_mode = 1
        elif self.render_mode == 'cloth':
            render_mode = 2
        elif self.render_mode == 'both':
            render_mode = 3

        num_nodes = len(self.nodes) // 3
        num_edges = len(self.edges) // 2
        scene_params = np.concatenate([ config['pos'][:], [config['scale'], config['rot']], config['vel'][:], [config['stiff'], config['mass'], config['radius']],
                                camera_params['pos'][:], camera_params['angle'][:], [camera_params['width'], camera_params['height']], [render_mode], 
                                [num_nodes, num_edges], self.nodes, self.edges])
        if self.version == 2:
            robot_params = [1.] if self.action_mode in ['sawyer', 'franka'] else []
            self.params = (scene_params, robot_params)
            pyflex.set_scene(env_idx, scene_params, 0, robot_params)
        elif self.version == 1:
            pyflex.set_scene(env_idx, scene_params, 0)

        if state is not None:
            self.set_state(state)

        self.default_pos = pyflex.get_positions().reshape(-1, 4)
        self.current_config = deepcopy(config)

    def _step(self, action):
        self.action_tool.step(action)
        if self.action_mode in ['sawyer', 'franka']:
            pyflex.step(self.action_tool.next_action)
        else:
            pyflex.step()
        return self._get_obs()

    def reset(self):
        config = self.get_default_config()
        self.set_scene(config)
        # center_object()
        self.prev_reward = 0.
        self.time_step = 0
        if hasattr(self, 'action_tool'):
            self.action_tool.reset([0, 0.1, 0])
        if self.recording:
            self.video_frames.append(self.render(mode='rgb_array'))

        self.camera_params[self.camera_name] = config['camera_params'][self.camera_name] 
        obs = self._get_obs()
        return obs

    # TODO: change this
    def compute_reward(self, action=None, obs=None, set_prev_reward=False):
        """ set_prev_reward is used for calculate delta rewards"""
        return 0

    def _get_info(self):
        return {}
