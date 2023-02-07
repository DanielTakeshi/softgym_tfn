from distutils.log import info
import numpy as np
from gym.spaces import Box
import pyflex
import copy
import random, math
from softgym.envs.fluid_env import FluidEnv
from softgym.utils.misc import rotate_rigid_object, quatFromAxisAngle
from softgym.utils.camera_projections import get_matrix_world_to_camera
from shapely.geometry import Polygon

from softgym.utils.segmentation_pour_water import SegmentationPourWater
from scipy.spatial.transform import Rotation as Rot
DEG_TO_RAD = np.pi / 180.
RAD_TO_DEG = 180 / np.pi

from scipy.linalg import eigh
from scipy.optimize import minimize_scalar


def ellipsoid_intersection_test(Sigma_A, Sigma_B, mu_A, mu_B, tau=1.0):
    lambdas, Phi = eigh(Sigma_A, b=Sigma_B)
    v_squared = np.dot(Phi.T, mu_A - mu_B) ** 2
    res = minimize_scalar(K_function,
                          bracket=[0.0, 0.5, 1.0],
                          args=(lambdas, v_squared, tau))
    # print(res.fun)
    return (res.fun >= 0)


def K_function(s, lambdas, v_squared, tau):
    return 1.-(1./tau**2)*np.sum(v_squared*((s*(1.-s))/(1.+s*(lambdas-1.))))


class PourWater6DEnv(FluidEnv):

    def __init__(self, observation_mode, action_mode, real_world_scale=False,
                 cached_states_path='pour_water_init_states.pkl', **kwargs):
        """Implements a pouring water task.

        Originally in SoftGym, options were:
            observation_mode: "cam_rgb" or "full_state"
            action_mode: "rotation_bottom, rotation_top"
        We change this for the tool flow project.

        Unfortunately the water leaks out of the box. Might be able to address this
        if we just import a box (e.g., from Blender) in the sim but that will make
        position control harder.

        Actions: (x,y,theta) where x,y represent the translation of the box to move
        (there is another box that is fixed) and theta is one rotation. Box to move
        starts with bottom floor center (x=0,y=0).

        Class attributes:
          (can change from agent)
            glass_x: box x coord (starts at 0, positive moves to target)
                Does not change if only rotating
            glass_y: box y coord (starts at 0 if rotation_bottom, positive moves up)
                Does not change if only rotating
            glass_rotation_z: box rotation (starts at 0, positive dumps in +x)
          (fixed per episode)
            border: thickness of controlled box's walls
            height: height of controlled box
            glass_dis_x: glass length
            glass_dis_z: glass width
            glass_distance: I think distance between two floor centers of boxes
            poured_border: thickness of target box's walls
            poured_height: height of target box
            poured_glass_dis_x: poured glass length
            poured_glass_dis_z: poured glass width

        The rotation can be either wrt the bottom of the box, or wrt the top. That
        sets orientation of the rotation. Also it affects how we init `self.glass_y`.
        We might need to change this for learning from point clouds.
        """
        self.name = 'PourWaterEnv'
        self.pw_env_version = 1
        sp = f'_v{str(self.pw_env_version).zfill(2)}.pkl'
        self.cached_states_path = cached_states_path.replace('.pkl', sp)
        self.act_noise = 0.0
        self.real_world_scale = real_world_scale
        self.observation_mode = observation_mode
        self.action_mode = action_mode
        self.wall_num = 5  # number of glass walls. floor/left/right/front/back
        self.alg_policy = None  # as with mixed media...
        self.use_fake_tool = False   # use False for now
        self.obs_dim_keypt_shape = (10,14)  # compatibility with MM & BC code
        self.inner_step = 0

        super().__init__(**kwargs)

        # Handle action space. Main change is we might need to rotate wrt the
        # center of the box, instead of one of its 'floors'. I'm keeing the
        # rotation_bottom for backwards compatibility but I think we want to
        # use an action with full translation / axis-angle for generality.
        default_config = self.get_default_config()
        b = default_config['glass']['border']  # self.border: set_pouring_glass_params
        assert action_mode in ['rotation_bottom', 'translation_axis_angle'], \
            f'Have not yet tested {action_mode}.'

        if action_mode in ["rotation_bottom", "rotation_top"]:
            # control (x,y) coordinate of floor center, and theta its rotation angle.
            self.action_direct_dim = 3
            if not self.real_world_scale:
                action_low = np.array([-b * 0.5, -b * 0.5, -0.015])
                action_high = np.array([b * 0.5, b * 0.5, 0.015])
                self.action_space = Box(action_low, action_high, dtype=np.float32)
            else:
                action_low = np.array([-b * 0.5, -b * 0.5, -0.01])
                action_high = np.array([b * 0.5, b * 0.5, 0.01])
                self.action_space = Box(action_low, action_high, dtype=np.float32)
        elif action_mode in ["rotation_bottom_3d", "rotation_top_3d"]:
            # control (x,y,z) coordinate of floor center, and theta its rotation angle.
            self.action_direct_dim = 4
            if not self.real_world_scale:
                action_low = np.array([-b * 0.5, -b * 0.5, -b * 0.5, -0.015])
                action_high = np.array([b * 0.5, b * 0.5, b * 0.5, 0.015])
                self.action_space = Box(action_low, action_high, dtype=np.float32)
            else:
                action_low = np.array([-b* 0.5, -b * 0.5, -b * 0.5, -0.01])
                action_high = np.array([b* 0.5, b * 0.5, b * 0.5, 0.01])
                self.action_space = Box(action_low, action_high, dtype=np.float32)
        elif action_mode in ['translation_axis_angle']:
            # Careful this has not been tested with >1 DoF rotation.
            self.action_direct_dim = 6
            action_low  = np.array([-b*0.5, -b*0.5, -b*0.5, -0.015, -0.015, -0.015])
            action_high = np.array([ b*0.5,  b*0.5,  b*0.5,  0.015,  0.015,  0.015])
            self.action_space = Box(action_low, action_high, dtype=np.float32)
            self.max_rot_axis_ang = (10. * DEG_TO_RAD) / self.action_repeat
        else:
            raise NotImplementedError(action_mode)

        self.get_cached_configs_and_states(self.cached_states_path, self.num_variations)

        # Point cloud stuff. Max_pts is the max # of points in PCL.
        self.max_pts = 2000
        self.oob_x = (-0.5, 1.5)
        self.oob_z = (-0.5, 0.5)
        self.pc_point_dim = 6  # (3 xyz, 3 classes)

        # Handle observation space. Daniel: keeping 'rim interpolation' stuff here
        # but I don't think we will need to use,.
        if observation_mode in ['key_point_1', 'key_point_2', 'key_point_3', 'key_point_4']:
            if observation_mode == 'key_point_1':
                obs_dim = 0
                obs_dim += 13  # Pos (x, z, theta) and shape (w, h, l) of the two cups and the water height.
            elif observation_mode == 'key_point_2': # no water inforamtion
                obs_dim = 10
            elif observation_mode == 'key_point_3': # no water & pouring height info
                obs_dim = 9
            elif observation_mode == 'key_point_4':
                obs_dim = 7
            # z and theta of the second cup (poured_glass) does not change and thus are omitted.
            # add: frac of water in control cup, frac of water in target cup
            self.observation_space = Box(
                    low=np.array([-np.inf] * obs_dim),
                    high=np.array([np.inf] * obs_dim),
                    dtype=np.float32)
        elif observation_mode in ['state']:
            obs_dim = 22
            self.observation_space = Box(
                    low=np.array([-np.inf] * obs_dim),
                    high=np.array([np.inf] * obs_dim),
                    dtype=np.float32)
        elif observation_mode in ['point_cloud']:
            # point_cloud: depth camera extracts box points, might have occlusion
            # (later?) point_cloud_gt: use a fixed set of known box points
            obs_dim = (self.max_pts, self.pc_point_dim)  # (N,d)
            self.observation_space = Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=obs_dim,
                    dtype=np.float32)
        elif observation_mode == 'cam_rgb':
            self.observation_space = Box(
                    low=-np.inf, high=np.inf,
                    shape=(self.camera_height, self.camera_width, 3),
                    dtype=np.float32)
        elif observation_mode == 'cam_rgbd':
            self.observation_space = Box(
                    low=-np.inf, high=np.inf,
                    shape=(self.camera_height, self.camera_width, 4),
                    dtype=np.float32)
        elif observation_mode == 'depth_img':
            # Stack the depth.
            self.observation_space = Box(
                    low=-np.inf, high=np.inf,
                    shape=(self.camera_height, self.camera_width, 3),
                    dtype=np.float32)
        elif observation_mode == 'depth_segm':
            # Design choice. 1st channel depth, 2,3,4 are tool, target cup, water.
            self.observation_space = Box(
                    low=-np.inf, high=np.inf,
                    shape=(self.camera_height, self.camera_width, 4),
                    dtype=np.float32)
        elif observation_mode == 'rgb_segm_masks':
            # Design choice. 6 channels: RGB, tool mask, targ mask, water mask.
            self.observation_space = Box(
                    low=-np.inf, high=np.inf,
                    shape=(self.camera_height, self.camera_width, 6),
                    dtype=np.float32)
        elif observation_mode == 'rgbd_segm_masks':
            # Design choice. 7 channels: RGBD, tool mask, targ mask, water mask.
            self.observation_space = Box(
                    low=-np.inf, high=np.inf,
                    shape=(self.camera_height, self.camera_width, 7),
                    dtype=np.float32)
        elif observation_mode in ['rim_interpolation', 'rim_interpolation_normalize', 'rim_interpolation_and_state']:
            # a random observation space, as KPConv do not need to use this.
            self.observation_space = Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(1000,  3),
                    dtype=np.float32)
        elif observation_mode == 'rim_interpolation_flatten':
            # a random observation space, as KPConv do not need to use this.
            max_point_num = 12 * 4 * 2
            self.max_point_num = max_point_num
            self.observation_space = Box(
                    low=np.array([-np.inf] * max_point_num * 6),
                    high=np.array([np.inf] * max_point_num * 6),
                    dtype=np.float32)
        elif observation_mode in ['rim_graph', 'rim_graph_hierarchy']:
            # a random observation space, as GNN will not really use this.
            self.observation_space = Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(1000,  3),
                    dtype=np.float32)
        elif observation_mode == 'combo':
            # Combination of stuff, mainly for BC (it will not really use this).
            self.observation_space = Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(1000,  3),
                    dtype=np.float32)
            self.particle_obs_dim = self.max_pts
        else:
            raise NotImplementedError(observation_mode)



        # If we add another action, check `rotate_glass()` to add there also.


        # Reward handling.
        self.prev_reward = 0
        self.reward_min = 0
        self.reward_max = 1
        self.reward_range = self.reward_max - self.reward_min
        self.success_thresh = None  # TODO(daniel)

        # Handle segmentation, but don't forget to do a reset in _reset().
        self.segm = SegmentationPourWater(self.use_fake_tool)

        # Initialize an 'Alg Policy' class if needed.
        self.AlgPolicyCls = PWAlgorithmicPolicy(env=self)

    def get_default_config(self):
        if not self.real_world_scale:
            config = {
                'fluid': {
                    'radius': 0.033,
                    'rest_dis_coef': 0.55,
                    'cohesion': 0.1,  # not actually used, instead, is computed as viscosity * 0.01
                    'viscosity': 2,
                    'surfaceTension': 0,
                    'adhesion': 0.0,  # not actually used, instead, is computed as viscosity * 0.001
                    'vorticityConfinement': 40,
                    'solidpressure': 0.,
                    'dim_x': 8,
                    'dim_y': 18,
                    'dim_z': 8,
                },
                'glass': {
                    'border': 0.02,
                    'height': 0.6, # not used, will be cacluated later in generate_env_variation
                    'glass_distance': 1.0, # not used, will be cacluated later in generate_env_variation
                    'poured_border': 0.02,
                    'poured_height': 0.6, # not used, will be cacluated later in generate_env_variation
                    'fake_offset_x': 2.0, # for fake tool
                },
                'camera_name': self.camera_name,
            }
        else:
            config = {
                'fluid': {
                    'radius': 0.006, # for calibration with real-world
                    'rest_dis_coef': 0.55,
                    'cohesion': 1,  # not actually used, instead, is computed as viscosity * 0.01
                    'viscosity': 10,
                    'surfaceTension': 0,
                    'adhesion': 0.0,  # not actually used, instead, is computed as viscosity * 0.001
                    'vorticityConfinement': 40,
                    'solidpressure': 0.,
                    'dim_x': 8,
                    'dim_y': 18,
                    'dim_z': 8,
                },
                'glass': {
                    'border': 0.004, # calibrate with real world
                    'height': 0.6, # not used, will be cacluated later in generate_env_variation
                    'glass_distance': 1.0, # not used, will be cacluated later in generate_env_variation
                    'poured_border': 0.004, # calibrate with real world
                    'poured_height': 0.6, # not used, will be cacluated later in generate_env_variation
                },
                'camera_name': self.camera_name,
            }
        return config

    def generate_env_variation(self, num_variations=5, config=None, **kwargs):
        """
        TODO: add more randomly generated configs instead of using manually specified configs.
        """
        # NOTE(daniel): would recommend not generating 11 or more.
        dim_xs = [4, 5, 6, 7, 8, 9, 10]
        dim_zs = [4, 5, 6, 7, 8, 9, 10]
        #dim_xs = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        #dim_zs = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

        self.cached_configs = []
        self.cached_init_states = []
        if config is None:
            config = self.get_default_config()
        config_variations = [copy.deepcopy(config) for _ in range(num_variations)]

        for idx in range(num_variations):
            print("\nPourWater generate env variations {}".format(idx))
            dim_x = random.choice(dim_xs)
            if not self.real_world_scale:
                dim_z = random.choice(dim_zs)
            else:
                dim_z = dim_x
            m = min(dim_x, dim_z)
            p = np.random.rand()
            water_radius = config['fluid']['radius'] * config['fluid']['rest_dis_coef']

            if not self.real_world_scale:
                if p < 0.5:  # medium water volumes
                    print("medium volume water")
                    dim_y = int(3.5 * m)
                    v = dim_x * dim_y * dim_z
                    h = v / ((dim_x + 1) * (dim_z + 1)) * water_radius / 2
                    print("h {}".format(h))
                    glass_height = h + (np.random.rand() - 0.5) * 0.001
                else:
                    print("large volume water")
                    dim_y = 4 * m
                    v = dim_x * dim_y * dim_z
                    h = v / ((dim_x + 1) * (dim_z + 1)) * water_radius / 3
                    print("h {}".format(h))
                    glass_height = h + (m + np.random.rand()) * 0.001
            else:
                dim_y = dim_x * np.random.randint(6, 8)
                glass_height = dim_x * 2 * water_radius

            print("dim_x {} dim_y {} dim_z {} glass_height {}".format(
                    dim_x, dim_y, dim_z, glass_height))
            config_variations[idx]['fluid']['dim_x'] = dim_x
            config_variations[idx]['fluid']['dim_y'] = dim_y
            config_variations[idx]['fluid']['dim_z'] = dim_z
            # if you want to change viscosity also, uncomment this
            # config_variations[idx]['fluid']['viscosity'] = self.rand_float(2.0, 10.0)

            config_variations[idx]['glass']['height'] = glass_height
            if not self.real_world_scale:
                config_variations[idx]['glass']['poured_height'] = glass_height + np.random.rand() * 0.1
                config_variations[idx]['glass']['glass_distance'] = self.rand_float(0.05 * m, 0.09 * m) + (dim_x + 4) * water_radius / 2.
                config_variations[idx]['glass']['poured_border'] = self.rand_float(0.008, 0.01)
            else:
                config_variations[idx]['glass']['poured_height'] = glass_height + np.random.rand() * 0.002
                config_variations[idx]['glass']['glass_distance'] = self.rand_float(0.1, 0.3)

            self.set_scene(config_variations[idx])
            init_state = copy.deepcopy(self.get_state())
            self.cached_configs.append(config_variations[idx])
            self.cached_init_states.append(init_state)

        combined = [self.cached_configs, self.cached_init_states]
        return self.cached_configs, self.cached_init_states

    def get_config(self):
        if self.deterministic:
            config_idx = 0
        else:
            config_idx = np.random.randint(len(self.config_variations))
        self.config = self.config_variations[config_idx]
        return self.config

    def _reset(self):
        """Reset env to initial state, return starting obs.

        As with mixed media, we might need to handle segmentation stuff here.
        Note: `set_scene()` which sets the boxes is called BEFORE this, so at
        the start we should be able to extract all points for the target box.
        Track that, then any other changes we know must be due to the tool box.
        """
        # Reset the point cloud dict and alg policy, if using point clouds.
        self.pcl_dict = {}
        self.AlgPolicyCls.reset()

        self.inner_step = 0
        self.performance_init = None
        info = self._get_info()
        self.performance_init = info['performance']
        pyflex.step(render=self.render_img)

        # -------------- NOTE(daniel) new stuff -------------- #
        cp = self.camera_params[self.camera_name]
        self.matrix_world_to_camera = get_matrix_world_to_camera(cp)
        self.segm.assign_camera(
                self.camera_params,
                self.camera_name,
                self.matrix_world_to_camera
        )

        # Get x values of rightmost glass part and leftmost poured part.
        tinfo = self.get_boxes_info()
        box_glass_x = tinfo['glass_right'][0]
        box_poured_x = tinfo['poured_left'][0]
        assert box_glass_x < box_poured_x, f'{box_glass_x} vs {box_poured_x}'
        self.segm.reset(
                off_x=self.get_default_config()['glass']['fake_offset_x'],
                glass_x=box_glass_x,
                poured_x=box_poured_x,
        )
        # -------------- end of new stuff -------------- #

        return self._get_obs()

    def get_state(self):
        '''
        get the postion, velocity of flex particles, and postions of flex shapes.
        '''
        particle_pos = pyflex.get_positions()
        particle_vel = pyflex.get_velocities()
        shape_position = pyflex.get_shape_states()
        return {'particle_pos': particle_pos, 'particle_vel': particle_vel, 'shape_pos': shape_position,
                'glass_x': self.glass_x, 'glass_y': self.glass_y,
                'glass_z': self.glass_z,
                'glass_rotation_z': self.glass_rotation_z,
                'glass_rotation_y': self.glass_rotation_y, 'glass_rotation_x': self.glass_rotation_x,
                'glass_states': self.glass_states, 'poured_glass_states': self.poured_glass_states,
                'glass_params': self.glass_params, 'config_id': self.current_config_id}

    def set_state(self, state_dic):
        '''
        set the postion, velocity of flex particles, and postions of flex shapes.
        '''
        pyflex.set_positions(state_dic["particle_pos"])
        pyflex.set_velocities(state_dic["particle_vel"])
        pyflex.set_shape_states(state_dic["shape_pos"])
        self.glass_x = state_dic['glass_x']
        self.glass_z = state_dic['glass_z']
        self.glass_y = state_dic['glass_y']
        self.glass_rotation_z = state_dic['glass_rotation_z']
        self.glass_rotation_x = state_dic['glass_rotation_x']
        self.glass_rotation_y = state_dic['glass_rotation_y']
        self.glass_states = state_dic['glass_states']
        self.poured_glass_states = state_dic['poured_glass_states']
        for _ in range(5):
            pyflex.step()

    def initialize_camera(self):
        """I used the indicated default_camera to get a sideways view for the paper."""
        if not self.real_world_scale:
            self.camera_params = {
                # The camera we normally use.
                'default_camera': {'pos': np.array([1.2, 1.5, 0.1]),
                                'angle': np.array([0.45 * np.pi, -60 / 180. * np.pi, 0]),
                                'width': self.camera_width,
                                'height': self.camera_height},
                # A good view for the 1st config in the cache we use.
                #'default_camera': {'pos': np.array([0.80, 0.85, 1.10]),
                #                 'angle': np.array([15.*DEG_TO_RAD,
                #                                    -30.*DEG_TO_RAD,
                #                                    0.*DEG_TO_RAD]),
                #                 'width': self.camera_width,
                #                 'height': self.camera_height},
                # A good view for seed 0 if we generate it and want a sidways view.
                # (I did this to generate a sideways flow figure.)
                #'default_camera': {'pos': np.array([0.60, 0.80, 1.10]),
                #                 'angle': np.array([15.*DEG_TO_RAD,
                #                                    -30.*DEG_TO_RAD,
                #                                    0.*DEG_TO_RAD]),
                #                 'width': self.camera_width,
                #                 'height': self.camera_height},
                'corl2020': {'pos': np.array([1.4, 1.5, 0.1]),
                                'angle': np.array([0.45 * np.pi, -60 / 180. * np.pi, 0]),
                                'width': self.camera_width,
                                'height': self.camera_height},
                'cam_2d': {'pos': np.array([0.5, .7, 4.]),
                        'angle': np.array([0, 0, 0.]),
                        'width': self.camera_width,
                        'height': self.camera_height},
                'sideways': {'pos': np.array([0.0, 0.30, 0.90]),
                             'angle': np.array([0., 0., 0.]),
                             'width': self.camera_width,
                             'height': self.camera_height},
            }
        else:
            self.camera_params = {
                'default_camera': {'pos': np.array([0.4, 0.5, 0.05]),
                                'angle': np.array([0.45 * np.pi, -60 / 180. * np.pi, 0]),
                                'width': self.camera_width,
                                'height': self.camera_height},
            }

    def set_poured_glass_params(self, config):
        params = config

        self.glass_distance = params['glass_distance']
        self.poured_border = params['poured_border']
        self.poured_height = params['poured_height']

        fluid_radis = self.fluid_params['radius'] * self.fluid_params['rest_dis_coef']
        if not self.real_world_scale:
            self.poured_glass_dis_x = self.fluid_params['dim_x'] * fluid_radis + 0.07  # glass floor length
            self.poured_glass_dis_z = self.fluid_params['dim_z'] * fluid_radis + 0.07  # glass width
        else:
            self.poured_glass_dis_x = self.fluid_params['dim_x'] * fluid_radis + 0.014 # glass floor length
            self.poured_glass_dis_z = self.fluid_params['dim_z'] * fluid_radis + 0.014  # glass width

        params['poured_glass_dis_x'] = self.poured_glass_dis_x
        params['poured_glass_dis_z'] = self.poured_glass_dis_z
        params['poured_glass_x_center'] = self.x_center + params['glass_distance']

        self.glass_params.update(params)

    def set_pouring_glass_params(self, config):
        params = config

        self.border = params['border']
        self.height = params['height']

        fluid_radis = self.fluid_params['radius'] * self.fluid_params['rest_dis_coef']
        if not self.real_world_scale:
            self.glass_dis_x = self.fluid_params['dim_x'] * fluid_radis + 0.1  # glass floor length
            self.glass_dis_z = self.fluid_params['dim_z'] * fluid_radis + 0.1  # glass width
        else:
            self.glass_dis_x = self.fluid_params['dim_x'] * fluid_radis + 0.02  # glass floor length
            self.glass_dis_z = self.fluid_params['dim_z'] * fluid_radis + 0.02 # glass width

        params['glass_dis_x'] = self.glass_dis_x
        params['glass_dis_z'] = self.glass_dis_z
        params['glass_x_center'] = self.x_center

        self.glass_params = params

    def set_scene(self, config, states=None, create_only=False):
        """Construct the pouring water scene, from `generate_env_variations`.

        New additions: add fake tool box (not the target) and have it mimic the tool
        so that we can get segmentation and track its pixels for point clouds.
        """
        super().set_scene(config)

        # compute glass params, see documentation above for more.
        if states is None:
            self.set_pouring_glass_params(config["glass"])
            self.set_poured_glass_params(config["glass"])
        else:
            glass_params = states['glass_params']
            self.border = glass_params['border']
            self.height = glass_params['height']
            self.glass_dis_x = glass_params['glass_dis_x']
            self.glass_dis_z = glass_params['glass_dis_z']
            self.glass_distance = glass_params['glass_distance']
            self.poured_border = glass_params['poured_border']
            self.poured_height = glass_params['poured_height']
            self.poured_glass_dis_x = glass_params['poured_glass_dis_x']
            self.poured_glass_dis_z = glass_params['poured_glass_dis_z']
            self.glass_params = glass_params

        # create pouring glass & poured glass (and a fake new tool)
        self.create_glass(self.glass_dis_x, self.glass_dis_z, self.height, self.border)
        self.create_glass(self.poured_glass_dis_x,
                          self.poured_glass_dis_z,
                          self.poured_height,
                          self.poured_border)
        if self.use_fake_tool:
            self.create_glass(self.glass_dis_x, self.glass_dis_z, self.height, self.border)

        # move pouring glass to be at ground

        self.glass_states = self.init_glass_state(
                x=self.x_center,
                y=0,
                glass_dis_x=self.glass_dis_x,
                glass_dis_z=self.glass_dis_z,
                height=self.height,
                border=self.border)

        # move poured glass to be at ground
        self.poured_glass_states = self.init_glass_state(
                x=self.x_center + self.glass_distance,
                y=0,
                glass_dis_x=self.poured_glass_dis_x,
                glass_dis_z=self.poured_glass_dis_z,
                height=self.poured_height,
                border=self.poured_border,
                block=self.real_world_scale)

        # move the fake tool mimicing pouring glass be offset by some x value.
        if self.use_fake_tool:
            self.fake_glass_states = self.init_glass_state(
                    x=self.x_center + config['glass']['fake_offset_x'],
                    y=0,
                    glass_dis_x=self.glass_dis_x,
                    glass_dis_z=self.glass_dis_z,
                    height=self.height,
                    border=self.border)
        else:
            self.fake_glass_states = None

        # Set pyflex values.
        self.set_shape_states(
                self.glass_states,
                self.poured_glass_states,
                self.fake_glass_states,
        )

        # record glass floor center x, y, and rotation
        self.glass_x = self.x_center
        if self.action_mode in ['rotation_bottom', 'translation_axis_angle']:
            self.glass_y = 0
        elif self.action_mode == 'rotation_top':
            self.glass_y = 0.5 * self.border + self.height
        self.glass_z = 0
        self.glass_rotation_z = 0
        self.glass_rotation_y = 0
        self.glass_rotation_x = 0

        # only create the glass and water, without setting their states
        # this is only used in the pourwater amount env.
        if create_only:
            return

        # no cached init states passed in
        if states is None:
            fluid_pos = np.ones((self.particle_num, self.dim_position))

            # move water all inside the glass
            fluid_radius = self.fluid_params['radius'] * self.fluid_params['rest_dis_coef']
            # fluid_dis = np.array([1.2 * fluid_radius, fluid_radius * 0.45, 1.2 * fluid_radius])
            fluid_dis = np.array([1.0 * fluid_radius, fluid_radius * 0.5, 1.0 * fluid_radius])
            lower_x = self.glass_params['glass_x_center'] - self.glass_params['glass_dis_x'] / 2. + self.glass_params['border']
            lower_z = -self.glass_params['glass_dis_z'] / 2 + self.glass_params['border']
            lower_y = self.glass_params['border']
            if self.action_mode in ['sawyer', 'franka']:
                lower_y += 0.56  # NOTE: robotics table
            lower = np.array([lower_x, lower_y, lower_z])
            cnt = 0
            rx = int(self.fluid_params['dim_x'] * 1)
            ry = int(self.fluid_params['dim_y'] * 1)
            rz = int(self.fluid_params['dim_z'] / 1)
            for x in range(rx):
                for y in range(ry):
                    for z in range(rz):
                        fluid_pos[cnt][:3] = lower + np.array([x, y, z]) * fluid_dis  # + np.random.rand() * 0.01
                        cnt += 1

            pyflex.set_positions(fluid_pos)
            print("stablize water!")
            for _ in range(100):
                pyflex.step()
                # pyflex.render()

            state_dic = self.get_state()
            water_state = state_dic['particle_pos'].reshape((-1, self.dim_position))
            in_glass = self.in_glass(water_state, self.glass_states, self.border, self.height)
            not_in_glass = 1 - in_glass
            not_total_num = np.sum(not_in_glass)

            while not_total_num > 0:
                max_height_now = np.max(water_state[:, 1])
                fluid_dis = np.array([1.0 * fluid_radius, fluid_radius * 1, 1.0 * fluid_radius])
                lower_x = self.glass_params['glass_x_center'] - self.glass_params['glass_dis_x'] / 4
                lower_z = -self.glass_params['glass_dis_z'] / 4
                lower_y = max_height_now
                lower = np.array([lower_x, lower_y, lower_z])
                cnt = 0
                dim_x = config['fluid']['dim_x']
                dim_z = config['fluid']['dim_z']
                for w_idx in range(len(water_state)):
                    if not in_glass[w_idx]:
                        water_state[w_idx][:3] = lower + fluid_dis * np.array([cnt % dim_x, cnt // (dim_x * dim_z), (cnt // dim_x) % dim_z])
                        cnt += 1

                pyflex.set_positions(water_state)
                for _ in range(40):
                    pyflex.step()
                    # pyflex.render()

                state_dic = self.get_state()
                water_state = state_dic['particle_pos'].reshape((-1, self.dim_position))
                in_glass = self.in_glass(water_state, self.glass_states, self.border, self.height)
                not_in_glass = 1 - in_glass
                not_total_num = np.sum(not_in_glass)



            for _ in range(50):
                pyflex.step()
                # pyflex.render()

            init_height = np.random.rand() * 0.1 + 0.05
            init_z = np.random.uniform(-0.2, 0.2)
            init_y_axis_angle = np.random.uniform(-np.pi / 4, np. pi / 4)
            init_x_axis_angle = np.random.uniform(-np.pi / 8, np. pi / 8)

            for i in range(60):
                self._step(np.array([0, init_height / 40, init_z / 40, 0, 0, 0]))
                # pyflex.render()

            for i in range(40):
                self._step(np.array([0, 0, 0, init_x_axis_angle / 40, init_y_axis_angle / 40, 0]))
                # pyflex.render()

            for _ in range(50):
                pyflex.step()
                # pyflex.render()

            print(f'num particles: {pyflex.get_n_particles()}')
        else:  # set to passed-in cached init states
            self.set_state(states)

    def _get_obs(self):
        """Return observation.

        05/30/2022, support segmented point cloud observations. Earlier,
            the point cloud was only the pyflex positions of water particles.
            Note that `point_cloud` returns an (N,d) array, not a flattened
            vector (as it does in the open-source SoftGym code).
        05/31/2022: now with tool flow, and some adjustments to make this as
            consistent as possible with Mixed Media code.
        """
        def get_water_info():
            # For CoRL Rebuttal. I think we should use the pyflex shapes with the
            # amount of water in the poured glass vs controlled glass.
            # For 6d, unlike 3d, don't worry about the controlled cup.
            # But maybe we should just use this info anyway?
            water_state = pyflex.get_positions().reshape([-1, 4])
            in_poured_glass = self.in_glass(
                    water_state, self.poured_glass_states, self.poured_border, self.poured_height)
            in_control_glass = self.in_glass(
                    water_state, self.glass_states, self.border, self.height)
            in_poured_glass = float(np.sum(in_poured_glass)) / len(water_state)
            in_control_glass = float(np.sum(in_control_glass)) / len(water_state)
            state = np.array([in_poured_glass, in_control_glass])
            return state

        def get_keypoints():
            # Return stuff here if we think we need this for the rotations and
            # SVD to work properly (e.g., for rescaling). Hopefully we will not
            # need such information, though. Maybe shape info is all we need?
            shape_state = pyflex.get_shape_states().reshape((-1,14))
            return shape_state

        def get_segm_img(with_depth=False):
            # Get RGB(D) images, possibly from multiple viewpoints. Then segment.
            images_dict = self.segm.query_images()
            segm_img = self.segm.segment(images=images_dict)
            if with_depth:
                return segm_img, images_dict['depth']
            return segm_img

        def get_pointcloud_array():
            # Will have 'ground truth' water info, but not for the two boxes.
            # Again must be called after `self.segm.segment()` which we can do
            # by calling `get_segm_img()` here.
            pc = self.segm.get_pointclouds()
            pc_tool  = pc['box_tool']  # box we control for pouring
            pc_targ  = pc['box_targ']  # fixed target box to get water in
            pc_water = pc['water']     # directly from pyflex positions

            # For water, filter such points that are clearly out of bounds.
            oob_x = np.logical_or(pc_water[:,0] < self.oob_x[0],
                                  pc_water[:,0] > self.oob_x[1])
            oob_z = np.logical_or(pc_water[:,2] < self.oob_z[0],
                                  pc_water[:,2] > self.oob_z[1])
            in_bounds = ~np.logical_or(oob_x, oob_z)
            pc_water = pc_water[in_bounds]

            # Also handle one-hot classes with subsampling (tool, targ), etc.
            # If subsampling PC _and_ we have tool flow, we should subsample
            # tool flow (so the tool pts coincide with the PC array's tool pts).
            n1, n2, n3 = len(pc_tool), len(pc_targ), len(pc_water)
            n_pts = n1 + n2 + n3

            pc_array = np.zeros((max(n_pts, self.max_pts), self.pc_point_dim))
            pc_array[           :n1, :3] = pc_tool
            pc_array[           :n1,  3] = 1.
            pc_array[      n1:n1+n2, :3] = pc_targ
            pc_array[      n1:n1+n2,  4] = 1.
            pc_array[n1+n2:n1+n2+n3, :3] = pc_water
            pc_array[n1+n2:n1+n2+n3,  5] = 1.

            # For simplicity, order idxs so all tool points are first, etc.
            # Should save the idxs we used to inform the tool flow subsampling.
            idxs = np.arange(n_pts)  # including if we don't need to subsample
            if n_pts > self.max_pts:
                idxs = np.sort( np.random.permutation(n_pts)[:self.max_pts] )
                pc_array = pc_array[idxs, :]
            self.segm.set_subsampling_tool_flow(idxs, n_tool=len(pc_tool))
            return pc_array

        def get_tool_flow():
            # I don't think we've been updating a `self.tool_state` so just
            # get it on the fly from pyflex? Might as well give all the shapes?
            tool_state = pyflex.get_shape_states().reshape((-1,14))
            return self.segm.get_tool_flow(tool_state)

        if self.observation_mode == 'cam_rgb':
            return self.get_image(self.camera_width, self.camera_height)
        elif self.observation_mode == 'cam_rgbd':
            # 08/19/2022: now RGBD for (width,height,4)-sized image. NOTE(daniel):
            # careful, RGB images in [0,255] (uint8) so check for any division by
            # 255 at runtime. Depth has values in [0,x] where x is 'SoftGym units',
            # usually x is about 1. Depth is (width,height) --> (width,height,1).
            cam_rgb = self.get_image(self.camera_width, self.camera_height)
            _, depth_img = get_segm_img(with_depth=True)
            rgbd_img = np.concatenate(
                (cam_rgb.astype(np.float32), depth_img[...,None]), axis=2)
            return rgbd_img
        elif self.observation_mode == 'depth_img':
            _, depth_img = get_segm_img(with_depth=True)
            w,h = depth_img.shape
            new_img = np.zeros([w,h,3])
            new_img[:,:,0] = depth_img
            new_img[:,:,1] = depth_img
            new_img[:,:,2] = depth_img
            return new_img
        elif self.observation_mode == 'depth_segm':
            # See documentation in Pouring for 3D motions
            cam_rgb = self.get_image(self.camera_width, self.camera_height)
            segm_img, depth_img = get_segm_img(with_depth=True)
            mask_tool  = segm_img[:,:,1]  # binary {0,255} image, TOOL (cup we control)
            mask_targ  = segm_img[:,:,2]  # binary {0,255} image, TARG (fixed target cup)
            mask_water = segm_img[:,:,3]  # binary {0,255} image, WATER, might be overlap

            # Concatenate and form the image. Must do the same during BC!
            mask_tool  = mask_tool.astype(np.float32) / 255.0  # has only {0.0, 1.0}
            mask_targ  = mask_targ.astype(np.float32) / 255.0  # has only {0.0, 1,0}
            mask_water = mask_water.astype(np.float32) / 255.0  # has only {0.0, 1,0}
            depth_segm = np.concatenate(
                    (depth_img[...,None],
                     mask_tool[...,None],
                     mask_targ[...,None],
                     mask_water[...,None]), axis=2)

            # # Debugging.
            # import cv2, os
            # k = len([x for x in os.listdir('tmp') if 'depth_' in x and '.png' in x])
            # cv2.imwrite(f'tmp/rgb_{str(k).zfill(3)}.png', cam_rgb)
            # cv2.imwrite(f'tmp/depth_{str(k).zfill(3)}.png', (depth_segm[:,:,0] / np.max(depth_segm[:,:,0]) * 255).astype(np.uint8))
            # cv2.imwrite(f'tmp/mask_tool_{str(k).zfill(3)}.png', (depth_segm[:,:,1] * 255).astype(np.uint8))
            # cv2.imwrite(f'tmp/mask_targ_{str(k).zfill(3)}.png', (depth_segm[:,:,2] * 255).astype(np.uint8))
            # cv2.imwrite(f'tmp/mask_water_{str(k).zfill(3)}.png', (depth_segm[:,:,3] * 255).astype(np.uint8))
            return depth_segm
        elif self.observation_mode == 'rgb_segm_masks':
            # 08/25/2022: RGB (not D) but with segmentation masks. 6-channel image.
            cam_rgb = self.get_image(self.camera_width, self.camera_height)
            segm_img, _ = get_segm_img(with_depth=True)
            mask_tool  = segm_img[:,:,1]  # binary {0,255} image, TOOL (cup we control)
            mask_targ  = segm_img[:,:,2]  # binary {0,255} image, TARG (fixed target cup)
            mask_water = segm_img[:,:,3]  # binary {0,255} image, WATER, might be overlap

            # New here, divide image by 255.0 so we get values in [0,1] to align w/others.
            # I think this will make it easier as compared to keeping them on diff scales.
            cam_rgb = cam_rgb.astype(np.float32) / 255.0

            # Concatenate and form the image. Must do the same during BC!
            mask_tool  = mask_tool.astype(np.float32) / 255.0  # has only {0.0, 1.0}
            mask_targ  = mask_targ.astype(np.float32) / 255.0  # has only {0.0, 1,0}
            mask_water = mask_water.astype(np.float32) / 255.0  # has only {0.0, 1,0}
            rgb_segm_masks = np.concatenate(
                    (cam_rgb,
                     mask_tool[...,None],
                     mask_targ[...,None],
                     mask_water[...,None]), axis=2)
            return rgb_segm_masks
        elif self.observation_mode == 'rgbd_segm_masks':
            # 08/25/2022: RGB-D, but with segmentation masks. 7-channel image.
            cam_rgb = self.get_image(self.camera_width, self.camera_height)
            segm_img, depth_img = get_segm_img(with_depth=True)
            mask_tool  = segm_img[:,:,1]  # binary {0,255} image, TOOL (cup we control)
            mask_targ  = segm_img[:,:,2]  # binary {0,255} image, TARG (fixed target cup)
            mask_water = segm_img[:,:,3]  # binary {0,255} image, WATER, might be overlap

            # NOTE! Divide image by 255.0 so we get values in [0,1] to align w/others.
            # I think this will make it easier as compared to keeping them on diff scales.
            cam_rgb = cam_rgb.astype(np.float32) / 255.0

            # Concatenate and form the image. Must do the same during BC!
            mask_tool  = mask_tool.astype(np.float32) / 255.0  # has only {0.0, 1.0}
            mask_targ  = mask_targ.astype(np.float32) / 255.0  # has only {0.0, 1,0}
            mask_water = mask_water.astype(np.float32) / 255.0  # has only {0.0, 1,0}
            rgbd_segm_masks = np.concatenate(
                    (cam_rgb,
                     depth_img[...,None],
                     mask_tool[...,None],
                     mask_targ[...,None],
                     mask_water[...,None]), axis=2)
            return rgbd_segm_masks
        elif self.observation_mode == 'point_cloud':
            _ = get_segm_img()
            return get_pointcloud_array()
        elif self.observation_mode == 'state':
            # New for the CoRL rebuttal, state-based policy baseline.
            # I think water in controlled cup is just an approximation.
            # The formula I think will only work if we have a non-rotated cup.
            pose_info = get_keypoints()
            water_info = get_water_info()
            #boxes = np.concatenate((pose_info[:,:3], pose_info[:,6:10]), axis=1).flatten()
            #boxes = np.concatenate(
            #    (pose_info[:,:3], pose_info[:,6:10]), axis=1).flatten()

            # 1 and 6 are left wall for both, can commpare with right wall.
            # I'm taking distances of all these box centers and taking a norm, this
            # should give all info we need.
            box_dims = np.array([
                np.linalg.norm(pose_info[0,:3] - pose_info[1,:3]),  # this will give height info
                np.linalg.norm(pose_info[1,:3] - pose_info[2,:3]),
                np.linalg.norm(pose_info[3,:3] - pose_info[4,:3]),
                np.linalg.norm(pose_info[5,:3] - pose_info[6,:3]),  # this will give height info
                np.linalg.norm(pose_info[6,:3] - pose_info[7,:3]),
                np.linalg.norm(pose_info[8,:3] - pose_info[9,:3]),
            ])

            state = np.concatenate((
                water_info,         # 2D water info
                pose_info[0,:3],    # 3D position
                pose_info[0,6:10],  # 4D quaternion (this is actually the same across boxes)
                pose_info[5,:3],    # 3D position, target box
                pose_info[5,6:10],  # 4D quaternion, target box (well this remains fixed...)
                box_dims,
            ))
            return state
        elif 'key_point' in self.observation_mode:
            pos = np.empty(0, dtype=np.float)

            water_state = pyflex.get_positions().reshape([-1, 4])
            in_poured_glass = self.in_glass(water_state, self.poured_glass_states, self.poured_border, self.poured_height)
            in_control_glass = self.in_glass(water_state, self.glass_states, self.border, self.height)
            in_poured_glass = float(np.sum(in_poured_glass)) / len(water_state)
            in_control_glass = float(np.sum(in_control_glass)) / len(water_state)

            if self.observation_mode == 'key_point_1':
                cup_state = np.array([self.glass_x, self.glass_y, self.glass_rotation_z, self.glass_dis_x, self.glass_dis_z, self.height,
                                    self.glass_distance + self.glass_x, self.poured_height, self.poured_glass_dis_x, self.poured_glass_dis_z,
                                    self._get_current_water_height(), in_poured_glass, in_control_glass])
            elif self.observation_mode == 'key_point_2': # no water information, but has height
                cup_state = np.array([self.glass_x, self.glass_y, self.glass_rotation_z, self.glass_dis_x, self.glass_dis_z, self.height,
                                    self.glass_distance, self.poured_height, self.poured_glass_dis_x, self.poured_glass_dis_z])
            # elif self.observation_mode == 'key_point_3': # has water information, no pouring height.
            #     cup_state = np.array([self.glass_x, self.glass_y, self.glass_rotation_z, self.glass_dis_x, self.glass_dis_z,
            #                         self.glass_distance, self.poured_height, self.poured_glass_dis_x, self.poured_glass_dis_z])
            elif self.observation_mode == 'key_point_3': # no water information, no pouring height. This is all information for KPConv.
                cup_state = np.array([self.glass_x, self.glass_y, self.glass_rotation_z, self.glass_dis_x, self.glass_dis_z,
                                    self.glass_distance, self.poured_height, self.poured_glass_dis_x, self.poured_glass_dis_z])
            elif self.observation_mode == 'key_point_4': # no water information, no pouring height. use relative x and relative y.
                cup_state = np.array([self.glass_distance - self.glass_x, self.poured_height - self.glass_y, self.glass_rotation_z,
                                        self.glass_dis_x, self.glass_dis_z,
                                        self.poured_glass_dis_x, self.poured_glass_dis_z])


            return np.hstack([pos, cup_state]).flatten()
        elif self.observation_mode in ['rim_interpolation', 'rim_interpolation_normalize',
            'rim_graph', 'rim_graph_hierarchy', 'rim_interpolation_flatten', 'rim_interpolation_and_state']:
            shape_states = pyflex.get_shape_states().reshape((-1, 14))

            pouring_right_wall_center = shape_states[2][:3]
            pouring_left_wall_center = shape_states[1][:3]
            rotation = self.glass_rotation_z

            # build the corner of the front wall of the control glass
            c_corner1_relative_cord = np.array([-self.border / 2., self.height / 2., self.glass_dis_z / 2])
            c_corner1_real = rotate_rigid_object(center=pouring_right_wall_center, axis=np.array([0, 0, -1]), angle=rotation,
                                                relative=c_corner1_relative_cord)

            c_corner2_relative_cord = np.array([-self.border / 2., self.height / 2., -self.glass_dis_z / 2])
            c_corner2_real = rotate_rigid_object(center=pouring_right_wall_center, axis=np.array([0, 0, -1]), angle=rotation,
                                                relative=c_corner2_relative_cord)

            c_corner3_relative_cord = np.array([self.border / 2., self.height / 2., -self.glass_dis_z / 2])
            c_corner3_real = rotate_rigid_object(center=pouring_left_wall_center, axis=np.array([0, 0, -1]), angle=rotation,
                                                relative=c_corner3_relative_cord)

            c_corner4_relative_cord = np.array([self.border / 2., self.height / 2., self.glass_dis_z / 2])
            c_corner4_real = rotate_rigid_object(center=pouring_left_wall_center, axis=np.array([0, 0, -1]), angle=rotation,
                                             relative=c_corner4_relative_cord)

            target_right_wall_center = shape_states[2 + 5][:3]
            target_left_wall_center = shape_states[1 + 5][:3]
            # target cup does not need rotation
            t_corner_1 = target_right_wall_center + np.array([-self.poured_border / 2., self.poured_height / 2., self.poured_glass_dis_z / 2])
            t_corner_2 = target_right_wall_center + np.array([-self.poured_border / 2., self.poured_height / 2., -self.poured_glass_dis_z / 2])
            t_corner_3 = target_left_wall_center + np.array([self.poured_border / 2., self.poured_height / 2., -self.poured_glass_dis_z / 2])
            t_corner_4 = target_left_wall_center + np.array([self.poured_border / 2., self.poured_height / 2., self.poured_glass_dis_z / 2])


            '''
            corner3 -x- corner 2
              |             |
              z             z
              |             |
            corner4 -x- corner 1
            '''
            cornerss = [
                [c_corner1_real, c_corner2_real, c_corner3_real, c_corner4_real],
                [t_corner_1, t_corner_2, t_corner_3, t_corner_4]
            ]
            z_segments = [
                math.floor(self.glass_dis_z / self.current_config['fluid']['radius']),
                math.floor(self.poured_glass_dis_z / self.current_config['fluid']['radius'])
            ]
            x_segments = [
                math.floor(self.glass_dis_x / self.current_config['fluid']['radius']),
                math.floor(self.poured_glass_dis_x / self.current_config['fluid']['radius']),
            ]

            all_points = []
            pouring_rim_point_num = 0
            for glass_idx in range(2):
                corners = cornerss[glass_idx]
                z_segment, x_segment = z_segments[glass_idx], x_segments[glass_idx]
                for idx in range(4):
                    # print(idx)
                    start_corner = corners[idx]
                    end_corner = corners[(idx+1)%4]
                    segment_num = z_segment if idx % 2 == 0 else x_segment
                    distance = (end_corner - start_corner) / segment_num
                    # print(start_corner)
                    add_points = np.array([distance * i for i in range(1, segment_num)])
                    interpolated_points = start_corner + add_points
                    all_points.append(start_corner)
                    all_points += list(interpolated_points)
                    if glass_idx == 0:
                        pouring_rim_point_num += (1 + len(interpolated_points))

            # add a root node that is the center of the rim
            if self.observation_mode == 'rim_graph_hierarchy':
                all_points.append(np.mean(all_points[:pouring_rim_point_num], axis=0))
                all_points.append(np.mean(all_points[pouring_rim_point_num:-1], axis=0))

            all_points = np.vstack(all_points)
            all_point_num = len(all_points)

            if self.observation_mode == 'rim_interpolation':
                # 0: controlled cup; 1: target cup.
                cup_type_indicator = np.zeros((len(all_points), 2))
                cup_type_indicator[:pouring_rim_point_num, 0] = 1
                cup_type_indicator[pouring_rim_point_num:, 1] = 1
                all_points = np.concatenate([all_points, cup_type_indicator], axis=1)

                return all_points

            if self.observation_mode == 'rim_interpolation_and_state':
                cup_type_indicator = np.zeros((len(all_points), 2))
                cup_type_indicator[:pouring_rim_point_num, 0] = 1
                cup_type_indicator[pouring_rim_point_num:, 1] = 1
                all_points = np.concatenate([all_points, cup_type_indicator], axis=1)

                reduced_state = self.get_cup_reduced_state()

                return all_points, reduced_state

            if self.observation_mode == 'rim_interpolation_normalize':
                # 0: controlled cup; 1: target cup.
                all_points[:, 0] -= np.mean(all_points[:, 0])
                all_points[:, 1] -= np.mean(all_points[:, 1])
                all_points[:, 2] -= np.mean(all_points[:, 2])
                cup_type_indicator = np.zeros((len(all_points), 2))
                cup_type_indicator[:pouring_rim_point_num, 0] = 1
                cup_type_indicator[pouring_rim_point_num:, 1] = 1
                all_points = np.concatenate([all_points, cup_type_indicator], axis=1)

                return all_points

            elif self.observation_mode == 'rim_interpolation_flatten':
                cup_type_indicator = np.zeros((len(all_points), 3))
                cup_type_indicator[:pouring_rim_point_num, 0] = 1
                cup_type_indicator[pouring_rim_point_num:, 1] = 1
                all_points = np.concatenate([all_points, cup_type_indicator], axis=1)

                obs = np.zeros((self.max_point_num, 6))
                obs[:all_point_num] = all_points.copy()
                obs[all_point_num:, 2] = 1 # fake point

                return obs.flatten()

            elif self.observation_mode in ['rim_graph', 'rim_graph_hierarchy']:
                # 0: controlled cup; 1: target cup.
                cup_type_indicator = np.ones((len(all_points), 1))
                cup_type_indicator[:pouring_rim_point_num, 0] = 0
                if self.observation_mode == 'rim_graph_hierarchy':
                    cup_type_indicator[-2] = 0 # rim 1 center
                    cup_type_indicator[-1] = 1 # rim 2 center

                all_points = np.concatenate([all_points, cup_type_indicator], axis=1)

                # construct edge
                edge_attribute = []

                # rim 1 graph
                source_node = [i for i in range(pouring_rim_point_num)]
                target_node = [i+1 for i in range(pouring_rim_point_num)]
                target_node[-1] = 0
                source_tmp = copy.deepcopy(source_node)
                source_node += target_node
                target_node += source_tmp
                edge_attribute.append(np.zeros(len(source_node)))

                if self.observation_mode == 'rim_graph':
                    # rim 2 graph
                    source_node_2 = [i for i in range(pouring_rim_point_num, all_point_num)]
                    target_node_2 = [i+1 for i in range(pouring_rim_point_num, all_point_num)]
                    target_node_2[-1] = pouring_rim_point_num
                    source_tmp = copy.deepcopy(source_node_2)
                    source_node_2 += target_node_2
                    target_node_2 += source_tmp
                    edge_attribute.append(np.ones(len(source_node_2)))

                    # rim 1 -> rim 2 graph
                    source_node_inter = []
                    target_node_inter = []
                    target_rim_point_num = all_point_num - pouring_rim_point_num
                    for i in range(pouring_rim_point_num):
                        source_node_inter += [i for _ in range(target_rim_point_num)]
                        target_node_inter += [j for j in range(pouring_rim_point_num, all_point_num)]
                    edge_attribute.append(np.ones(len(source_node_inter)) * 2)

                    source_nodes = source_node + source_node_2 + source_node_inter
                    target_nodes = target_node + target_node_2 + target_node_inter
                    edge_idx = np.vstack([source_nodes, target_nodes])

                elif self.observation_mode == 'rim_graph_hierarchy':
                    # print("adding hierarchy graph connection!")
                    target_rim_point_num = all_point_num - pouring_rim_point_num - 2

                    # rim 2 graph
                    source_node_2 = [i for i in range(pouring_rim_point_num, all_point_num - 2)]
                    target_node_2 = [i+1 for i in range(pouring_rim_point_num, all_point_num - 2)]
                    target_node_2[-1] = pouring_rim_point_num
                    source_tmp = copy.deepcopy(source_node_2)
                    source_node_2 += target_node_2
                    target_node_2 += source_tmp
                    edge_attribute.append(np.ones(len(source_node_2)))

                    # glass 1 rim <-> glass 1 rim center
                    source_node_1_to_center = [i for i in range(pouring_rim_point_num)]
                    target_node_1_to_center = [all_point_num - 2  for _ in range(pouring_rim_point_num)]
                    source_tmp = copy.deepcopy(source_node_1_to_center)
                    source_node_1_to_center += target_node_1_to_center
                    target_node_1_to_center += source_tmp
                    edge_attribute.append(np.zeros(len(source_node_1_to_center)))

                    # glass 2 rim <-> glass 2 rim center
                    source_node_2_to_center = [i for i in range(pouring_rim_point_num, all_point_num - 2)]
                    target_node_2_to_center = [all_point_num - 1  for _ in range(target_rim_point_num)]
                    source_tmp = copy.deepcopy(source_node_2_to_center)
                    source_node_2_to_center += target_node_2_to_center
                    target_node_2_to_center += source_tmp
                    edge_attribute.append(np.ones(len(source_node_2_to_center)))

                    # glass 1 rim center -> glass 2 rim center
                    source_node_inter = [all_point_num - 2]
                    target_node_inter = [all_point_num - 1]
                    edge_attribute.append(np.ones(len(source_node_inter)) * 2)

                    source_nodes = source_node + source_node_2 + source_node_1_to_center + source_node_2_to_center + source_node_inter
                    target_nodes = target_node + target_node_2 + target_node_1_to_center + target_node_2_to_center + target_node_inter
                    edge_idx = np.vstack([source_nodes, target_nodes])

                # print("edge_idx shape: ", edge_idx.shape)
                edge_attribute = np.hstack(edge_attribute).reshape((-1, 1))

                if True:
                    print("all_points shape: ", all_points.shape)
                    import matplotlib.pyplot as plt
                    from mpl_toolkits.mplot3d import Axes3D
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    for p in all_points:
                        ax.scatter(p[0], p[2], p[1])

                    # print(len(source_node))
                    # print(pouring_rim_point_num)
                    # for i in range(len(source_node) + target_rim_point_num * 2, len(source_nodes)):
                    #     print(edge_idx[0][i], edge_idx[1][i])
                    #     ax.plot([all_points[edge_idx[0][i]][0], all_points[edge_idx[1][i]][0]],
                    #         [all_points[edge_idx[0][i]][2], all_points[edge_idx[1][i]][2]],
                    #         [all_points[edge_idx[0][i]][1], all_points[edge_idx[1][i]][1]])

                    for i in range(len(source_nodes)): #, len(source_node) + len(source_node_2)):
                        # print(source_node_2[i], target_node_2[i])
                        ax.plot([all_points[source_nodes[i]][0], all_points[target_nodes[i]][0]],
                            [all_points[source_nodes[i]][2], all_points[target_nodes[i]][2]],
                            [all_points[source_nodes[i]][1], all_points[target_nodes[i]][1]])

                    ax.set_xlim(-0.3, 0.6)
                    ax.set_ylim(-0.3, 0.6)
                    ax.set_zlim(0, 0.6)

                    ax.set_xlabel('X Label')
                    ax.set_ylabel('Y Label')
                    ax.set_zlabel('Z Label')
                    plt.show()

                return all_points, edge_idx, edge_attribute
        elif self.observation_mode == 'combo':
            # Get a variety of observations for BC testing. The return values
            # are in the same order as MM usage, to be consistent. We can also
            # use keypoints here if we need (e.g., for rescaling coordinates)?
            # For now the 'keypts' are literally the pyflex shape states.
            keypts = get_keypoints()
            img_rgb = self.get_image(self.camera_width, self.camera_height)
            segm_img, depth_img = get_segm_img(with_depth=True)
            tool_flow = get_tool_flow()
            pc_array = get_pointcloud_array()
            water_info = get_water_info()
            return (keypts, img_rgb, segm_img, pc_array, tool_flow, depth_img, water_info)
        else:
            raise NotImplementedError

    def compute_reward(self, obs=None, action=None, set_prev_reward=False):
        """
        The reward is computed as the fraction of water in the poured glass.
        NOTE: the obs and action params are made here to be compatiable with the MultiTask env wrapper.
        """
        state_dic = self.get_state()
        water_state = state_dic['particle_pos'].reshape((-1, self.dim_position))
        water_num = len(water_state)

        in_poured_glass = self.in_glass(water_state, self.poured_glass_states, self.poured_border, self.poured_height)
        # in_control_glass = self.in_glass(water_state, self.glass_states, self.border, self.height)
        good_water = in_poured_glass # * (1 - in_control_glass)
        good_water_num = np.sum(good_water)

        reward = float(good_water_num) / water_num
        return reward

    def _get_info(self):
        """# Duplicate of the compute reward function!

        NOTE(daniel) what is a good performance? At least 75% of particles?
        Just tweak this threshold, or use `performance` for the raw fraction.
        Also I'd return the angle of the demonstrator in case we care about
        resetting to neutral.
        """
        state_dic = self.get_state()
        water_state = state_dic['particle_pos'].reshape((-1, self.dim_position))
        water_num = len(water_state)

        in_poured_glass = self.in_glass(water_state, self.poured_glass_states, self.poured_border, self.poured_height)
        # in_control_glass = self.in_glass(water_state, self.glass_states, self.border, self.height)
        good_water = in_poured_glass # * (1 - in_control_glass)
        good_water_num = np.sum(good_water)

        performance = float(good_water_num) / water_num
        performance_init = performance if self.performance_init is None else self.performance_init  # Use the original performance
        normalized = (performance - performance_init) / (self.reward_max - performance_init)

        info = {
            'done': performance > 0.75,
            'normalized_performance': normalized,
            'performance': performance,
            'glass_rotation_z': self.glass_rotation_z,
            'glass_rotation_x': self.glass_rotation_x,
            'glass_rotation_y': self.glass_rotation_y,
        }
        return info

    def get_random_or_alg_action(self):
        """For external code, either an algorithmic action or a random action.
        See `MixedMediaEnv.get_random_or_alg_action()` for more.

        Returns
        -------
        (action, denorm_action): These ARE repeated according to action repetition,
            so it executes this 8 (or however action repeat is) times.
        """
        if self.alg_policy is not None:
            action = self._test_policy()
        else:
            action = self.action_space.sample()

        # Adding some noise to the actual action if desired.
        # Not being used now but could consider testing later.
        if self.act_noise is not None and self.act_noise > 0:
            for k in range(len(action)):
                action[k] += np.random.uniform(low=-self.act_noise, high=self.act_noise)

        # Clip action.
        action = np.clip(action, a_min=self.action_space.low, a_max=self.action_space.high)

        # As we observed from the initializer code in SoftAgent, we 'de-normalize'
        # the action because we later 'normalize' it.
        # TODO(daniel): check these bounds with rotations and flow.
        lb = self.action_space.low
        ub = self.action_space.high
        denorm_action = (action - lb) / ((ub - lb) * 0.5) - 1.0

        # Return both types of actions.
        return (action, denorm_action)

    def _step(self, action):
        """Take a step with the given action, performs collision checking!

        If using the action with (x,y,theta) control:

            action: np.ndarray of dim 1x3, (x, y, theta).
                (x, y) specifies the floor center coordinate
                theta specifies the rotation.

            bounds:
                self.action_space.low
                    array([-0.01 , -0.01 , -0.015], dtype=float32)
                self.action_space.high
                    array([0.01 , 0.01 , 0.015], dtype=float32)

        Collision checking must be done heuristically. Assumes that if a
        collision happens, we just don't update the glass (not realistic
        but fine for SoftGym purposes).
        """
        if self.action_mode == 'rotation_bottom':
            assert len(action) == 3, action
            move = action[:2]
            rotate = action[2]
        elif self.action_mode == 'translation_axis_angle':
            # NOTE(daniel): this is hacky because we are not actually doing
            # 'axis-angle' but more like just taking the 6D vector and
            # assuming we can clear-out components. We should change this
            # if we really want more flexibility with the rotations.
            assert len(action) == 6, action
            move = action[:3]
            rotate = action[[3, 4, 5]]  # assume the last two
        else:
            raise NotImplementedError(self.action_mode)

        # Make action as increasement (i.e., a delta change), clip its range
        move = np.clip(move,
                       a_min=self.action_space.low[0],
                       a_max=self.action_space.high[0])
        rotate = np.clip(rotate,
                         a_min=self.action_space.low[-1],
                         a_max=self.action_space.high[-1])
        dx, dy, dz, dtheta_x, dtheta_y, dtheta_z = move[0], move[1], move[2], rotate[0], rotate[1], rotate[2]
        x = self.glass_x + dx
        y = self.glass_y + dy
        z = self.glass_z + dz
        theta_x = self.glass_rotation_x + dtheta_x
        theta_z = self.glass_rotation_z + dtheta_z
        theta_y = self.glass_rotation_y + dtheta_y

        # check if the movement of the pouring glass collide with the poured glass.
        # the action only take effects if there is no collision
        new_states = self.rotate_glass(self.glass_states, x, y, z, theta_z, theta_y, theta_x)
        # NOTE: collision checking code is not updated for full 6d motions!
        if (not self.judge_glass_collide(new_states, theta_z, theta_y, theta_x) and
                self.above_floor(new_states, theta_z, theta_y, theta_x)):
            self.glass_states = new_states
            self.glass_x, self.glass_y, self.glass_z = x, y, z
            self.glass_rotation_z, self.glass_rotation_y, self.glass_rotation_x = theta_z, theta_y, theta_x
        else:
            # invalid move, old state becomes the same as the current state
            self.glass_states[:, 3:6] = self.glass_states[:, :3].copy()
            self.glass_states[:, 10:] = self.glass_states[:, 6:10].copy()

        # Handle fake tool.
        if self.use_fake_tool:
            _offs_x = self.get_default_config()['glass']['fake_offset_x']
            self.fake_glass_states = np.copy(self.glass_states)
            self.fake_glass_states[:, 3] += _offs_x
        else:
            self.fake_glass_states = None

        # pyflex takes a step to update the glass and the water fluid
        self.set_shape_states(
                self.glass_states,
                self.poured_glass_states,
                self.fake_glass_states,
        )
        pyflex.step(render=self.render_img)
        self.inner_step += 1

    def create_glass(self, glass_dis_x, glass_dis_z, height, border):
        """
        the glass is a box, with each wall of it being a very thin box in Flex.
        each wall of the real box is represented by a box object in Flex with really small thickness (determined by the param border)
        dis_x: the length of the glass
        dis_z: the width of the glass
        height: the height of the glass.
        border: the thickness of the glass wall.

        the halfEdge determines the center point of each wall.
        Note: this is merely setting the length of each dimension of the wall, but not the actual position of them.
        That's why left and right walls have exactly the same params, and so do front and back walls.
        """
        center = np.array([0., 0., 0.])
        quat = quatFromAxisAngle([0, 0, -1.], 0.)
        boxes = []

        # floor
        halfEdge = np.array([glass_dis_x / 2. + border, border / 2., glass_dis_z / 2. + border])
        boxes.append([halfEdge, center, quat])

        # left wall
        halfEdge = np.array([border / 2., (height) / 2., glass_dis_z / 2. + border])
        boxes.append([halfEdge, center, quat])

        # right wall
        boxes.append([halfEdge, center, quat])

        # back wall
        halfEdge = np.array([(glass_dis_x) / 2., (height) / 2., border / 2.])
        boxes.append([halfEdge, center, quat])

        # front wall
        boxes.append([halfEdge, center, quat])

        for i in range(len(boxes)):
            halfEdge = boxes[i][0]
            center = boxes[i][1]
            quat = boxes[i][2]
            pyflex.add_box(halfEdge, center, quat)

        return boxes

    def rotate_glass(self, prev_states, x, y, z=0, theta_z=0, theta_y=0., theta_x=0.):
        """Given the previous states of the glass, rotate it with angle theta.

        update the states of the 5 boxes that form the box: floor, left/right wall, back/front wall.
        rotate the glass, where the center point is the center of the floor or the top.

        state:
        0-3: current (x, y, z) coordinate of the center point
        3-6: previous (x, y, z) coordinate of the center point
        6-10: current quat
        10-14: previous quat

        NOTE(daniel):
        Given updated `theta` we first modify the quaternion accordingly. However, we
        also have to modify the positions, and must do that for the 5 walls separately.
        The rotation will rotate the bottom wall by default, but by doing this rotation,
        it necessarily must adjust the positioning of all the other walls since those 4
        walls must rotate _and_ move as the rotation center is not at their origins.

        ALSO, `theta` should describe the full rotation (not a _delta_). So, if the delta
        rotation is 0, then `theta` should be the same across multiple time steps. This will
        keep the item orientation fixed (good) because, cleverly, we define `relative_coord`
        to be wrt the default box initialization. This way, each time `rotate_rigid_object`
        is called, it 'assumes' the box starts from the default config, and continually gets
        the right position for that rotation. (It's a subtle point.) Furthermore, the xyz
        movement still gets applied correctly as that's updated in `rotate_center`.

        The axis_ang should be [0,0,-1] or [0,0,1], as long as it's consistently handled
        throughout the env, it should be OK. It follows the right-hand rule as expected.

        Parameters
        ----------
        prev_state: need this since a 'state' requires previous information.
        x, y, theta: the current configuration of the glass, we continually track this
            info, and if there's no collision, should update its class attributes with
            the updated values.
        """
        axis_ang_z = np.array([0., 0., -1.])
        axis_ang_y = np.array([0., 1., 0.])
        axis_ang_x = np.array([1, 0., 0.])
        axis_angle_z = axis_ang_z * theta_z
        axis_angle_y = axis_ang_y * theta_y
        axis_angle_x = axis_ang_x * theta_x
        Rotation = Rot.from_rotvec(axis_angle_x) * Rot.from_rotvec(axis_angle_y) * Rot.from_rotvec(axis_angle_z)
        final_quaternion = Rotation.as_quat()
        # NOTE: I need to handle the quaternion here super carefully. Let's always assume that I first rotate around the z axis, and then the y axis, and then x axis
        quat_curr = final_quaternion

        dis_x, dis_z = self.glass_dis_x, self.glass_dis_z
        border = self.border
        x_center = x

        # states of 5 walls, first populate previous info
        states = np.zeros((5, self.dim_shape_state))
        for i in range(5):
            states[i][3:6] = prev_states[i][:3]
            states[i][10:] = prev_states[i][6:10]

        # rotation center is the floor center
        rotate_center = np.array([x_center, y, z])

        if (self.action_mode in ['rotation_bottom', 'translation_axis_angle']):
            # floor: center position does not change
            states[0, :3] = np.array([x_center, y, z])

            # left wall: center must move right and move down.
            relative_coord = np.array([-(dis_x + border) / 2., (self.height + border) / 2., 0.])
            states[1, :3] = rotate_rigid_object(center=rotate_center, axis=axis_ang_z, angle=theta_z, relative=relative_coord)
            relative_coord = states[1, :3] - rotate_center
            states[1, :3] = rotate_rigid_object(center=rotate_center, axis=axis_ang_y, angle=theta_y, relative=relative_coord)
            relative_coord = states[1, :3] - rotate_center
            states[1, :3] = rotate_rigid_object(center=rotate_center, axis=axis_ang_x, angle=theta_x, relative=relative_coord)

            # right wall
            relative_coord = np.array([(dis_x + border) / 2., (self.height + border) / 2., 0.])
            states[2, :3] = rotate_rigid_object(center=rotate_center, axis=axis_ang_z, angle=theta_z, relative=relative_coord)
            relative_coord = states[2, :3] - rotate_center
            states[2, :3] = rotate_rigid_object(center=rotate_center, axis=axis_ang_y, angle=theta_y, relative=relative_coord)
            relative_coord = states[2, :3] - rotate_center
            states[2, :3] = rotate_rigid_object(center=rotate_center, axis=axis_ang_x, angle=theta_x, relative=relative_coord)

            # back wall
            relative_coord = np.array([0, (self.height + border) / 2., -(dis_z + border) / 2.])
            states[3, :3] = rotate_rigid_object(center=rotate_center, axis=axis_ang_z, angle=theta_z, relative=relative_coord)
            relative_coord = states[3, :3] - rotate_center
            states[3, :3] = rotate_rigid_object(center=rotate_center, axis=axis_ang_y, angle=theta_y, relative=relative_coord)
            relative_coord = states[3, :3] - rotate_center
            states[3, :3] = rotate_rigid_object(center=rotate_center, axis=axis_ang_x, angle=theta_x, relative=relative_coord)

            # front wall
            relative_coord = np.array([0, (self.height + border) / 2., (dis_z + border) / 2.])
            states[4, :3] = rotate_rigid_object(center=rotate_center, axis=axis_ang_z, angle=theta_z, relative=relative_coord)
            relative_coord = states[4, :3] - rotate_center
            states[4, :3] = rotate_rigid_object(center=rotate_center, axis=axis_ang_y, angle=theta_y, relative=relative_coord)
            relative_coord = states[4, :3] - rotate_center
            states[4, :3] = rotate_rigid_object(center=rotate_center, axis=axis_ang_x, angle=theta_x, relative=relative_coord)

        elif self.action_mode == 'rotation_top':
            # floor
            relative_coord = np.array([0, -self.height - self.border / 2., 0.])
            states[0, :3] = rotate_rigid_object(center=rotate_center, axis=np.array([0, 0, -1]), angle=theta_z, relative=relative_coord)

            # left wall
            relative_coord = np.array([-(dis_x + border) / 2., -self.height / 2., 0.])
            states[1, :3] = rotate_rigid_object(center=rotate_center, axis=np.array([0, 0, -1]), angle=theta_z, relative=relative_coord)

            # right wall
            relative_coord = np.array([(dis_x + border) / 2., -self.height / 2., 0.])
            states[2, :3] = rotate_rigid_object(center=rotate_center, axis=np.array([0, 0, -1]), angle=theta_z, relative=relative_coord)

            # back wall
            relative_coord = np.array([0, -self.height / 2., -(dis_z + border) / 2.])
            states[3, :3] = rotate_rigid_object(center=rotate_center, axis=np.array([0, 0, -1]), angle=theta_z, relative=relative_coord)

            # front wall
            relative_coord = np.array([0, -self.height / 2., (dis_z + border) / 2.])
            states[4, :3] = rotate_rigid_object(center=rotate_center, axis=np.array([0, 0, -1]), angle=theta_z, relative=relative_coord)

        states[:, 6:10] = quat_curr
        return states

    def init_glass_state(self, x, y, glass_dis_x, glass_dis_z, height, border, block=False):
        '''
        set the initial state of the glass.
        '''
        dis_x, dis_z = glass_dis_x, glass_dis_z
        x_center, y_curr, y_last = x, y, 0.
        quat = quatFromAxisAngle([0, 0, -1.], 0.)

        # states of 5 walls
        states = np.zeros((5, self.dim_shape_state))

        if not block:
            # floor
            states[0, :3] = np.array([x_center, y_curr, 0.])
            states[0, 3:6] = np.array([x_center, y_last, 0.])

            # left wall
            states[1, :3] = np.array([x_center - (dis_x + border) / 2., (height + border) / 2. + y_curr, 0.])
            states[1, 3:6] = np.array([x_center - (dis_x + border) / 2., (height + border) / 2. + y_last, 0.])

            # right wall
            states[2, :3] = np.array([x_center + (dis_x + border) / 2., (height + border) / 2. + y_curr, 0.])
            states[2, 3:6] = np.array([x_center + (dis_x + border) / 2., (height + border) / 2. + y_last, 0.])

            # back wall
            states[3, :3] = np.array([x_center, (height + border) / 2. + y_curr, -(dis_z + border) / 2.])
            states[3, 3:6] = np.array([x_center, (height + border) / 2. + y_last, -(dis_z + border) / 2.])

            # front wall
            states[4, :3] = np.array([x_center, (height + border) / 2. + y_curr, (dis_z + border) / 2.])
            states[4, 3:6] = np.array([x_center, (height + border) / 2. + y_last, (dis_z + border) / 2.])
        else:
            # floor
            states[0, :3] = np.array([x_center, y_curr, 0.])
            states[0, 3:6] = np.array([x_center, y_last, 0.])

            # left wall
            states[1, :3] = np.array([x_center - (dis_x + border) / 2., (height) / 2. + y_curr, 0.])
            states[1, 3:6] = np.array([x_center - (dis_x + border) / 2., (height) / 2. + y_last, 0.])

            # right wall
            states[2, :3] = np.array([x_center + (dis_x + border) / 2., (height) / 2. + y_curr, 0.])
            states[2, 3:6] = np.array([x_center + (dis_x + border) / 2., (height) / 2. + y_last, 0.])

            # back wall
            states[3, :3] = np.array([x_center, (height) / 2. + y_curr, -(dis_z + border) / 2.])
            states[3, 3:6] = np.array([x_center, (height) / 2. + y_last, -(dis_z + border) / 2.])

            # front wall
            states[4, :3] = np.array([x_center, (height) / 2. + y_curr, (dis_z + border) / 2.])
            states[4, 3:6] = np.array([x_center, (height) / 2. + y_last, (dis_z + border) / 2.])

        states[:, 6:10] = quat
        states[:, 10:] = quat
        return states

    def set_shape_states(self, glass_states, poured_glass_states,
            fake_glass_states=None):
        """Set shape states of all glasses."""
        if fake_glass_states is not None:
            all_states = np.concatenate(
                    (glass_states, poured_glass_states, fake_glass_states), axis=0)
        else:
            all_states = np.concatenate(
                    (glass_states, poured_glass_states), axis=0)
        pyflex.set_shape_states(all_states)

    def in_glass(self, water, glass_states, border, height):
        '''
        judge whether a water particle is in the poured glass
        water: [x, y, z, 1/m] water particle state.
        '''

        # floor, left, right, back, front
        # state:
        # 0-3: current (x, y, z) coordinate of the center point
        # 3-6: previous (x, y, z) coordinate of the center point
        # 6-10: current quat
        # 10-14: previous quat
        x_lower = glass_states[1][0] - border / 2.
        x_upper = glass_states[2][0] + border / 2.
        z_lower = glass_states[3][2] - border / 2.
        z_upper = glass_states[4][2] + border / 2
        y_lower = glass_states[0][1] - border / 2.
        y_upper = glass_states[0][1] + height + border / 2.
        x, y, z = water[:, 0], water[:, 1], water[:, 2]

        res = (x >= x_lower) * (x <= x_upper) * (y >= y_lower) * (y <= y_upper) * (z >= z_lower) * (z <= z_upper)
        return res

    def get_ellipisode(self, cup_states):
        pouring_floor_wall_center = cup_states[0, :3]
        pouring_left_wall_center = cup_states[1, :3]
        pouring_right_wall_center = cup_states[2, :3]
        pouring_back_wall_center = cup_states[3, :3]
        pouring_center = (pouring_left_wall_center + pouring_right_wall_center) / 2.
        axis_1 = pouring_right_wall_center - pouring_center
        axis_2 = pouring_center - pouring_floor_wall_center
        axis_3 = pouring_back_wall_center - pouring_center
        len1, len2, len3 = np.linalg.norm(axis_1), np.linalg.norm(axis_2), np.linalg.norm(axis_3)
        len1 += self.border
        len2 += self.border
        len3 += self.border
        axis_1 = axis_1 / len1
        axis_2 = axis_2 / len2
        axis_3 = axis_3 / len3

        mat1 = np.zeros((3, 3))
        mat1[:, 0] = axis_1
        mat1[:, 1] = axis_2
        mat1[:, 2] = axis_3
        mat2 = np.zeros((3, 3))
        mat2[0, 0] = 1 / len1 ** 2
        mat2[1, 1] = 1 / len2 ** 2
        mat2[2, 2] = 1 / len3 ** 2
        A = mat1 @ mat2 @ mat1.T

        return pouring_center, A

    def judge_glass_collide(self, new_states, rotation_z, rotation_y=0, rotation_x=0):
        '''
        fit an ellipsode to the glass and judge if the ellipsode intersect with each other.
        '''
        pouring_cup_states = new_states
        target_cup_states = self.poured_glass_states
        a, A = self.get_ellipisode(pouring_cup_states)
        b, B = self.get_ellipisode(target_cup_states)
        res = ellipsoid_intersection_test(np.linalg.inv(A), np.linalg.inv(B), a, b)

        return res

    def above_floor(self, states, rotation_z, rotation_y=0, rotation_x=0):
        '''
        judge all the floors are above the ground.
        '''

        floor_center = states[0][:3]
        corner_relative = [
            np.array([self.glass_dis_x / 2., -self.border / 2., self.glass_dis_z / 2.]),
            np.array([self.glass_dis_x / 2., -self.border / 2., -self.glass_dis_z / 2.]),
            np.array([-self.glass_dis_x / 2., -self.border / 2., self.glass_dis_z / 2.]),
            np.array([-self.glass_dis_x / 2., -self.border / 2., -self.glass_dis_z / 2.]),

            np.array([self.glass_dis_x / 2., self.border / 2. + self.height, self.glass_dis_z / 2.]),
            np.array([self.glass_dis_x / 2., self.border / 2. + self.height, -self.glass_dis_z / 2.]),
            np.array([-self.glass_dis_x / 2., self.border / 2. + self.height, self.glass_dis_z / 2.]),
            np.array([-self.glass_dis_x / 2., self.border / 2. + self.height, -self.glass_dis_z / 2.]),
        ]

        for idx in range(len(corner_relative)):
            corner_rel = corner_relative[idx]
            corner_real = rotate_rigid_object(center=floor_center, axis=np.array([0, 0, -1]), angle=rotation_z, relative=corner_rel)
            corner_rel = corner_real - floor_center
            corner_real = rotate_rigid_object(center=floor_center, axis=np.array([0, 1, 0]), angle=rotation_y, relative=corner_rel)
            corner_rel = corner_real - floor_center
            corner_real = rotate_rigid_object(center=floor_center, axis=np.array([1, 0, 0]), angle=rotation_x, relative=corner_rel)
            if corner_real[1] < - self.border:
                return False

        return True

    ####################################
    # NOTE(daniel): various state info #
    ####################################

    def get_cup_reduced_state(self):
        """Seems to be for rim interpolation and state."""
        redu_state = np.array([self.glass_distance - self.glass_x,
                               self.poured_height - self.glass_y,
                               self.glass_rotation_z,
                               self.glass_rotation_x,
                               self.glass_rotation_y,
                               self.glass_dis_x,
                               self.glass_dis_z,
                               self.poured_glass_dis_x,
                               self.poured_glass_dis_z])
        return redu_state

    def get_boxes_info(self):
        """Shape of states is (10,14), first 5 are for the cup to control.
        FYI the 'left' walls mean the one with lowest x-coordinate.
        """
        states = pyflex.get_shape_states().reshape((-1,14))
        tool_info = {
            'glass_floor':  states[0,:], # first 2 should be glass_x,glass_y
            'glass_left':   states[1,:],
            'glass_right':  states[2,:],
            'glass_back':   states[3,:],
            'glass_front':  states[4,:],
            'poured_floor': states[5,:],
            'poured_left':  states[6,:],
            'poured_right': states[7,:],
            'poured_back':  states[8,:],
            'poured_front': states[9,:],
        }
        return tool_info

    def get_glass_center(self):
        """Get glass center if we assume the cube that forms it."""
        tinfo = self.get_boxes_info()
        w1 = tinfo['glass_left']
        w2 = tinfo['glass_right']
        w3 = tinfo['glass_back']
        w4 = tinfo['glass_front']
        center = np.array([
            np.mean([w1[0], w2[0], w3[0], w4[0]]),
            np.mean([w1[1], w2[1], w3[1], w4[1]]),
            np.mean([w1[2], w2[2], w3[2], w4[2]]),
        ])
        return center

    def get_poured_center(self):
        """Get poured center if we assume the cube that forms it."""
        tinfo = self.get_boxes_info()
        w1 = tinfo['poured_left']
        w2 = tinfo['poured_right']
        w3 = tinfo['poured_back']
        w4 = tinfo['poured_front']
        center = np.array([
            np.mean([w1[0], w2[0], w3[0], w4[0]]),
            np.mean([w1[1], w2[1], w3[1], w4[1]]),
            np.mean([w1[2], w2[2], w3[2], w4[2]]),
        ])
        return center

    ######################################
    # NOTE(daniel): algorithmic policies #
    ######################################

    def _print_glass(self):
        print(f'Glass: ({self.glass_x:0.3f}, {self.glass_y:0.3f}, {self.glass_z:0.3f}, {self.glass_rotation_z:0.3f}, {self.glass_rotation_y:0.3f}, {self.glass_rotation_x:0.3f})')

    def set_alg_policy(self, alg_policy):
        if alg_policy is not None and alg_policy != 'None':
            self.alg_policy = alg_policy

    def _test_policy(self):
        """Pick which one we want to use. Should not use for real experiments.

        BUT, it's good to have these to benchmark the algorithmic policy.
        NOTE(daniel): due to action repetition, `self.time_step` repeats (8 by default).
        This means that when we interpret 'step', it's going to be 100 (that's the usual
        time) times 8, so `self.inner_step` goes from 0 to 799.

        No input right now, but we pass in the _inner_ step to the algorithmic.
        policies. We could also pass in the action if we want to fine-tune that?
        """
        if self.alg_policy == 'noop':
            # Nothing, useful if testing sliders from GUI.
            action = np.zeros((self.action_direct_dim,))
        elif self.alg_policy == 'pw_algo_v01':
            # 05/30/2022 basic version.
            assert self.action_mode == 'rotation_bottom', self.action_mode
            assert self.action_repeat == 8, self.action_repeat
            action = self._pw_algo_v01(self.inner_step)
        elif self.alg_policy == 'pw_algo_v02':
            # 05/31/2022 similar version with translation_axis_angle
            assert self.action_mode == 'translation_axis_angle', self.action_mode
            assert self.action_repeat == 8, self.action_repeat
            action = self._pw_algo_v02(self.inner_step)
        else:
            raise ValueError(self.alg_policy)
        return action

    def _pw_algo_v01(self, step):
        """Basic version, move to box, raise, rotate."""
        return self.AlgPolicyCls.get_action(step, version=1)

    def _pw_algo_v02(self, step):
        """Basic version, move to box, raise, rotate."""
        return self.AlgPolicyCls.get_action(step, version=2)


class PWAlgorithmicPolicy():
    """Separate class might be easier since it may involve tracking state.

    The dx, dy, dz are what we actually insert into the action vector, for
    `env.step(action)`. We supply an action (e.g., with `get_action()`) and
    the env will step though it based on `action_repeat`.
    """

    def __init__(self, env):
        self.env = env
        self.state = 0
        self._print = False
        self._degree_delta = 4.0

    def reset(self):
        self.state = 0
        self._prior = np.array([0., 0., 0.])

    def get_action(self, step, version):
        """Careful about the action bounds.

        We might want to record the prior state info to check if we have moved
        at all (if not, then we know a collision has occurred, or we have finished
        the demonstration). We can also use the `version` count if desired.

        NOTE: glass_center is not the same as the glass_x and glass_y, since the
        glass_center is the center of the full cube, while the glass_x and y are
        for the bottom floor only.
        """
        dx, dy, drot = 0., 0., 0.
        dx_rot, dy_rot = 0, 0
        dz = 0
        collision = False

        # Can check curr env with the prior, if same, then collision happened.
        # NOTE: will need to change if we introduce a resting state! So this
        # is also misleading since at the end, the tool isn't moving.
        same_x = self._prior[0] == self.env.glass_x
        same_y = self._prior[1] == self.env.glass_y
        same_rot = self._prior[2] == self.env.glass_rotation_z
        collision = same_x and same_y and same_rot

        # Set the prior info for the next call to `get_action()`.
        self._prior[0] = self.env.glass_x
        self._prior[1] = self.env.glass_y
        self._prior[2] = self.env.glass_rotation_z

        # Get info about the tool box which does the pouring.
        tinfo = self.env.get_boxes_info()
        glass_center = self.env.get_glass_center()
        poured_center = self.env.get_poured_center()
        glass_floor = tinfo['glass_floor'][:3]
        glass_x = glass_center[0]

        if self.state == 0: # NOTE from yufei, also revert the initial random rotation along x and y axis
            # Move glass towards target, move up if collisions. We could
            # also move up earlier to get some more state coverage. Edit:
            # this might now lift earlier based on this threshold.
            x_thresh = (glass_x + (self.env.glass_dis_x / 2.)) + \
                (self.env.height)

            advance = np.abs(x_thresh - poured_center[0]) < 0.15

            if advance:
                dx = 0.0
            elif x_thresh < poured_center[0]:
                dx = 0.002
            else:
                dx = -0.001

            rot_x_finished = np.abs(self.env.glass_rotation_x) < DEG_TO_RAD * 2
            rot_y_finished = np.abs(self.env.glass_rotation_y) < DEG_TO_RAD * 2
            z_finished = np.abs(self.env.glass_z) < 0.02
            if not rot_x_finished:
                dx_rot = (self._degree_delta / self.env.action_repeat) * DEG_TO_RAD * -np.sign(self.env.glass_rotation_x)
            if not rot_y_finished:
                dy_rot = (self._degree_delta /self.env.action_repeat) * DEG_TO_RAD * -np.sign(self.env.glass_rotation_y)
            if not z_finished:
                dz = 0.002 * np.sign(self.env.glass_z) * -1


            if advance and rot_x_finished and rot_y_finished and z_finished:
                self.state += 1

            # In case there's some noise
            # if self.env.glass_y > 0:
            #     dy = -0.001

            # Only if we do not advance earlier.
            if (not advance) and (step > 0) and collision:
                self.state += 1

        elif self.state == 1:
            # Raise the glass until a little bit above height.
            dy = 0.003
            if self.env.glass_y >= (1.8 * self.env.poured_height):
                self.state += 1
        elif self.state == 2:
            # Move a little more over the target box (or backwards).
            # Note: if box is very tall, need to move back. Want the
            # box opening to be roughly about the poured glass center.
            # Edit: actually this seems to be working though it tends
            # to move the box backwards a bit. Hmm... maybe use this
            # for the state 0 threshld instead?
            x_thresh = (glass_x + (self.env.glass_dis_x / 2.)) + \
                (self.env.height)

            advance = np.abs(x_thresh - poured_center[0]) < 0.02

            if advance:
                dx = 0.0
            elif x_thresh < poured_center[0]:
                dx = 0.001
            else:
                dx = -0.001

            rot_x_finished = np.abs(self.env.glass_rotation_x) < DEG_TO_RAD * 1.1
            rot_y_finished = np.abs(self.env.glass_rotation_y) < DEG_TO_RAD * 1.1
            z_finished = np.abs(self.env.glass_z) < 0.02
            if not rot_x_finished:
                dx_rot = (self._degree_delta / 2 / self.env.action_repeat) * DEG_TO_RAD * -np.sign(self.env.glass_rotation_x)
            if not rot_y_finished:
                dy_rot = (self._degree_delta / 2 / self.env.action_repeat) * DEG_TO_RAD * -np.sign(self.env.glass_rotation_y)
            if not z_finished:
                dz = 0.001 * np.sign(self.env.glass_z) * -1


            if advance and rot_x_finished and rot_y_finished and z_finished:
                self.state += 1

            # Originally I thought this would be reasonable.
            #dx = 0.003
            #x_thresh = 0.90 * tinfo['poured_left'][0] + \
            #           0.10 * poured_center[0]
            #if tinfo['glass_right'][0] >= x_thresh:
            #    self.state += 1
        elif self.state == 3:
            # Rotate! If collision, move up a bit more.
            if collision:
                dy = 0.003
            else:
                drot = (self._degree_delta / self.env.action_repeat) * DEG_TO_RAD
            # Also if the rotation is a bit big, might increase x as well.
            if 90.0 <= (self.env.glass_rotation_z * RAD_TO_DEG) <= 95.0:
                dx = 0.001
            # At this point, should be done.
            if (self.env.glass_rotation_z * RAD_TO_DEG) >= 120.0:
                self.state += 1
        elif self.state == 4:
            # Rotate back to neutral! (This is more of a qualitative thing,
            # can choose to ignore this if desired.)
            if (self.env.glass_rotation_z * RAD_TO_DEG) <= 5.0:
                self.state += 1
            elif collision:
                dy = 0.003
            else:
                drot = -(self._degree_delta / self.env.action_repeat) * DEG_TO_RAD
        elif self.state == 5:
            # Resting state.
            pass

        # Form the action vector.
        if version == 1:
            action = np.array([dx, dy, drot])
        elif version == 2:
            # We treat this as movement in xy, and positive rotation about
            # the _negative_ z-axis.
            action = np.array([dx, dy, dz, dx_rot, dy_rot, drot])
        else:
            raise NotImplementedError(version)

        if self._print:
            glass_rot = self.env.glass_rotation_z * RAD_TO_DEG
            print((f'is {self.env.inner_step}, state {self.state}, '
                   f'action: {action} '
                   f'prior: {self._prior}, now: {glass_floor}; '
                   f'rotdeg: {glass_rot:0.3f}'))
        return action


if __name__ == '__main__':
    from softgym.utils.visualization import save_numpy_as_gif
    import cv2
    from matplotlib import pyplot as plt

    env = PourWater6DEnv(
        action_mode='translation_axis_angle',
        num_variations=2000,
        headless=False,
        cached_states_path="PourWater6D_nVars_2000.pkl",
        # use_cached_states=False,
        observation_mode='combo',
        action_repeat=8,
        render_mode='fluid',
    )
    # exit()

    env.set_alg_policy("pw_algo_v02")

    for i in range(14, 15):
        images = []
        env.reset(config_id=i)
        for t in range(env.horizon):
            action, denorm_action = env.get_random_or_alg_action()
            obs, _, _, _ = env.step(action)
            images.append(env.get_image())
            depth = obs[-1]

            if t % 10 == 0:
                plt.imshow(depth)
                plt.show()

        # image = env.get_image()
        # print(image.shape)
        # images.append(image)
        # for t in range(20):
        #     env.step(np.array([0.002, 0.002, 0.002, 0, 0, 0]))
        #     image = env.get_image()
        #     images.append(image)

        # for t in range(50):
        #     env.step(np.array([0, 0, 0, np.deg2rad(0.2), np.deg2rad(0.2), np.deg2rad(0.2)]))
        #     image = env.get_image()
        #     images.append(image)

        # save_numpy_as_gif(np.array(images), f'{i}.gif')


    # all_image = np.concatenate(images, axis=1)
    # print(all_image.shape)
    # cv2.imwrite("tmp.png", all_image)
