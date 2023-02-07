import os
import pyflex
import numpy as np
from pyquaternion import Quaternion
from softgym.envs.flex_env import FlexEnv
from softgym.utils.misc import quatFromAxisAngle
from softgym.utils.camera_projections import get_matrix_world_to_camera
from softgym.utils.segmentation import Segmentation
from gym.spaces import Box, Tuple
DEG_TO_RAD = np.pi / 180.
RAD_TO_DEG = 180 / np.pi


class MixedMediaEnv(FlexEnv):
    """Adapted from `FluidEnv`, superclass to handle common MM functionality."""

    def __init__(self, observation_mode, action_mode, render_mode='particle', **kwargs):
        self.debug = False
        assert render_mode in ['particle', 'fluid']
        self.render_mode = 0 if render_mode == 'particle' else 1
        self.wall_num = 5  # number of glass walls. floor/left/right/front/back

        # Mainly to keep track of stuff we assign to later in code, mainly for general env
        # management, and the segmentation. There's also `particle_num` from the superclass.
        self.alg_policy = None  # Use to enforce algorithmic policy.
        self.rigid_idxs = None
        self.fluid_idxs = None
        self.n_fluid_particles = None
        self.n_shapes = None
        self.water_height_init = None
        self.glass_height_init = None
        self.matrix_world_to_camera = None
        self.sunk_height_cutoff = np.inf
        self.particle_to_item = None
        self.tool_vec_info = None
        self.tool_state = None
        self.tool_state_tip = None

        # PyFlex shape indices and tool init values.
        self.tool_idx = 0                       # don't change!
        self.tool_idx_fake = 1                  # don't change!
        self.tool_idx_tip = 2                   # don't change!
        self.wall_idxs = np.array([5,6,7,8,9])  # don't change!
        self.tool_init_x = -0.10
        self.tool_init_y =  0.28  # smaller = ladle is closer to water surface
        self.tool_init_z = -0.10

        # Stuff relevant to the reward and env termination. Assumes we retrieve ONE item!
        self.height_cutoff = 0.40
        self.prev_reward = 0
        self.reward_min = 0
        self.reward_max = 1
        self.performance_init = 0  # TODO(daniel)
        self.time_exceeded = 0  # How long sphere is currently above threshold
        self.time_exceeded_dist_1 = 0  # Same but for a distractor.
        self.time_cutoff = 10  # How long sphere needs to be there before success

        # FlexEnv init, then obs and action mode (I think this ordering is OK).
        self.action_mode = action_mode
        self.observation_mode = observation_mode
        super().__init__(**kwargs)

        # Choose observation mode. See `self._get_obs()` for documentation. The
        # combo one should not normally be used, mainly for BC so we can collect
        # as many different obs types as possible given the data.
        self.max_pts = 2000  # 1000-2500 seems standard. Yufei uses <=4000.
        if observation_mode == 'key_point':
            self.observation_space = Box(
                    low=np.array([-np.inf] * self.obs_dim_keypt),
                    high=np.array([np.inf] * self.obs_dim_keypt),
                    dtype=np.float32)
        if observation_mode == 'state':
            # 3d ball pos, 3d ladle tip, 3d ladle quaternion (using what is in pyflex
            # shapes for the tool, not the tip).
            dim = 10
            self.observation_space = Box(
                    low=np.array([-np.inf] * dim),
                    high=np.array([np.inf] * dim),
                    dtype=np.float32)
        elif observation_mode in ['point_cloud', 'point_cloud_gt_v01', 'point_cloud_gt_v02']:
            obs_dim = (self.max_pts, self.pc_point_dim)  # (N,d)
            self.observation_space = Box(
                    low=-np.inf, high=np.inf, shape=obs_dim, dtype=np.float32)
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
            # Turn to 3 channel image.
            self.observation_space = Box(
                    low=-np.inf, high=np.inf,
                    shape=(self.camera_height, self.camera_width, 3),
                    dtype=np.float32)
        elif observation_mode == 'depth_segm':
            # Design choice. 1st channel depth, 2nd and 3rd as tool and item.
            self.observation_space = Box(
                    low=-np.inf, high=np.inf,
                    shape=(self.camera_height, self.camera_width, 3),
                    dtype=np.float32)
        elif observation_mode == 'rgb_segm_masks':
            # Design choice. 5 channels: RGB, tool mask, ball mask.
            self.observation_space = Box(
                    low=-np.inf, high=np.inf,
                    shape=(self.camera_height, self.camera_width, 5),
                    dtype=np.float32)
        elif observation_mode == 'rgbd_segm_masks':
            # Design choice. 6 channels: RGBD, tool mask, ball mask.
            self.observation_space = Box(
                    low=-np.inf, high=np.inf,
                    shape=(self.camera_height, self.camera_width, 6),
                    dtype=np.float32)
        elif observation_mode == 'segm':
            self.observation_space = Box(
                    low=-np.inf, high=np.inf,
                    shape=(self.camera_height, self.camera_width, self.n_segm_classes),
                    dtype=np.float32)
        elif observation_mode == 'flow':
            # Just to get anything here, we (probably) don't call obs space shape?
            obs_dim = (self.max_pts, self.pc_point_dim)  # (N,d)
            box1 = Box(low=-np.inf, high=np.inf, shape=obs_dim, dtype=np.float32)
            box2 = Box(low=-np.inf, high=np.inf, shape=obs_dim, dtype=np.float32)
            self.observation_space = Tuple((box1, box2))
        elif observation_mode in ['combo', 'combo_gt_v01', 'combo_gt_v02']:
            # The combination of earlier stuff, mainly for Behavioral Cloning.
            # Once again this tuple here is just to get any value here.
            obs_dim = (self.max_pts, self.pc_point_dim)  # (N,d)
            box1 = Box(low=np.array([-np.inf] * self.obs_dim_keypt),
                    high=np.array([np.inf] * self.obs_dim_keypt),
                    dtype=np.float32)
            box2 = Box(low=-np.inf, high=np.inf,
                    shape=(self.camera_height, self.camera_width, 3),
                    dtype=np.float32)
            box3 = Box(low=-np.inf, high=np.inf,
                    shape=(self.camera_height, self.camera_width, self.n_segm_classes),
                    dtype=np.float32)
            box4 = Box(low=-np.inf, high=np.inf, shape=obs_dim, dtype=np.float32)
            box5 = Box(low=-np.inf, high=np.inf, shape=obs_dim, dtype=np.float32)
            self.observation_space = Tuple((box1, box2, box3, box4, box5))
        else:
            raise NotImplementedError()

        # Action space, should tune low/high ranges for per time step and global pos.
        # `action_low, action_high` are per time step bounds (for each action).
        # `action_low_b, action_high_b` are global positional constraints, but they
        # might not be strict enough for collision checking. WARNING: needs re-tuning
        # if we don't use the lower voxel corner of ladle, if we use rotations, etc.
        default_config = self.get_default_config()
        b = default_config['glass']['border']
        self.max_transl_axis_ang = None
        if action_mode == 'translation':
            # 3 DoFs, (deltax, deltay, deltaz). Remember, y points up.
            action_low  = np.array([-b*0.2, -b*0.2, -b*0.2])
            action_high = np.array([ b*0.2,  b*0.2,  b*0.2])
            self.action_space = Box(action_low, action_high, dtype=np.float32)
            self.action_low_b  = np.array([-0.25, -0.10, -0.25])
            self.action_high_b = np.array([ 0.25,  0.50,  0.25])
        elif action_mode == 'translation_onescale':
            # 3 DoFs, (deltax, deltay, deltaz). Remember, y points up.
            action_low  = np.array([-1, -1, -1])
            action_high = np.array([ 1,  1,  1])
            self.action_space = Box(action_low, action_high, dtype=np.float32)
            self.action_low_b  = np.array([-0.25, -0.10, -0.25])
            self.action_high_b = np.array([ 0.25,  0.50,  0.25])
            self.max_transl_axis_ang = b*0.2
        elif action_mode == 'translation_yrot':
            # Technically 4 DoFs: (deltax, deltay, deltaz, yrot).
            # Rotation ranges of (-0.04, 0.04) seem reasonable? That's for a
            # single axis (in radian) but I actually don't think we want this
            # since it means our system has to regress / predict 1 rotation
            # value and it can't do 6DoF? Also if we did the whole flow to pose,
            # we'd have to 'map' the pose to an angle about the y axis?
            # TODO(daniel) really should be removed.
            action_low  = np.array([-b*0.2, -b*0.2, -b*0.2, -0.04])
            action_high = np.array([ b*0.2,  b*0.2,  b*0.2,  0.04])
            self.action_space = Box(action_low, action_high, dtype=np.float32)
            self.action_low_b  = np.array([-0.25, -0.10, -0.25, -np.pi])
            self.action_high_b = np.array([ 0.25,  0.50,  0.25,  np.pi])
        elif action_mode == 'translation_axis_angle':
            # This is what we were using for the CoRL 2022 submission. Also I
            # think this is challenging as the magnitudes are +/- 0.004 for
            # translation, but +/-1 for rotation.
            action_low  = np.array([-b*0.2, -b*0.2, -b*0.2, -1, -1, -1])
            action_high = np.array([ b*0.2,  b*0.2,  b*0.2,  1,  1,  1])
            self.action_space = Box(action_low, action_high, dtype=np.float32)
            self.action_low_b  = np.array([-0.25, -0.10, -0.25, -1, -1, -1])
            self.action_high_b = np.array([ 0.25,  0.50,  0.25,  1,  1,  1])
            self.max_rot_axis_ang = (10. * DEG_TO_RAD) / self.action_repeat
        elif action_mode == 'translation_axis_angle_onescale':
            # Now try and follow what robosuite / Wenxuan was using? The policy has
            # a tanh to get output into (-1,1) so the bounds should be (-1,1) for all
            # action components. The translation should be divided by 0.004. Also the
            # rotation is: `action[3:] * max_rotation/np.sqrt(3)`. I think this will
            # be easier for a policy to learn via RL?
            action_low  = np.array([-1, -1, -1, -1, -1, -1])
            action_high = np.array([ 1,  1,  1,  1,  1,  1])
            self.action_space = Box(action_low, action_high, dtype=np.float32)
            self.action_low_b  = np.array([-0.25, -0.10, -0.25, -1, -1, -1])
            self.action_high_b = np.array([ 0.25,  0.50,  0.25,  1,  1,  1])
            self.max_transl_axis_ang = b*0.2
            self.max_rot_axis_ang = (10. * DEG_TO_RAD) / self.action_repeat
        elif action_mode == 'translation_quaternion':
            # Here, (deltax, deltay, deltaz, q1, q2, q3, q4) where now the last
            # part should output the change in rotations. Maybe define 1 as bounds?
            # To actually limit rotations, we would need something like SLERP to
            # lineraly interpolate between two quaternions? The goal is to enable
            # full generality, but an algorithmic policy might restrict itself?
            # TODO(daniel) have not tested.
            raise NotImplementedError()
        else:
            raise NotImplementedError(action_mode)

        # TODO(daniel) HACK! For now, assume y axis points _downwards_. If changing,
        # have to adjust algorithmic policies, etc. See the `_reset()`!
        self.curr_roty = None

        # Mainly for collision checking, should be compatible w/Eddie's changes.
        if self.tool_data in [2, 3, 4]:
            self.SPHERE_RAD = 0.076  # 'outer' sphere (the 'bowl' is thick)
            self.TIP_TO_CENTER_Y = 0.18
            self.collisions_sphere_rad = 0.050
            self.collisions_offset = 0.08  # TODO(daniel) is this necessary?
            self._init_ladle_bowl_vec = np.array(
                [0.0, -self.TIP_TO_CENTER_Y, self.SPHERE_RAD])
        else:
            raise ValueError(self.tool_data)

        # Use for sampling sphere points for 'ground truth' ball point clouds.
        self._sphere_radius_PCL = 0.030

        # Always have this as an option for handling tool rotation.
        self.segm = Segmentation(n_segm_classes=self.n_segm_classes,
                                 n_targs=self.n_targets,
                                 n_distr=self.n_distractors)

        # Initialize an 'Alg Policy' class if needed.
        self.AlgPolicyCls = AlgorithmicPolicy(env=self)

    def _reset(self):
        """Reset to initial state and return starting obs.

        NOTE(daniel): seems like a good spot to set the camera transformation stuff,
        plus segmentation stuff? Also, this gets called AFTER `set_scene()`. Assigns
        to the `self.tool_state_tip`, which we had to wait until we got depth info.
        """
        # Reset the point cloud dict and alg policy, if using point clouds.
        self.pcl_dict = {}
        self.AlgPolicyCls.reset()

        self.inner_step = 0
        self.performance_init = None
        info = self._get_info()
        self.performance_init = info['performance']
        pyflex.step(render=self.render_img)
        self.curr_roty = np.pi / 2.0

        # -------------- NOTE(daniel) new stuff -------------- #
        # Maybe we should always do this because it helps us get the tool ladle tip?
        cp = self.camera_params[self.camera_name]
        self.matrix_world_to_camera = get_matrix_world_to_camera(cp)
        self.segm.assign_other(self.rigid_idxs,
                               self.fluid_idxs,
                               self.water_height_init,
                               self.get_default_config()['tool']['off_x'],
                               self.get_default_config()['tool']['off_z'],
                               self.particle_to_item)
        self.segm.assign_camera(self.camera_params,
                                self.camera_name,
                                self.matrix_world_to_camera)

        # Get tool ladle tip (fake tool) vs the queried position, giving vector offset.
        # We can use to create a shape with the tip, then that shape moves w/the tool.
        # NOTE(daniel): assumes that we can see the ladle tip in the starting image!
        self.tool_vec_info = self.segm.get_tool_vec_offset()
        shape_states = pyflex.get_shape_states().reshape((-1,14))
        shape_states[self.tool_idx_tip, :3] = self.tool_vec_info['tool_tip']
        shape_states[self.tool_idx_tip, 3:6] = self.tool_vec_info['tool_tip']
        pyflex.set_shape_states(shape_states)
        shape_states = pyflex.get_shape_states().reshape((-1,14))
        self.tool_state_tip = np.reshape(shape_states[self.tool_idx_tip, :], (1,14))

        # Bells and whistles, or debugging stuff.
        assert self.sunk_height_cutoff < self.water_height_init
        #self._visualize_action_boundary()  # to debug action ranges
        #self._debug_diameter_bowl()  # to debug center of ladle

        # Init center of ladle bowl wrt coordinate frame of the tool tip.
        self._ladle_bowl_vec = np.copy(self._init_ladle_bowl_vec)

        # Debug positions of the spheres, in case some are sunk at start? For this
        # it might be easier to diagnose this if we print and save to files. Only save
        # if there's actually sunk stuff at the start.
        count = len([x for x in os.listdir('.') if 'count_sunk_episodes' in x])
        res_file = f'count_sunk_episodes_{count}.txt'
        res_str = ''
        item_height_0 = self._get_rigid_pos(item_index=0)[1]
        if item_height_0 < self.sunk_height_cutoff:
            res_str = f'reset(): target height: {item_height_0:0.3f}\n'
        for k in range(1, self.n_distractors+1):
            item_height_k = self._get_rigid_pos(item_index=k)[1]
            if item_height_k < self.sunk_height_cutoff:
                res_str = f'{res_str}reset(): dist {k} height: {item_height_k:0.3f}\n'
        if len(res_str) > 0:
            with open(res_file, 'w') as tf:
                print(res_str, file=tf)
        # -------------- end of new stuff -------------- #

        return self._get_obs()

    def sample_fluid_params(self, fluid_param_dic):
        """Sample params for the fluid.

        Takes our `dim_x,dim_y,dim_z` from our config and produces `x,y,z`.
        The former determines the liquid particle count, the latter determines the
        location where these get created in space. See `CreateParticleGrid()`.
        """
        params = fluid_param_dic
        self.fluid_params = fluid_param_dic

        # center of the glass floor. lower corner of the water fluid grid along x,y,z-axis.
        fluid_radis = params['radius'] * params['rest_dis_coef']
        self.x_center = 0
        self.fluid_params['x'] = self.x_center - (self.fluid_params['dim_x'] - 3) / 1. * fluid_radis + 0.1
        self.fluid_params['y'] = fluid_radis / 2 + 0.05
        self.fluid_params['z'] = 0. - (self.fluid_params['dim_z'] - 2) * fluid_radis / 1.5

        # Must have consistent ordering in the C++ file!
        _params = np.array([
            params['radius'], params['rest_dis_coef'], params['cohesion'],
            params['viscosity'], params['surfaceTension'], params['adhesion'],
            params['vorticityConfinement'], params['solidpressure'],
            self.fluid_params['x'], self.fluid_params['y'], self.fluid_params['z'],
            self.fluid_params['dim_x'], self.fluid_params['dim_y'], self.fluid_params['dim_z']
        ])
        return _params

    def initialize_camera(self):
        """Set the camera width, height, position and angle.

        Note from Xingyu/Yufei: width and height is actually the screen width and screen
        height of FleX. I suggest to keep them the same as the ones used in pyflex.cpp.

        Notes from Daniel:
        1.0 unit in position is conveniently equal to an edge of the FleX square in sim.
        For a top-down camera, use `top_down`, and I can use this for cached configs.
        The top down camera uses -0.4999*np.pi and not -0.50*np.pi because if we do the
        latter, it causes the keyboard commands to reverse when trying to navigate in the
        viewer (might be hitting some threshold?).
        """
        config = self.get_default_config()
        _r = config['fluid']['radius']

        self.camera_params = {
            'default_camera': {
                    'pos': np.array([0.0, 0.80, 0.20]),
                    'angle': np.array([0 * np.pi, -65 / 180. * np.pi, 0]),
                    'width': self.camera_width,
                    'height': self.camera_height},
            'top_down': {
                    'pos': np.array([0.0, 0.85, 0.0]),
                    'angle': np.array([0, -0.499999 * np.pi, 0.]),
                    'width': self.camera_width,
                    'height': self.camera_height},
            'cam_2d': {
                    'pos': np.array([0.5, .7, 4.]),
                    'angle': np.array([0, 0, 0.]),
                    'width': self.camera_width,
                    'height': self.camera_height},
            'left_side': {
                    'pos': np.array([-1, .2, 0]),
                    'angle': np.array([-0.5 * np.pi, 0, 0]),
                    'width': self.camera_width,
                    'height': self.camera_height},
            'right_side': {
                    'pos': np.array([2, .2, 0]),
                    'angle': np.array([0.5 * np.pi, 0, 0]),
                    'width': self.camera_width,
                    'height': self.camera_height}
        }

        # Adjust based on the particle radius for better view.
        if _r in [0.03, 0.033]:
            pass
        elif _r == 0.1:
            self.camera_params['default_camera']['pos'] = np.array([0.0, 2.25, 0.60])
        else:
            raise ValueError(_r)

    def set_scene(self, config, states=None):
        """Child envs can pass in specific fluid params through fluid param dic.

        Daniel: we only seem to sample fluid params here in this class.
        Requires careful integration with the C++ file to make sure everything
        is in order in the parameters! Also, we sample the camera parameters.

        Due to some earlier mistakes in fabrics projects (though not mixed media),
        as of Jan 19, we're enforcing an env version test to check that the cached
        configs have the intended meaning.
        """
        assert 'mm_env_version' in config.keys(), \
            f'Error, did we use the right cache? See keys: {config.keys()}'
        assert self.mm_env_version == config['mm_env_version'], \
            f'Check: {self.mm_env_version} vs config: {config}'

        # Back to normal, start with fluid parameters.
        fluid_params = self.sample_fluid_params(config['fluid'])

        # set camera parameters.
        self.initialize_camera()
        camera_name = config.get('camera_name', self.camera_name)
        camera_params = np.array([
            *self.camera_params[camera_name]['pos'],
            *self.camera_params[camera_name]['angle'],
            self.camera_width,
            self.camera_height,
            self.render_mode,
        ])

        # set tool parameters.
        tool_params = np.array([
            config['tool']['scale'],
            config['tool']['use_sdf'],
            # config['tool']['data'],
            self.tool_data, # HORRIBLE HACK!!! REPLACE LATER
            config['tool']['off_x'],
            config['tool']['off_z'],
        ])

        # set item parameters.
        item_params = np.array([
            config['item']['n_items'],
            config['item']['scale'],
            config['item']['inv_mass'],
            config['item']['spacing'],
        ])

        # set simulator parameters.
        sim_params = np.array([
            config['sim']['n_substeps'],
            config['sim']['n_iters'],
            config['sim']['inv_dt'],
            config['sim']['collision_distance'],
            config['sim']['particle_collision_margin'],
            config['sim']['shape_collision_margin'],
        ])

        # create scene parameters by concatenting stuff in order.
        scene_params = np.concatenate((fluid_params,
                                       camera_params,
                                       tool_params,
                                       item_params,
                                       sim_params))
        assert len(scene_params) == 38, len(scene_params)  # for C++ code

        # Daniel: using index 07. TODO(daniel) save the config somewhere
        pyflex_root = os.environ['PYFLEXROOT']
        if 'PyFlexRobotics' in pyflex_root:
            env_idx = 8
        else:
            assert 'PyFlex' in pyflex_root, pyflex_root
            env_idx = 7

        if self.version == 2:
            robot_params = []
            self.params = (scene_params, robot_params)
            pyflex.set_scene(env_idx, scene_params, 0, robot_params)
        elif self.version == 1:
            pyflex.set_scene(env_idx, scene_params, 0)

        self.particle_num = pyflex.get_n_particles()

    def get_state(self):
        """Get postion, velocity of flex particles, and postions of flex shapes.

        Also need to add extra attributes which track the tool and other items, such
        as the glass and the tool state. Regarding the tool tip, see `set_state()`.
        """
        particle_pos = pyflex.get_positions()
        particle_vel = pyflex.get_velocities()
        shape_position = pyflex.get_shape_states()
        return {'particle_pos': particle_pos,
                'particle_vel': particle_vel,
                'shape_pos': shape_position,
                'glass_x': self.glass_x,
                'glass_states': self.glass_states,
                'glass_params': self.glass_params,
                'config_id': self.current_config_id,
                'tool_state': self.tool_state,}
                #'tool_state_tip': self.tool_state_tip,}

    def set_state(self, state_dic):
        """Set postion, velocity of flex particles, and postions of flex shapes.

        We need to set glass parameters and tool states here, in order for the env to
        reset properly. (See Eddie's 12/03/2021 commit.) Before, we were also taking
        pyflex steps (see `PassWater`) but are removing that for CEM / RL.

        NOTE(daniel): We don't set the `self.tool_state_tip` here because it's going
        to be None when called from `set_scene()` -> `set_state()`, because the tip is
        formed after in the `_reset()` call because we need to get the camera matrix.
        IDK how best to do this but it should hopefully not matter for RL. Maybe it's
        also OK for CEM, please check. If we track tool rotations, we might have to
        also add that here.
        """
        # Hack to make sure tool inits properly in CEM (and probably other RL algos)
        if 'tool_state' in state_dic:
            self.tool_state = state_dic['tool_state']
            #self.tool_state_tip = state_dic['tool_state_tip']
            # Step pyflex to change rigid body state before setting positions
            #pyflex.step(self.tool_state[:3])  # NOTE(daniel) only if rigid body tool
        self.glass_params = state_dic['glass_params']
        self.glass_x = state_dic['glass_x']
        self.glass_states = state_dic['glass_states']
        pyflex.set_positions(state_dic["particle_pos"])
        pyflex.set_velocities(state_dic["particle_vel"])
        pyflex.set_shape_states(state_dic["shape_pos"])

    def _get_rigid_pos(self, item_index=0):
        """Get only positions of rigid particles, specified with `item_index`.
        WARNING: this returns a 4D vector (last item is inverse mass, I think)."""
        all_pos = self.get_state()['particle_pos'].reshape((-1, self.dim_position))
        item_idxs = np.where(self.particle_to_item == item_index)[0].astype(np.int32)
        assert len(item_idxs) > 0, f'Did you pick the index {item_index} right?'
        rigid_avg_pos = np.mean(all_pos[item_idxs], axis=0)
        return rigid_avg_pos

    def _get_rigid_radius(self, item_index=0):
        """Get only positions of the target item, specified with `item_index`.

        Assume we can take max over all x and min over all x. Seems reasonable?
        This is actually an approximation, and a slight underestimate of the radius,
        if we judge by pixels. Maybe also take the average of y and z? I wonder if
        this gives a more robust estimate.
        """
        all_pos = self.get_state()['particle_pos'].reshape((-1, self.dim_position))
        item_idxs = np.where(self.particle_to_item == item_index)[0].astype(np.int32)
        assert len(item_idxs) > 0, f'Did you pick the index {item_index} right?'
        max_x = np.max(all_pos[item_idxs, 0])
        min_x = np.min(all_pos[item_idxs, 0])
        max_y = np.max(all_pos[item_idxs, 1])
        min_y = np.min(all_pos[item_idxs, 1])
        max_z = np.max(all_pos[item_idxs, 2])
        min_z = np.min(all_pos[item_idxs, 2])
        rad_x = (max_x - min_x) / 2.
        rad_y = (max_y - min_y) / 2.
        rad_z = (max_z - min_z) / 2.
        return (rad_x + rad_y + rad_z) / 3.0

    def _set_particle_to_shape(self, n_items):
        """Sets np.array which maps from particle index to item index (ignore fluids).

        ASSUMPTIONS: we know the number of items, and each has the same number of
        particles, that they are arranged sequentially, and that we assigned to
        `self.rigid_idxs` and `self.fluid_idxs`. Here we'll assume item indices from
        0 are the desired items. We might use this for segmentation, for changing item
        positions, etc.

        IMPORTANT IMPORTANT: assume rigid idxs come BEFORE the water idxs.

        Example, with 3 items, we set:
            array([0,0,...,0,1,1,...,1,2,2,...,2])
        where the number of 0s, 1s, and 2s are equal.

        ALSO this doubles as a way to add offsets to the item positions when we sample
        them, since we put the items at (2,2), (3,3), etc.

        If we assume we can just index into this array, then we want the rigid indices
        to start from 0 (as stated earlier).
        """
        assert len(self.rigid_idxs) > 0, self.rigid_idxs
        particles_per_item = int(len(self.rigid_idxs) / n_items)
        assert particles_per_item * n_items == len(self.rigid_idxs)
        self.particle_to_item = np.concatenate(
            np.array([[i]*particles_per_item for i in range(n_items)]))

    def _get_init_tool_state(self, new_config=None):
        """Daniel: determine the tool state and return it (including fake tool).

        In C++, we can spawn the tool wherever we want, but let's set it here. Might
        be good to do that before we spawn water, so the tool 'moves out of the way'
        while the water gets spawned.

        Note: follows the same coordinate system, so the plane is (x,z) so to adjust
        the bowl we need to adjust it that way by adjusting np.array([x,y,z]), also
        seems like it has to be done for both of the positions (first 3 and next 3
        dims), otherwise weird physics. Note: if _moving_ the tool during real actions,
        current and previous have to be correctly set, but for initialization we can
        assign them the same values.

        NOTE(daniel): I don't think the `quat` matters here? To be safe I am setting
        it to the positive-y axis...

        08/16/2022: called from set_scene in subclass to move tool back above the
        center of the water. BUT if we want to randomize it, we should not be using
        the default `self.tool_init_x`, etc. Pass in an optional `new_config`.
        """
        tx, ty, tz = self.tool_init_x, self.tool_init_y, self.tool_init_z
        if (new_config is not None) and ('init_x' in new_config['tool']):
            tx = new_config['tool']['init_x']
            ty = new_config['tool']['init_y']
            tz = new_config['tool']['init_z']
        else:
            # This is really only needed if using MMOneSphere v02. Ignore if multi-sphere.
            if self.mm_env_version == 2:
                print('WARNING: new_config=None in `_get_init_tool_state()`.')
        T = self.tool_idx
        Tf = self.tool_idx_fake
        quat = quatFromAxisAngle([0, 1., 0.], 0.)
        offx = self.get_default_config()['tool']['off_x']
        offz = self.get_default_config()['tool']['off_z']

        states = np.zeros((2, self.dim_shape_state))
        states[T, :3] = np.array([tx, ty, tz])
        states[T, 3:6] = np.array([tx, ty, tz])
        states[T, 6:10] = quat
        states[T, 10:] = quat
        states[Tf, :3] = np.array([tx + offx, ty, tz + offz])
        states[Tf, 3:6] = np.array([tx + offx, ty, tz + offz])
        states[Tf, 6:10] = quat
        states[Tf, 10:] = quat
        return states

    def _step(self, action):
        """Take an action for ONE inner step (might have action repetition).

        By default, we use action repeat of 8 in mixed media envs, so `env.step()`
        in external code would call _this_ method 8 times.

        CAREFUL, `self.tool_state` needs to be updated each move so we can
        accurately input the "previous" tool state for shapes.

        For now we always clip the action. For translation the clipping is clear, for
        rotations we still clip the xyz portion, and for yrot we clip the single angle.
        For axis-angle, we clip items to (-1,1) and then later clip dtheta. TODO(daniel)
        need to carefully check if this is what Robosuite does. Also, for no rotations,
        we would need to set `dtheta=0` but have `axis` be nonzero right?

        To entirely disable collisions: comment out both calls of `_clip_bounds`. To
        debug collisions, put these lines at the end of this method:
            self._debug_ladle_bowl()
            pyflex.render()

        Parameters
        ----------
        Action: if translation only, np.array of dim (3,), with (dx,dy,dz), etc.
        """
        act_clip = np.clip(action, a_min=self.action_space.low, a_max=self.action_space.high)
        act_tran = act_clip[:3]

        # Careful: curr_roty should only change _after_ this method, but we change
        # it early. Also this might only work if we ignore collisions?
        if self.action_mode in ['translation', 'translation_onescale']:
            if self.action_mode == 'translation_onescale':
                act_tran *= self.max_transl_axis_ang
            assert len(act_clip) == 3, act_clip
            dtheta = 0.
            axis = [0., -1., 0.]
        elif self.action_mode == 'translation_yrot':
            # Use y axis (pointing down) to be compatible with (xz) plane in SoftGym.
            assert len(act_clip) == 4, act_clip
            dtheta = act_clip[3]
            axis = [0., -1., 0.]
            self.curr_roty += dtheta
        elif self.action_mode in ['translation_axis_angle','translation_axis_angle_onescale']:
            if self.action_mode == 'translation_axis_angle_onescale':
                act_tran *= self.max_transl_axis_ang

            # All components in `move[3:]` shoud be within (-1,1).
            assert len(act_clip) == 6, act_clip
            axis = act_clip[3:]

            # Confusion clarification 08/16/2022. For rotations, we take magnitude of
            # axis-angle and cap it according to self.max_rot_axis_ang.
            dtheta = np.linalg.norm(act_clip[3:])
            dtheta = min(dtheta, self.max_rot_axis_ang) # HACK

            # BUT here is what was in the code for the CoRL Submission:
            #
            # dtheta = np.linalg.norm(act_clip[3:])
            # if dtheta > self.max_rot_axis_ang:
            #     dtheta = dtheta * self.max_rot_axis_ang / np.sqrt(3)
            #
            # This will only scale down rotation if it excceds some value. BUT this
            # value is such that if dtheta is epsilon above the threshold, it gets a
            # lot smaller compared to if it were epsilon below the threshold. This
            # is discontinuous behavior but I notice that demonstration data ALWAYS
            # was under the self.max_rot_axis_ang threshold by 2X (whew) and that re-
            # running TFN and naive methods showed the same results (whew, again).

            # Here's what I _originally_ intended to do, to always scale:
            #
            # dtheta = np.linalg.norm(act_clip[3:]) * self.max_rot_axis_ang / np.sqrt(3)
            #
            # We can switch to that later but for now we have this cap above to make
            # sure the rotations won't be too wild.

            # Special case of no rotation to avoid quaternion errors?
            if dtheta == 0:
                axis = np.array([0., -1., 0.])

            # Hacky ... only because we track curr_roty for a demonstrator! If
            # positive rotation, axis is (0,-1,0) but scaled by some factor.
            if dtheta > 0:
                if axis[1] < 0:
                    self.curr_roty += dtheta
                else:
                    self.curr_roty -= dtheta
        elif self.action_mode == 'translation_quaternion':
            assert len(act_clip) == 7, act_clip
            raise NotImplementedError()

        # Using current tool state, adjust it based on action using axis angle.
        new_tool_state, new_tool_state_tip = self._move_tool(
                prev_tool_state=self.tool_state,
                prev_tool_tip=self.tool_state_tip,
                move=act_tran,
                axis=axis,
                dtheta=dtheta,
        )

        # Clip new tool state to collision and tool bounds
        collision_low, collision_high = self._get_collision_bounds()

        # Will only handle _positions_, not rotations.
        new_tool_state, new_tool_state_tip = self._clip_bounds(
            new_tool_state,
            new_tool_state_tip,
            collision_low,
            collision_high,
        )

        # Do this again to apply global constraints.
        new_tool_state, new_tool_state_tip = self._clip_bounds(
            new_tool_state,
            new_tool_state_tip,
            self.action_low_b,
            self.action_high_b,
        )

        # Update tool states
        self.tool_state = new_tool_state
        self.tool_state_tip = new_tool_state_tip

        # Get all current shapes and just override the tools assuming known indices.
        curr_shape_states = pyflex.get_shape_states()
        curr_shape_states = curr_shape_states.reshape((-1,14))
        new_shape_states = np.copy(curr_shape_states)
        new_shape_states[:2, :] = self.tool_state
        new_shape_states[self.tool_idx_tip, :] = self.tool_state_tip

        # Pyflex takes steps to update physics.
        pyflex.set_shape_states(new_shape_states)
        pyflex.step(render=self.render_img)
        self.inner_step += 1

    def _move_tool(self, prev_tool_state, prev_tool_tip, move, axis=[0., 1., 0.],
            dtheta=0.):
        """Move the tool, assuming no collisions (handle that elsewhere).

        IMPORTANT: we assume that the tool is at index 0, and fake tool is index 1.
        The fake tool must be offset in the positive x direction.

        Careful, with quaternions: q_rot2 * q_rot1 means composing rot1, then rot2,
        and is not commutative.

        Parameters
        ----------
        prev_tool_state: (2,14) np.array representing the pyflex tool shape state,
            and the fake tool. At index 0 is the 'real' tool:
                [0:3]:   x,y,z of tool (current)
                [3:6]:   x,y,z of tool (previous)
                [6:10]:  quaternion of tool (current)
                [10:14]: quaternion of tool (previous)
            See the other envs for the distinction between current and previous.
            After this, the 'current' state is in [0:3] and [6:10], so use those
            indices to then check if the tool is within bounds, etc.
        prev_tool_tip: (1,14) np.array representing the tool tip, which can be the
            center of rotation.
        move: (3,) np.array representing (dx,dy,dz) translations.
        axis: (3,) list representing the axis for rotation.
        dtheta: scalar representing the _change_ in rotation, in axis-angle form about
            the `axis`. E.g., if `dtheta` remains a constant positive value, the ladle
            should be rotating at a consistent rate.
        """
        assert len(move) == 3, len(move)

        # First, `states` needs to have previous state assigned to previous parts.
        states = np.zeros((2, self.dim_shape_state))
        states_tip = np.zeros((1, self.dim_shape_state))
        states[0, 3:6] = prev_tool_state[0, :3]
        states[0, 10:] = prev_tool_state[0, 6:10]
        states[1, 3:6] = prev_tool_state[1, :3]
        states[1, 10:] = prev_tool_state[1, 6:10]
        states_tip[0, 3:6] = prev_tool_tip[0, :3]   # tip
        states_tip[0, 10:] = prev_tool_tip[0, 6:10] # tip

        # Then, update the _current_ positions by adding to the _previous_.
        states[0, :3] = states[0, 3:6] + move
        states[1, :3] = states[1, 3:6] + move
        states_tip[0, :3] = states_tip[0, 3:6] + move

        # Excitingly, we get to play with rotations now
        curr_q = prev_tool_state[0, 6:10]
        qt_current = Quaternion(w=curr_q[3], x=curr_q[0], y=curr_q[1], z=curr_q[2])

        # Just to be safe, though I think pyquaternion can handle non-unit vectors.
        axis = axis / np.linalg.norm(axis)

        # Change axis from local frame to world
        axis_world = qt_current.rotate(axis)
        qt_rotate = Quaternion(axis=axis_world, angle=dtheta)
        qt_new = qt_rotate * qt_current

        # Assign new rotation to states
        new_items = qt_new.elements
        new_rot = np.array([new_items[1], new_items[2], new_items[3], new_items[0]])
        states[0, 6:10] = new_rot
        states[1, 6:10] = new_rot

        # Compute translation to make rotation work
        relative = states[0, :3] - states_tip[0, :3]
        relative_rot = qt_rotate.rotate(relative)
        delta = relative_rot - relative

        # Update positions
        states[0, :3] = states[0, :3] + delta
        states[1, :3] = states[1, :3] + delta

        # I think this is how to use rotation, but need to test collisions.
        self._ladle_bowl_vec = qt_rotate.rotate(self._ladle_bowl_vec)

        # Return updated `states` which later gets assigned to `self.tool_state`.
        return (states, states_tip)

    def _get_collision_bounds(self):
        """Get collision bounds.

        Careful, for the axis_angle, we assume here that we can use same bounds
        as with yrot and translation-only. While a demonstrator might restrict
        itself to yrot even with using axis_angle formulation, a policy could
        learn arbitrary rotations and thus these bounds might not apply
        """
        if (self.action_mode not in ['translation', 'translation_onescale',
                'translation_axis_angle', 'translation_axis_angle_onescale']):
            print('Warning: We have no collision checking.')
            return self.action_low_b, self.action_high_b

        # Using tuned values. Actually I have found that this radius might be
        # too conservative so downscaling this, a hack I know.
        sphere_rad = self.SPHERE_RAD * 0.75  # test with collision alg policy
        offset = self.collisions_offset

        abs_x_bound = self.glass_dis_x / 2. - self.border / 2. - sphere_rad
        abs_z_bound = self.glass_dis_z / 2. - self.border / 2. - sphere_rad

        low = np.array([-abs_x_bound, self.border / 2. + sphere_rad, -abs_z_bound])
        high = np.array([abs_x_bound, self.action_high_b[1] + offset, abs_z_bound])
        return low, high

    def _clip_bounds(self, tool_state, tool_state_tip, low, high):
        """Only handling positions for now."""
        assert len(low) == len(high)
        if len(low) > 3:
            low = low[:3]
            high = high[:3]

        # Position of the new tool tip, this we don't need to approximate.
        tip_x = tool_state_tip[0,0]
        tip_y = tool_state_tip[0,1]
        tip_z = tool_state_tip[0,2]

        # Find position of the center of the ladle's bowl's sphere.
        curr_pos = np.array([tip_x, tip_y, tip_z]) + self._ladle_bowl_vec

        # Clip tool center pos (new way):
        clipped_tool_pos = np.clip(curr_pos, low, high)
        clip_offset = clipped_tool_pos - curr_pos
        tool_state[0, :3] += clip_offset

        # Offset tool center followers
        tool_state[1, :3] += clip_offset
        tool_state_tip[0, :3] += clip_offset

        return tool_state, tool_state_tip

    def _set_glass_params(self, config=None):
        """Daniel: just setting params no pyflex calls here.

        This can help to understand dimensions and to tune distance thresholds. Given
        glass_dis_{x,z}, we can deduce roughly good action magnitudes for moving tools.
        """
        params = config
        self.border = params['border']
        self.height = params['height']
        fluid_radis = self.fluid_params['radius'] * self.fluid_params['rest_dis_coef']
        self.glass_dis_x = self.fluid_params['dim_x'] * fluid_radis + fluid_radis * 4  # glass floor length
        self.glass_dis_z = self.fluid_params['dim_z'] * fluid_radis + fluid_radis * 4  # glass width
        params['glass_dis_x'] = self.glass_dis_x
        params['glass_dis_z'] = self.glass_dis_z
        params['glass_x_center'] = self.x_center
        self.glass_params = params

    def _create_glass(self, glass_dis_x, glass_dis_z, height, border):
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

    def _init_glass_state(self, x, y, glass_dis_x, glass_dis_z, height, border):
        """Set init state of glass.

        NOTE(daniel) when creating a new shape in the C++ .h file, this will wreck the
        creation of the shapes and make the glass look weird. This creates a (5,14)
        numpy array, but I think an indexing issue causes this to affect the shape
        of only 4 of the 5 'walls' of the glass? Also why does each shape have dim 14?
        Looks like 3+3+4+4 where the 3s are for maybe start/end position, and the 4s
        are start/end quaternions?
        """
        dis_x, dis_z = glass_dis_x, glass_dis_z
        x_center, y_curr, y_last = x, y, 0.
        if self.action_mode in ['sawyer', 'franka']:
            y_curr = y_last = 0.56 # NOTE: robotics table
        quat = quatFromAxisAngle([0, 0, -1.], 0.)

        # states of 5 walls
        states = np.zeros((self.wall_num, self.dim_shape_state))

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

        states[:, 6:10] = quat
        states[:, 10:] = quat
        return states

    def _in_glass(self, water, glass_states, border, height, return_sum=True):
        """Judge whether a water particle is in the existing glass.
        water particle states are in [x, y, z, 1/m] format.

        NOTE(daniel): taken from the existing PassWater. We can keep it for now,
        and note that this will not count water that has been scooped out since
        the z-coordinates will be above the glass beight.

        Returns
        -------
        If `return_sum`, then returns the number of water particles in the glass.
        Otherwise, returns an array of booleans.
        """
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
        if return_sum:
            res = np.sum(res)
            return res
        return res

    def _add_box_boundary(self):
        """This is only because we have shooting water particles.

        CAREFUL! We want this to be the index _after_ the tool (and fake tool), but
        we also want this to be before the glass indices. Will require some tuning.
        We might want to increase spacing as well. Can also add more since it seems
        like particles really like to go through them!
        """
        center = np.array([0.8, -0.01, 0.8])
        quat = quatFromAxisAngle([0, 0, -1.], 0.)
        halfEdge = np.array([0.040, 0.500, 0.500])
        pyflex.add_box(halfEdge, center, quat)

        center = np.array([1.0, -0.01, 1.0])
        quat = quatFromAxisAngle([0, 0, -1.], 0.)
        halfEdge = np.array([0.040, 0.500, 0.500])
        pyflex.add_box(halfEdge, center, quat)

    def _add_box_tool_tip(self):
        """Add a tool tip shape which we use as the rotation center.

        This way if the tool moves, it moves with the box, so the box gives us a
        consistent position we can use to provide an offset from the tool 'position'.
        Create it here, then LATER reset it to the tool tip during initialization
        (during self._reset()). Assumes the tip is visible at the start.

        NOTE(daniel) if there's no rotation, then this doesn't really have too much
        effect as we just directly translate the existing tool, but it doesn't hurt
        to have, and we can make it small enough so it's imperceptible.

        TODO(daniel) does the quaternion here have to be aligned with the rotation
        angle that we use for rotations to move the item?
        """
        center = np.array([-1., -1., -1.])  # arbitrary
        quat = quatFromAxisAngle([0., 1., 0.], 0.)
        half_edge = np.array([0.0010, 0.0010, 0.0010])
        pyflex.add_box(half_edge, center, quat)
        new_shape_states = pyflex.get_shape_states().reshape((-1,14))
        return new_shape_states.shape[0] - 1

    def _get_water_height_init(self):
        """If calling after init, careful if water got scooped out."""
        pos = pyflex.get_positions().reshape(-1, 4)
        return np.max(pos[self.fluid_idxs, 1])

    def get_sunk_height_cutoff(self):
        """Returns height where if an item starts below, it is considered sunk."""
        return self.sunk_height_cutoff

    def set_video_recording_params(self):
        """
        Set the following parameters if video recording is needed:
            video_idx_st, video_idx_en, video_height, video_width
        NOTE(daniel) I don't think we normally use this.
        """
        self.video_height = 240
        self.video_width = 320

    def get_config(self):
        """I think we can call this for RL code?"""
        if self.deterministic:
            config_idx = 0
        else:
            config_idx = np.random.randint(len(self.config_variations))
        self.config = self.config_variations[config_idx]
        return self.config

    def _get_superclass_default_config(self):
        """Daniel: Get configuration and visually check if the physics looks good.

        Note: these are defaults but a bunch of these get overriden (e.g.,
        dim_x, dim_y, dim_z). Relevant references for parameters:

        https://github.com/Xingyu-Lin/softgym/blob/master/softgym/envs/pour_water.py
        https://github.com/YunzhuLi/PyFleX/tree/master/demo/scenes

        See also the FleX manual:
        https://docs.nvidia.com/gameworks/content/gameworkslibrary/physx/flex/manual.html
        Note: the 2:1 ratio of radius:restDistance is done via `rest_dis_coef`.

        TODO(daniel) prob. want a camera_param if we are to load in a cached
            state. I just copied whatever seems to be the default here.
        TODO(daniel) avoid overwriting them, and redefine some values. How
            to make it easy to tune?
        NOTE(daniel) the main parameter to NOT change too much is the radius. Almost
            everything is conditioned on that. Also, if changing tool vs sphere, it's
            easier to change the sphere size (for now).
        NOTE(daniel) subclasses can take this and add stuff to it.
        """
        config = {
            'fluid': {
                'radius': 0.033,  # We've really only tuned {0.030,0.033} and 0.100.
                'rest_dis_coef': 0.55,
                'cohesion': 0.1,
                'viscosity': 2.0,
                'surfaceTension': 0.,
                'adhesion': 0.0,
                'vorticityConfinement': 40,
                'solidpressure': 0.,
                'dim_x': 10,
                'dim_y': 40,  # gets overriden later as it's a fxn of x,z.
                'dim_z': 10,
            },
            'glass': {
                'border': 0.020,  # controls _thickness_ of the glass
                'height': 0.6,  # gets overwritten by generating variation
            },
            'camera_name': self.camera_name,
        }

        # Other configs for mixed media. Will have to tune.
        _r = config['fluid']['radius']

        # Tool usage. Type=0 means a bowl, 1, means 'denser' bowl. Others?
        # The off_x shows the offset for a fake tool that we use (for segmentation).
        config['tool'] = {
            'scale': self.tool_scale if _r in [0.030,0.033] else 0.36,
            'use_sdf': 1 if self.tool_type == 'sdf' else 0,
            'data': self.tool_data,
            'off_x': 2.0,
            'off_z': 2.0,
        }

        # Items. For now we assume it's just 0 or 1 items (and a sphere). Unfortunately
        # IDK how to derive a scale or inv_mass that causes 'density=1'-like behavior.
        # We do need one universal particle size. Also, for the offsets, check in with
        # the .h file. We're using (2.0, 0.5, 2.0) as the initial item spawn. Actually
        # this isn't used in C++, just in Python...
        # NOTE(daniel) spacing formula comes from Yunzhu Li.
        config['item'] = {
            #'n_items': 1,  # NOTE(daniel) actually the subclass should add to this.
            'scale': self.sphere_scale if _r in [0.030,0.033] else 0.20,
            'inv_mass': self.inv_mass,
            'spacing': (_r * 0.5) * 0.8,
            'off_x': -2.0 if _r in [0.030,0.033] else 0.0,
            'off_y':  0.4 if _r in [0.030,0.033] else 0.0,
            'off_z': -2.0 if _r in [0.030,0.033] else 0.0,
        }

        # Simulator params. Yunzhu made collision distance 0.01 with radius of 0.1?
        # https://github.com/YunzhuLi/PyFleX/blob/master/bindings/scenes/yz_boxbath.h
        # And similarly for the shape collision margin (with pasta scene).
        # https://github.com/YunzhuLi/PyFleX/blob/master/bindings/scenes/pasta.h
        # However shape collision margin tuning doesn't seem to improve the issue
        # with particles near the glass wall "flowing inwards".
        config['sim'] = {
            'n_substeps': self.n_substeps,
            'n_iters': self.n_iters,
            'inv_dt': self.inv_dt,
            'collision_distance': _r * 0.1,     # tune, fxn of radius?
            'particle_collision_margin': 0.0,   # tune, 0 for now?
            'shape_collision_margin': _r * 0.0, # tune, 0 for now?
        }

        # Adding as an extra layer of protection against mis-copied cached states.
        config['mm_env_version'] = self.mm_env_version
        return config

    def _item_in_bounds(self, item_pos):
        """Checks if item is in bounds and returns x and z conditions.
        Used in the info() methods for the envs, plus algorithmic policies.
        """
        in_bounds_x = np.abs(item_pos[0]) <= (self.glass_dis_x / 2.)
        in_bounds_z = np.abs(item_pos[2]) <= (self.glass_dis_z / 2.)
        return (in_bounds_x, in_bounds_z)

    def get_random_or_alg_action(self):
        """For external code, either an algorithmic action or a random action.

        Originally, we called `env.action_space.sample()` from  `demonstrator.py`
        and then enforced an algorithmic policy choice in `_step(). However for BC,
        we should return the actual action we used, including any possible noise
        and/or clipping, so we ultimately see the actual (obs, action) used. If we
        want to inject noise in DART-style (CoRL 2017) then do that in BC code.

        Confusing: we normalize the action using normalized envs, so we ancitipate
        this and de-normalize before returning, so the env later undoes this. But
        what action does BC use? We have (action, denorm_action). I think `action`
        is easier for BC as that is the actual value in meters to adjust the tool
        position, and it's how we debug / test such policies. Also due to action
        repeat, the `_step()` will run `action` multiple times, but BC should only
        'see' the first, so we have to scale by this factor in BC code.

        This also applies for the random action sampling, since calling it here
        will not call the normalized env, but in `demonstrator.py` we go to the
        normalized env.

        04/25/2022: actually maybe it's OK to use the `act` in `env.step(act)` as
        the BC target? That `act` is actually scaled in [-1,1] so the code did
        this for us.

        07/31/2022: easier to use translation_onescale or with axis angle, since
        if it's in (-1,1) then action == denorm_action.

        Returns
        -------
        (action, denorm_action): I originally used `action` to train BC, and then
            used scaling after that, but `denorm_action` does htis for us? NOTE!
            This action will be repeated according to the action repetition!
        """
        if self.alg_policy is not None:
            action = self._test_policy()
        else:
            action = self.action_space.sample()

        # Adding some noise to the actual action if desired.
        if self.act_noise is not None and self.act_noise > 0:
            action[0] += np.random.uniform(low=-self.act_noise, high=self.act_noise)
            action[1] += np.random.uniform(low=-self.act_noise, high=self.act_noise)
            action[2] += np.random.uniform(low=-self.act_noise, high=self.act_noise)

        # Clip action.
        action = np.clip(action, a_min=self.action_space.low, a_max=self.action_space.high)

        # As we observed from the initializer code in SoftAgent, we 'de-normalize'
        # the action because we later 'normalize' it.
        lb = self.action_space.low
        ub = self.action_space.high
        denorm_action = (action - lb) / ((ub - lb) * 0.5) - 1.0

        # Return both types of actions.
        return (action, denorm_action)

    # --------------------------------------------------------------------------- #
    # NOTE(daniel) temporary debugging / testing of various algorithmic policies. #
    # --------------------------------------------------------------------------- #

    def _visualize_action_boundary(self):
        """From action_space.py, `Picker` class, to visualize the range."""
        half_edge = np.array(self.action_high_b - self.action_low_b) / 2.
        center = np.array(self.action_high_b + self.action_low_b) / 2.
        quat = np.array([1., 0., 0., 0.])
        pyflex.add_box(half_edge, center, quat)

    def _debug_ladle_bowl(self):
        """Debugs the ladle's bowl?

        Ideally, we can continually track the ladle's bowl and use it to approximate
        collision checking for 6DoF manipulation?

        I think we can do this by initializing a ladle bowl vec which describes the
        offset from the tip center to the bowl center (w.r.t. a frame centered at the
        tip). Then if we just translate the tip, the center will move as well. If we
        rotate the ladle (tip), then we should apply a similar rotation on the vector
        storing this offset.

        Despite its name, `pyflex.pop_box()` works for pyflex shapes in general.
        """
        tip_center = np.array([
            self.tool_state_tip[0,0],
            self.tool_state_tip[0,1],
            self.tool_state_tip[0,2],
        ])
        bowl_center = tip_center + self._ladle_bowl_vec
        pyflex.add_sphere(self.SPHERE_RAD, bowl_center, [1, 0, 0, 0])
        pyflex.render()
        pyflex.pop_box(1)

    def _debug_diameter_bowl(self):
        """Debugs the diameter of the bowl in world space.

        Use case: can tune the center position and half edge length to measure
        the diameter of the bowl. Then we can use half of this to get the radius
        and thus if the ladle is rotated in some way about the y axis, we can
        roughly approximate the true center, which makes collision detection a
        lot more reliable.

        Update: seems like the tool we use has bowl diameter 0.076, assuming we
        start it from the outer edges instead of the inner edges, which I think
        is fine since we're going to be basing it off of where the ladle tip is
        located, and that's at the outer edge.
        """
        half_edge = np.array([0.02, 0.01, 0.076])
        center = np.array([
            self.tool_state_tip[0,0],
            self.tool_state_tip[0,1] - 0.160,  # hack
            self.tool_state_tip[0,2] + 0.076,  # hack
        ])
        print(center)
        print(self.tool_state_tip)
        quat = np.array([1., 0., 0., 0.])
        pyflex.add_box(half_edge, center, quat)
        while True:
            pyflex.render()

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
        if self.alg_policy == 'ladle_noop':
            # Nothing, useful if testing sliders from GUI.
            action = self._ladle_noop()
        elif self.alg_policy == 'ladle_collisions':
            # Test collisions.
            action = self._ladle_collisions(self.inner_step)
        elif self.alg_policy == 'ladle_6dof_rotations':
            action = self._ladle_6dof_rotations(self.inner_step)
        elif self.alg_policy == 'ladle_6dof_rotations_scoop':
            action = self._ladle_6dof_rotations_scoop(self.inner_step)
        elif self.alg_policy == 'ladle_algorithmic':
            # Algorithmic as of Feb 11.
            action = self._ladle_algorithmic(self.inner_step)
        elif self.alg_policy == 'ladle_algorithmic_v02':
            # Algorithmic v02 as of Feb 14.
            assert self.action_mode == 'translation', self.action_mode
            assert self.action_repeat == 8, self.action_repeat
            action = self._ladle_algorithmic_v02(self.inner_step)
        elif self.alg_policy == 'ladle_algorithmic_v03':
            # Algorithmic v03 as of April 24, with 4DoF (1 top-down rotation).
            # Edit: should probably ignore this.
            assert self.action_mode == 'translation_yrot', self.action_mode
            action = self._ladle_algorithmic_v03(self.inner_step)
        elif self.alg_policy == 'ladle_algorithmic_v04':
            # Algorithmic v04 as of April 26, w/axis_angle but really using 1DoF.
            assert self.action_mode == 'translation_axis_angle', self.action_mode
            assert self.action_repeat == 8, self.action_repeat
            action = self._ladle_algorithmic_v04(self.inner_step)
        elif self.alg_policy == 'ladle_algorithmic_v05':
            # As of May 03, 3DoF?
            assert self.action_mode == 'translation', self.action_mode
            assert self.action_repeat == 1, self.action_repeat
            action = self._ladle_algorithmic_v05(self.inner_step)
        elif self.alg_policy == 'ladle_algorithmic_v06':
            # As of May 03, 4DoF? VERY SIMPLE ROTATIONS.
            assert self.action_mode == 'translation_axis_angle', self.action_mode
            assert self.action_repeat == 1, self.action_repeat
            action = self._ladle_algorithmic_v06(self.inner_step)
        elif self.alg_policy == 'ladle_algorithmic_v07':
            # As of May 03, 4DoF? More complex rotations.
            assert self.action_mode == 'translation_axis_angle', self.action_mode
            assert self.action_repeat == 1, self.action_repeat
            action = self._ladle_algorithmic_v07(self.inner_step)
        elif self.alg_policy == 'ladle_algorithmic_v08':
            # As of May 13, 3DoF but back to act repeat 8, w/smarter demo.
            assert self.action_mode == 'translation', self.action_mode
            assert self.action_repeat == 8, self.action_repeat
            action = self._ladle_algorithmic_v08(self.inner_step)
        elif self.alg_policy == 'ladle_algorithmic_v09':
            # As of May 17, 4DoF but back to act repeat 8, w/smarter demo.
            assert self.action_mode == 'translation_axis_angle', self.action_mode
            assert self.action_repeat == 8, self.action_repeat
            action = self._ladle_algorithmic_v09(self.inner_step)
        elif self.alg_policy == 'ladle_high_level':
            # High-level policy to help SAC/CURL.
            action = self._ladle_high_level(self.inner_step)
        else:
            raise ValueError(self.alg_policy)
        return action

    def _ladle_noop(self):
        if self.action_mode == 'translation':
            action = np.array([0., 0., 0.])
        else:
            action = np.array([0., 0., 0., 0.])
        return action

    def _ladle_collisions(self, step):
        """Policy which deliberately moves the ladle to test collisions."""
        dx, dy, dz, wx, wy, wz = 0., 0., 0., 0., 0., 0.

        # Through these, sprinkle in a few values to the rotations.
        if 0 <= step < 100:
            dy = -0.0045
        elif 100 <= step < 150:
            dx = -0.0030
            wx = 0.005
        elif 150 <= step < 200:
            dz = -0.0030
            wy = -0.005
        elif 200 <= step < 300:
            dx = 0.0030
        elif 300 <= step < 400:
            #dz = 0.0030
            wz = 0.005
            wy = 0.005
        elif 400 <= step < 500:
            dx = -0.0030
        elif 500 <= step < 600:
            dz = 0.0030
            wz = 0.005
        else:
            wz = 0.005
            dy = 0.0045

        if self.action_mode == 'translation':
            action = np.array([dx, dy, dz])
        elif self.action_mode == 'translation_onescale':
            action = np.array([dx, dy, dz])
            action /= self.max_transl_axis_ang
        elif self.action_mode == 'translation_axis_angle':
            action = np.array([dx, dy, dz, wx, wy, wz])
        elif self.action_mode == 'translation_axis_angle_onescale':
            action = np.array([dx, dy, dz, wx, wy, wz])
            action[:3] /= self.max_transl_axis_ang
        else:
            raise ValueError(self.action_mode)
        return action

    def _ladle_6dof_rotations(self, step):
        """Policy to test full 6DoF motions of the ladle.

        This will NOT react to the ball, FYI. It's just hard-coded.
        """
        dx, dy, dz, wx, wy, wz = 0., 0., 0., 0., 0., 0.
        if 0 <= step < 50:
            pass
        elif 50 <= step < 100:
            dx = -0.0030
        elif 100 <= step < 200:
            dz = -0.0030
            wx = 0.5
        elif 200 <= step < 300:
            dx = 0.0010
            dy = -0.0020
            wx = 0
        elif 300 <= step < 400:
            dz = 0.0020
            wy = 0
        elif 400 <= step < 500:
            dx = -0.0030
            wy = 0
            wz = 0.2
        elif 500 <= step < 600:
            dx = -0.0030
            dy = 0.0010
            wx = -0.6
            wz = -0.3
        elif 600 <= step < 700:
            dy =  0.0
            wz = -0
        else:
            dy = 0.0020

        if self.action_mode == 'translation_axis_angle':
            action = np.array([dx, dy, dz, wx, wy, wz])
        elif self.action_mode == 'translation_axis_angle_onescale':
            action = np.array([dx, dy, dz, wx, wy, wz])
            action[:3] /= self.max_transl_axis_ang
        else:
            raise ValueError(self.action_mode)
        return action

    def _ladle_6dof_rotations_scoop(self, step):
        """Now this will try and scoop to the ball.

        Wow,this actually looks really good! Huh, wasn't expecting that. It was a
        quick hack that ended up working well. :) So for this let's just set it so
        that we ignore any actions where it's all 0. That's usually around step 456
        or so. Thus only about half of the time steps are useful generally.
        """
        action = self.AlgPolicyCls.get_6dof_action(step)
        if self.action_mode == 'translation_axis_angle_onescale':
            action[:3] /= self.max_transl_axis_ang
        return action

    def _ladle_algorithmic(self, step):
        """Move in xyz direction, with one 'vertical' rotation.

        General idea: go down, then move to match average (x,z) of the sphere, then move
        upwards. Unfortunately the tool position has to be tuned carefully, if we take
        the position it could be anywhere on the bowl. If the bowl is like this from a top
        down camera view:
               _
             /   \
            |     |
             \ _ /

        with +x pointing to the right and +z pointing down (yes), then the "position" seems
        to be somewhere in the upper left region.

        NOTE(daniel) need to look at the .obj files and deduce how the "position" is chosen.
        NOTE(daniel) we have to be careful when we describe the shape position, when we
            take the position I think it's a corner of the voxel used to construct it?
        NOTE(daniel) maybe an easy way to calibrate: we should know the glass distance, right?
            So we can kind of determine reasonable offsets to (tx,tz). In our case the glass
            distance is ~0.25, and I made the particle spawn to the upper left and then just
            changed tx,tz offsets until things got centered correctly, but we need a better way.

        (With rotations)
        General idea: same as `policy_move_xyz` except here we have a ladle with a
        stick, so we have to rotate the bowl about the y-axis so that the stick won't
        collide with the sphere.
        """
        T = self.tool_idx
        dx, dy, dz = 0., 0., 0.
        rotation = 0.

        # Position of the tool. Unfortunately adding to tx and tz is a real hack.
        tx = self.tool_state[T,0]
        ty = self.tool_state[T,1]
        tz = self.tool_state[T,2]
        tx += 0.090
        tz += 0.090
        #print(f'{tx:0.3f} {ty:0.3f} {tz:0.3f}')

        # Position of the item.
        rigid_avg_pos = self._get_rigid_pos()
        ix = rigid_avg_pos[0]
        iy = rigid_avg_pos[1]
        iz = rigid_avg_pos[2]
        #print(f'{ix:0.3f} {iy:0.3f} {iz:0.3f}')

        # Unfortunately distance thresholds have to be tuned carefully.
        dist_xz = np.sqrt((tx-ix)**2 + (tz-iz)**2)

        thresh1 = 220
        thresh2 = 330
        if 0 <= step < 15:
            # Only move if the item is below the sphere.
            if dist_xz < 0.050:
                # Go the _negative_ direction
                dx = -(ix - tx)
                dz = -(iz - tz)
            #print(f'step: {step}, dist: {dist_xz:0.3f}, dx: {dx:0.3f}, dz: {dz:0.3f}')
        elif 20 <= step < 210:
            # try to make it low to avoid too much water speed
            dy = -0.0019
        elif thresh1 <= step < thresh2:
            # Try to correct for the discrepancy
            if dist_xz > 0.005:
                dx = ix - tx
                dz = iz - tz
            #print(f'step: {step}, dist: {dist_xz:0.3f}, dx: {dx:0.3f}, dz: {dz:0.3f}')
        elif thresh2 <= step < 600:
            # Ah, it would actually be hard for us to do another xz correction here, since
            # if it causes collision, we'd just stop the motion. :(
            dy = 0.0020
        else:
            pass

        # Try to normalize (to magnitude 1) then downscale by a tuned amount.
        action = np.array([dx, dy, dz, 0.])
        if (5 <= step < 15) or (thresh1 <= step < thresh2):
            if np.linalg.norm(action) > 0:
                action = action / np.linalg.norm(action) * 0.0020

        if self.action_mode == 'translation':
            action = action[:3]
        else:
            # TODO(daniel) -- handle rotation
            action[3] = rotation
        return action

    def _ladle_algorithmic_v02(self, step, thresh1=75, thresh2=75+175):
        """This should be a much faster algorithmic policy.

        The sparse success rate will likely be similar to the first case, except the
        median will be MUCH higher due to fast movements.

        By default there are 800 `step`s in an episode due to action repeat, so for
        BC we should probably get rid of the last 1/4-ths because that's just the agent
        doing nothing... so just keep the first 600. But it WILL be good to have some
        data showing it stably holding the ball.

        Also used for v03 and v04 algo. policies, which use yrot and axis angles.
        However the axis angle demonstrator deliberately 'restricts' itself to use
        just a single rotation axis, but it outputs action vectors in the axis-angle
        formulation so that we can directly use it for full 6DoF supervision. Also,
        for these, after ladle goes up, we can rotate it back to the starting angle
        of pi/2 which might also help BC do recovery behavior. TODO(daniel): could
        instead use the 'smart initializer' that Eddie implemented.

        The v03, v04 algorithms limited to {-5, 0, 5} degree changes each action.
        """
        T = self.tool_idx
        dx, dy, dz, dyrot = 0., 0., 0., 0.

        # Position of the tip, this we don't need to approximate.
        tip_x = self.tool_state_tip[0,0]
        tip_y = self.tool_state_tip[0,1]
        tip_z = self.tool_state_tip[0,2]

        # Using our tuned values (to tune, make pyflex shapes at this spot).
        cx = tip_x + (self.SPHERE_RAD * np.cos(self.curr_roty))
        cy = tip_y - self.TIP_TO_CENTER_Y  # not needed, leaving just in case
        cz = tip_z + (self.SPHERE_RAD * np.sin(self.curr_roty))

        # Position of the item, this is kind of an approximation but a good one.
        rigid_avg_pos = self._get_rigid_pos()
        ix = rigid_avg_pos[0]
        iz = rigid_avg_pos[2]

        # Unfortunately distance thresholds have to be tuned carefully.
        dist_xz = np.sqrt((cx-ix)**2 + (cz-iz)**2)

        if 0 <= step < 15:
            # Only move if the item is below the sphere.
            if dist_xz < 0.050:
                # Go the _negative_ direction
                dx = -(ix - cx)
                dz = -(iz - cz)
        elif 15 <= step < 66:
            # Actually it seems like faster strangely means less water movement.
            dy = -0.0040  # stronger -dy won't have effect due to action bounds
        elif thresh1 <= step < thresh2:
            # Try to correct for the discrepancy. If using yrot, this is where
            # we should consider rotating the ladle. See method docs for how this
            # calculation works.
            if dist_xz > 0.004:
                dx = ix - cx
                dz = iz - cz
                if self.action_mode in ['translation_yrot', 'translation_axis_angle']:
                    # Find angle between item and tool tip center. (y,x) coords, or in
                    # our case, (z,x) coords. Requires extremely careful tuning. Also,
                    # positive rotations are clockwise from the top-down view, instead
                    # of counter-clockwise as would be the norm, due to the z direction.
                    ic_x = ix - tip_x
                    ic_z = iz - tip_z
                    curr_angle_r = np.arctan2(ic_z, ic_x)
                    curr_angle_d = curr_angle_r * RAD_TO_DEG
                    diff_angle_r = curr_angle_r - self.curr_roty

                    # Good catch. If |diff| >= 180 deg, then add or subtract 360.
                    if diff_angle_r <= - np.pi:
                        diff_angle_r += 2 * np.pi
                    elif diff_angle_r >= np.pi:
                        diff_angle_r -= 2 * np.pi

                    diff_angle_d = diff_angle_r * RAD_TO_DEG

                    # If it's within a few deg, we might just avoid any rotation?
                    # Remember that there is action repeat. Also maybe if this is
                    # nonzero, we actually override the translation?
                    thresh_deg = 20.0 / self.action_repeat  # rotate if >=20 deg diff
                    thresh_act =  5.0 / self.action_repeat  # move by 5 deg each action
                    if diff_angle_d <= -thresh_deg:
                        delta_deg = -thresh_act
                    elif diff_angle_d >= thresh_deg:
                        delta_deg = thresh_act
                    else:
                        delta_deg = 0
                    dyrot = delta_deg * DEG_TO_RAD
        elif thresh2 <= step < 600:
            # Just go upwards.
            dy = 0.0040

            # It would be better to measure if it hits the height limit instead of using
            # 380 steps here which is a hack and must change if we change the thresh{1,2}.
            if (self.action_mode in ['translation_yrot', 'translation_axis_angle']
                    and 380 <= step):
                if self.curr_roty < np.pi/2. - np.pi:
                    # Less than 90 deg?
                    self.curr_roty += 2 * np.pi
                elif self.curr_roty > np.pi/2. + np.pi:
                    # Greater than 270 deg?
                    self.curr_roty -= 2 * np.pi

                # Move by +/-5 deg each action
                thresh_act =  5.0 / self.action_repeat
                if self.curr_roty < (np.pi / 2):
                    delta_deg = thresh_act
                elif self.curr_roty > (np.pi / 2):
                    delta_deg = -thresh_act
                else:
                    delta_deg = 0
                dyrot = delta_deg * DEG_TO_RAD
        else:
            # This data can be ignored for BC (remove last 1/4 of data) but it's nice
            # to leave the episode length longer due to some recovery behavior. For
            # training, the extra actions of 'do nothing' here might be confusing.
            pass

        # Try to normalize (to magnitude 1) then downscale by a tuned amount.
        # Unfortunately these numbers just come from tuning / visualizing.
        # As is the heuristic that we'll ignore rotations for this process.
        action = np.array([dx, dy, dz])  # only position magnitudes
        if (5 <= step < 15) or (thresh1 <= step < thresh2):
            if np.linalg.norm(action) > 0:
                action = action / np.linalg.norm(action) * 0.0020

        # Bells and whistles.
        if self.action_mode == 'translation':
            pass
        elif self.action_mode == 'translation_yrot':
            if dyrot != 0:
                action = np.array([0.,0.,0.,dyrot])
            else:
                action = np.array([action[0], action[1], action[2], 0.])
        elif self.action_mode == 'translation_axis_angle':
            # As with v03, rotation changes to be +/-5 degrees each action.
            if dyrot != 0:
                targ_delta_deg = 5.0 / self.action_repeat
                targ_delta_rad = targ_delta_deg * DEG_TO_RAD
                if dyrot > 0:
                    aa = np.array([0., -1., 0.])
                else:
                    aa = np.array([0., 1., 0.])
                aa = (aa / np.linalg.norm(aa)) * targ_delta_rad
                action = np.array([0., 0., 0., aa[0], aa[1], aa[2]])
            else:
                action = np.concatenate((action, np.array([0., 0., 0.])))
        else:
            raise NotImplementedError()
        #print(f'rot {self.curr_roty*RAD_TO_DEG:0.2f}, action: {action}')
        return action

    def _ladle_algorithmic_v03(self, step):
        """For now, call v02 directly because we'll use same stuff."""
        return self._ladle_algorithmic_v02(step, thresh1=75, thresh2=75+200)

    def _ladle_algorithmic_v04(self, step):
        """For now, call v02 directly because we'll use same stuff."""
        return self._ladle_algorithmic_v02(step, thresh1=75, thresh2=75+200)

    def _ladle_algorithmic_v05(self, step):
        """Starting new set of algorithmic demos with action_repeat=1,
        but this can still be used with our earlier setup.

        I also think we should NOT try the initial displacement strategy because
        that would be hard for a policy to learn from due to occlusions, hence
        not doing that here (so I expect raw performance to be slightly worse).
        """
        return self.AlgPolicyCls.get_action()

    def _ladle_algorithmic_v06(self, step):
        """Now with SIMPLEST POSSIBLE rotation."""
        return self.AlgPolicyCls.get_action(rotate_start=True)

    def _ladle_algorithmic_v07(self, step):
        """Now with rotation based on ball reaction."""
        return self.AlgPolicyCls.get_action_rotation()

    def _ladle_algorithmic_v08(self, step):
        """Translation-only action repeat=8 with smart demonstrator.

        Same as v05 except using action_repeat=8.
        Same as v02 except using the more intelligent demonstrator.
        """
        return self.AlgPolicyCls.get_action()

    def _ladle_algorithmic_v09(self, step):
        """Translation + 1D Rotation repeat=8 with smart demonstrator.
        Same as v06 except using action_repeat=8.
        Same as v04 except using the more intelligent demonstrator and also
        simplifying the rotation (v04 has a more complex but potentially
        more interesting set of rotations).
        """
        return self.AlgPolicyCls.get_action(rotate_start=True)

    def _ladle_high_level(self, step):
        """Testing high-level policy to facilitate SAC/CURL for later.

        action space per time step, per component bounds:
            [-0.004 -0.004 -0.004] [0.004 0.004 0.004]
        NOTE(daniel): this is only enforced for the first few time steps in an episode.
        It runs the `_ladle_algorithmic_v02` as a subroutine.
        If we do 200 steps for this, that means the first 1/4 of the episode is taken up
        by the algorithmic policy, the remaining 3/4 has to be done from RL.
        """
        thresh2 = 75 + 125  # check this with `_ladle_algorithmic_v02`
        if step < thresh2:
            action = self._ladle_algorithmic_v02(step)
        else:
            action = np.array([0.,0.,0.])  # should be passed to RL
        return action


class AlgorithmicPolicy():
    """Separate class might be easier since it may involve tracking state.

    The dx, dy, dz are what we actually insert into the action vector, for
    `env.step(action)`. We supply an action (e.g., with `get_action()`) and
    the env will step though it based on `action_repeat`.
    """

    def __init__(self, env):
        self.env = env
        self.state = 0
        self._rest = 0
        self._rest_period = int(16 / self.env.action_repeat)
        self._horiz_thresh = 0.005
        self._print = False

    def reset(self):
        self.state = 0
        self._rest = 0

    def get_6dof_action(self, step, info=None):
        """TBD"""
        dx, dy, dz, wx, wy, wz = 0., 0., 0., 0., 0., 0.

        # Position of the tip, this we don't need to approximate.
        tip_x = self.env.tool_state_tip[0,0]
        tip_y = self.env.tool_state_tip[0,1]
        tip_z = self.env.tool_state_tip[0,2]

        # Find center of the sphere ladle's bowl.
        curr_pos = np.array([tip_x, tip_y, tip_z]) + self.env._ladle_bowl_vec
        cx, cy, cz = curr_pos[0], curr_pos[1], curr_pos[2]

        # Position of the item, this is kind of an approximation but a good one.
        rigid_avg_pos = self.env._get_rigid_pos()
        ix = rigid_avg_pos[0]
        _  = rigid_avg_pos[1]
        iz = rigid_avg_pos[2]

        # Unfortunately distance thresholds have to be tuned carefully.
        dist_xz = np.sqrt((cx-ix)**2 + (cz-iz)**2)
        #print(f'item is: {rigid_avg_pos} and is on left side? {ix < 0}')

        if self.state == 0:
            # Just move ladle backwards a bit, lower a bit.
            wx = 0.002
            dy = -0.0010
            # Ball is to left or right? Edit: we're biasing the sampling to be left,
            # but sometimes it will be 'slightly right' in which case we don't need
            # to do a rotation in this axis.
            if ix < 0:
                wz = 0.0040
            if step > 50:
                self.state += 1
        elif self.state == 1:
            # Just lower? Or also rotate a tiny bit.
            dy = -0.0020
            wy = 0.0030
            if cy <= 0.21:
                self.state += 1
        elif self.state == 2:
            # Assume this state means horizontal movement to get item.
            if dist_xz >= self._horiz_thresh:
                dx = (ix - cx) * 0.008
                dz = (iz - cz) * 0.008
            else:
                self.state += 1
            # Note: sometimes this triggers before the dist_xz threshold?
            if step > 250:
                self.state += 1
        elif self.state == 3:
            # Revert earlier rotations.
            wx = -0.0020
            wy = -0.0030
            wz = -0.0040
            if np.abs(self.env._ladle_bowl_vec[0]) < 0.02:
                self.state += 1
        elif self.state == 4:
            dy = 0.0020
            if cy > 0.45:
                self.state += 1
        else:
            # Our code should detect and probably ignore in BC, so that it just
            # learns to go upwards (which will cause it to hit env bounds).
            pass

        action = np.array([dx, dy, dz, wx, wy, wz])
        if self._print:
            print(f'is {self.env.inner_step}, state {self.state}, '
                   f'dist_xz {dist_xz:0.3f}, {action}')
        return action

    def get_action(self, rotate_start=False):
        """Careful about the action bounds.

        With action repeat, this can be susceptible to 'back and forth' motions.
        Need to also be careful about 'resting periods' since that depends on how
        often this method is called.

        Parameters
        ----------
        rotate_start: True if we rotate at the start by 90 degrees to make the
        ladle stick turn a certain direction before doing 3DoF translations. At
        minimum, our method has to be able to do that! Also assume that if this
        is true, we want axis-angle actions, hence 6D actions.
        """
        dx, dy, dz = 0., 0., 0.
        dyrot = 0.

        # Position of the tip, this we don't need to approximate.
        tip_x = self.env.tool_state_tip[0,0]
        tip_y = self.env.tool_state_tip[0,1]
        tip_z = self.env.tool_state_tip[0,2]

        # Using our tuned values (to tune, make pyflex shapes at this spot).
        # This indicates the center of the ladle's 'sphere bowl' roughly.
        cx = tip_x + (self.env.SPHERE_RAD * np.cos(self.env.curr_roty))
        cy = tip_y - self.env.TIP_TO_CENTER_Y
        cz = tip_z + (self.env.SPHERE_RAD * np.sin(self.env.curr_roty))

        # Position of the item, this is kind of an approximation but a good one.
        rigid_avg_pos = self.env._get_rigid_pos()
        ix = rigid_avg_pos[0]
        iy = rigid_avg_pos[1]
        iz = rigid_avg_pos[2]

        # Unfortunately distance thresholds have to be tuned carefully.
        dist_xz = np.sqrt((cx-ix)**2 + (cz-iz)**2)

        # Check in-bounds conditions. If not, in 'death' state.
        in_bounds_x, in_bounds_z = self.env._item_in_bounds(rigid_avg_pos)
        if (not in_bounds_x) or (not in_bounds_z):
            self.state = -1

        # This avoids when the 1st action outputs 0 translation.
        if (not rotate_start) and (self.state == 0):
            self.state = 1

        if self.state == 0:
            assert rotate_start
            # Now attempt to rotate to 45 degrees.
            if self.env.curr_roty < np.pi/2. - np.pi:
                # Less than 90 deg?
                self.env.curr_roty += 2 * np.pi
            elif self.env.curr_roty > np.pi/2. + np.pi:
                # Greater than 270 deg?
                self.env.curr_roty -= 2 * np.pi

            # From `env.curr_roty`, move by +/-5 deg each action
            thresh_act =  1.0
            if self.env.curr_roty > 45.0 * DEG_TO_RAD:
                delta_deg = -thresh_act
            else:
                delta_deg = 0
            dyrot = delta_deg * DEG_TO_RAD
            if delta_deg == 0:
                self.state += 1
        elif self.state == 1:
            # Lowering ladle state, as long as ladle is above a threshold.
            if cy > 0.08:
                dy = -0.004
            else:
                self.state += 1
        elif self.state == 2:
            # Assume this state means horizontal movement to get item.
            if dist_xz >= self._horiz_thresh:
                dx = ix - cx
                dz = iz - cz
            else:
                self.state += 1
        elif self.state == 3:
            # Use as a 'resting' period for the ladle to improve stability.
            self._rest += 1
            if dist_xz >= self._horiz_thresh:
                self._rest = 0
                self.state -= 1
            elif self._rest >= self._rest_period:
                self._rest = 0
                self.state += 1
        elif self.state == 4:
            # Raise the ladle. BUT if the item falls out, revert to state 0.
            # Note: item might potentially stick into the ladle. Resolve this
            # with the ladle going up above a thresold.
            dy = 0.004
            if (iy + 0.05 < cy) and (cy > self.env.water_height_init + 0.20):
                self.state = 1  # Not 0 since that's for rotations.
            # Also check if we've hit the height limit, if so we should just
            # disable dy (but might sitll need to check if item falls out).
            if cy > 0.45:
                dy = 0.0
        else:
            pass

        # Try to normalize (to magnitude 1) then downscale by a tuned amount.
        # Unfortunately these numbers just come from tuning / visualizing.
        action = np.array([dx, dy, dz])
        if self.state == 2 and np.linalg.norm(action) > 0:
            # In this state, sideways movement. Might want to lower the
            # magnitude if we're already really close?
            if dist_xz > self._horiz_thresh * 2:
                action = action / np.linalg.norm(action) * 0.002
            else:
                action = action / np.linalg.norm(action) * 0.001

        if rotate_start:
            if self.state == 0:
                targ_delta_deg = 1.0
                targ_delta_rad = targ_delta_deg * DEG_TO_RAD
                if dyrot > 0:
                    aa = np.array([0., -1., 0.])
                else:
                    aa = np.array([0., 1., 0.])
                aa = (aa / np.linalg.norm(aa)) * targ_delta_rad
                action = np.array([0., 0., 0., aa[0], aa[1], aa[2]])
            else:
                action = np.array([action[0], action[1], action[2], 0., 0., 0.])

        if self._print:
            print((f'is {self.env.inner_step}, state {self.state}, '
                   f'rest {self._rest}, dist_xz {dist_xz:0.3f}, {action}, '
                   f'rot {self.env.curr_roty*RAD_TO_DEG:0.1f}'))
        return action

    def get_action_rotation(self, info=None):
        """Now with rotation? This is a bit more complex but should still be a
        good test bed for evaluating methods on demonstrators with rotations.

        Once again, actions are ONLY translations, or ONLY rotation, we don't
        currently mix these together. A policy should learn that, right?
        """
        dx, dy, dz = 0., 0., 0.
        dyrot = 0.

        # Position of the tip, this we don't need to approximate.
        tip_x = self.env.tool_state_tip[0,0]
        tip_y = self.env.tool_state_tip[0,1]
        tip_z = self.env.tool_state_tip[0,2]

        # Using our tuned values (to tune, make pyflex shapes at this spot).
        # This indicates the center of the ladle's 'sphere bowl' roughly.
        cx = tip_x + (self.env.SPHERE_RAD * np.cos(self.env.curr_roty))
        cy = tip_y - self.env.TIP_TO_CENTER_Y
        cz = tip_z + (self.env.SPHERE_RAD * np.sin(self.env.curr_roty))

        # Position of the item, this is kind of an approximation but a good one.
        rigid_avg_pos = self.env._get_rigid_pos()
        ix = rigid_avg_pos[0]
        iy = rigid_avg_pos[1]
        iz = rigid_avg_pos[2]

        # Unfortunately distance thresholds have to be tuned carefully.
        dist_xz = np.sqrt((cx-ix)**2 + (cz-iz)**2)

        # Check in-bounds conditions.
        in_bounds_x, in_bounds_z = self.env._item_in_bounds(rigid_avg_pos)
        if (not in_bounds_x) or (not in_bounds_z):
            self.state = -1

        # -------------------------------------------------------------- #
        # Find angle between item and tool tip center. (y,x) coords, or in
        # our case, (z,x) coords. Requires extremely careful tuning. Also,
        # positive rotations are clockwise from the top-down view, instead
        # of counter-clockwise as would be the norm, due to the z direction.
        ic_x = ix - tip_x
        ic_z = iz - tip_z
        curr_angle_r = np.arctan2(ic_z, ic_x)
        curr_angle_d = curr_angle_r * RAD_TO_DEG
        diff_angle_r = curr_angle_r - self.env.curr_roty

        # Good catch. If |diff| >= 180 deg, then add or subtract 360.
        if diff_angle_r <= - np.pi:
            diff_angle_r += 2 * np.pi
        elif diff_angle_r >= np.pi:
            diff_angle_r -= 2 * np.pi

        diff_angle_d = diff_angle_r * RAD_TO_DEG

        # If it's within a few deg, we might just avoid any rotation?
        # Remember that there is action repeat. Also maybe if this is
        # nonzero, we actually override the translation?
        thresh_deg = 10.0  # rotate if >=10 deg diff
        thresh_act =  1.0  # move by 1 deg each action
        if diff_angle_d <= -thresh_deg:
            delta_deg = -thresh_act
        elif diff_angle_d >= thresh_deg:
            delta_deg = thresh_act
        else:
            delta_deg = 0
        # -------------------------------------------------------------- #

        if self.state == 0:
            # Lowering ladle state, as long as ladle is above a threshold.
            if cy > 0.08:
                dy = -0.004
            else:
                self.state += 1
        elif self.state == 1:
            # Rotation? If delta_deg is 0, move to the resting period.
            dyrot = delta_deg * DEG_TO_RAD
            if delta_deg == 0:
                self.state += 1
        elif self.state == 2:
            # Use as a 'resting' period for the ladle to improve stability.
            self._rest += 1
            if delta_deg != 0:
                self._rest = 0
                self.state -= 1
            elif self._rest >= self._rest_period:
                self._rest = 0
                self.state += 1
        elif self.state == 3:
            # Assume this state means horizontal movement to get item.
            if dist_xz >= self._horiz_thresh:
                dx = ix - cx
                dz = iz - cz
            else:
                self.state += 1
        elif self.state == 4:
            # Use as a 'resting' period for the ladle to improve stability.
            self._rest += 1
            if dist_xz >= self._horiz_thresh:
                self._rest = 0
                self.state -= 1
            elif self._rest >= self._rest_period:
                self._rest = 0
                self.state += 1
        elif self.state == 5:
            # Raise the ladle. BUT if the item falls out, revert to state 0.
            # Note: item might potentially stick into the ladle. Resolve this
            # with the ladle going up above a thresold.
            dy = 0.004
            if (iy + 0.05 < cy) and (cy > self.env.water_height_init + 0.20):
                self.state += 1
            # Also check if we've hit the height limit, if so we should just
            # disable dy (but might sitll need to check if item falls out).
            if cy > 0.45:
                # Steady state.
                dy = 0.0
        elif self.state == 6:
            # Wait, if the ball falls out, let's actually revert back to default
            # first, then we can lower the ladle. So if we come here we assume
            # the ball has dropped and we now rotate back to default.

            # Now attempt to rotate back to default.
            if self.env.curr_roty < np.pi/2. - np.pi:
                # Less than 90 deg?
                self.env.curr_roty += 2 * np.pi
            elif self.env.curr_roty > np.pi/2. + np.pi:
                # Greater than 270 deg?
                self.env.curr_roty -= 2 * np.pi

            # Move by +/-5 deg each action
            thresh_act =  1.0
            if self.env.curr_roty < (np.pi / 2):
                delta_deg = thresh_act
            elif self.env.curr_roty > (np.pi / 2):
                delta_deg = -thresh_act
            else:
                delta_deg = 0
            dyrot = delta_deg * DEG_TO_RAD

            # Now revert back to lowering.
            if delta_deg == 0:
                self.state = 0
        else:
            pass

        # Try to normalize (to magnitude 1) then downscale by a tuned amount.
        # Unfortunately these numbers just come from tuning / visualizing.
        action = np.array([dx, dy, dz])
        if self.state == 3 and np.linalg.norm(action) > 0:
            # In this state, sideways movement. Might want to lower the
            # magnitude if we're already really close?
            if dist_xz > self._horiz_thresh * 2:
                action = action / np.linalg.norm(action) * 0.002
            else:
                action = action / np.linalg.norm(action) * 0.001

        # Handle rotation. Aim to move 1 degree each action.
        if dyrot != 0:
            targ_delta_deg = 1.0
            targ_delta_rad = targ_delta_deg * DEG_TO_RAD
            if dyrot > 0:
                aa = np.array([0., -1., 0.])
            else:
                aa = np.array([0., 1., 0.])
            aa = (aa / np.linalg.norm(aa)) * targ_delta_rad
            action = np.array([0., 0., 0., aa[0], aa[1], aa[2]])
        else:
            action = np.concatenate((action, np.array([0., 0., 0.])))

        if self._print:
            print((f'is {self.env.inner_step}, state {self.state}, '
                   f'rest {self._rest}, dist_xz {dist_xz:0.3f}, {action}, '
                   f'rot {self.env.curr_roty*RAD_TO_DEG:0.1f}'))
        return action
