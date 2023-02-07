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


class SpheresEnv(FlexEnv):
    """Adapted from `MixedMediaEnv`. This is mainly to simplify things.
    Env versions (see subclasses) mainly specify geometry and location of the tool.
    """

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

        # Stuff that we originally tuned but probably should keep fixed.
        self.n_substeps = 2
        self.n_iters = 4
        self.inv_dt = 100
        self.inv_mass = 0.50
        self.sphere_scale = 0.060
        self.act_noise = 0.0
        self.tool_type = 'sdf'
        self.tool_data = 2

        # Handle starting ladle. Call the subclass __init__ before this.
        # NOTE! Special to this env, override the kwargs camera_name.
        # For most of these, I think we can generate the cache and then adjust them
        # later, but it may be simpler to have fixed version numbers.
        # Debug tool properties with `self._debug_diameter_bowl()`.
        if self.spheres_env_version == 1:
            self.tool_scale = 0.28
            self.tool_init_x = -0.10
            self.tool_init_y =  0.28  # smaller = ladle is closer to water surface
            self.tool_init_z = -0.10
            self._sphere_radius_PCL = 0.030  # for sampling 'ground truth' sphere PCLs.
            self.SPHERE_RAD = 0.076  # 'outer' sphere (the 'bowl' is thick)
            self.TIP_TO_CENTER_Y = 0.180  # tip to center of the ladle's bowl
            self.DISTANCE_THRESH = (self.SPHERE_RAD + self._sphere_radius_PCL) * 0.85
            kwargs['camera_name'] = 'top_down'
        elif self.spheres_env_version in [2,3]:
            self.tool_scale = 0.20
            self.tool_init_x = -0.10
            self.tool_init_y =  0.35  # smaller = ladle is closer to water surface
            self.tool_init_z = -0.10
            self._sphere_radius_PCL = 0.030  # for sampling 'ground truth' sphere PCLs.
            self.SPHERE_RAD = 0.054  # 'outer' sphere (the 'bowl' is thick)
            self.TIP_TO_CENTER_Y = 0.128  # tip to center of the ladle's bowl
            self.DISTANCE_THRESH = (self.SPHERE_RAD + self._sphere_radius_PCL) * 0.90
            kwargs['camera_name'] = 'top_down_v02'
        else:
            raise ValueError(self.spheres_env_version)

        # Stuff relevant to the reward and env termination. Assumes one target ball.
        self.height_cutoff = 0.40
        self.prev_reward = 0
        self.reward_min = 0
        self.reward_max = 1
        self.performance_init = 0  # TODO(daniel)

        # FlexEnv init, then obs and action mode (I think this ordering is OK).
        self.action_mode = action_mode
        self.observation_mode = observation_mode
        super().__init__(**kwargs)

        # Special for these sphere envs, allow for early termination.
        self.terminate_early = True

        # Choose observation mode. See `self._get_obs()` for documentation. The
        # combo one should not normally be used, mainly for BC so we can collect
        # as many different obs types as possible given the data.
        self.max_pts = 2000  # 1000-2500 seems standard. Yufei uses <=4000.
        if observation_mode == 'key_point':
            self.observation_space = Box(
                    low=np.array([-np.inf] * self.obs_dim_keypt),
                    high=np.array([np.inf] * self.obs_dim_keypt),
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
        if action_mode == 'translation':
            # 3 DoFs, (deltax, deltay, deltaz). Remember, y points up.
            action_low  = np.array([-0.003, -0.003, -0.003])
            action_high = np.array([ 0.003,  0.003,  0.003])
            self.action_space = Box(action_low, action_high, dtype=np.float32)
            self.action_low_b  = np.array([-0.25, -0.10, -0.25])
            self.action_high_b = np.array([ 0.25,  0.50,  0.25])
        else:
            raise NotImplementedError(action_mode)

        # TODO(daniel) HACK! For now, assume y axis points _downwards_. If changing,
        # have to adjust algorithmic policies, etc. See the `_reset()`!
        self.curr_roty = None

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
        self.curr_roty = np.pi / 2.0  # set before `_get_info()`
        info = self._get_info()
        self.performance_init = info['performance']
        pyflex.step(render=self.render_img)

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

        assert self.sunk_height_cutoff < self.water_height_init
        #self._visualize_action_boundary()  # to debug action ranges
        #self._debug_tool_properties()  # debug physical properties of tool

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
            'top_down': {  # spheres_env_version == 1
                    'pos': np.array([0.0, 0.85, 0.0]),
                    'angle': np.array([0, -0.499999 * np.pi, 0.]),
                    'width': self.camera_width,
                    'height': self.camera_height},
            'top_down_v02': {  # spheres_env_version in [2,3]
                    'pos': np.array([0.0, 0.85, 0.25]),
                    'angle': np.array([0, -0.35 * np.pi, 0.]),
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
        assert 'spheres_env_version' in config.keys(), \
            f'Error, did we use the right cache? See keys: {config.keys()}'
        assert self.spheres_env_version == config['spheres_env_version'], \
            f'Check: {self.spheres_env_version} vs config: {config}'

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

    def _get_init_tool_state(self):
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
        """
        tx, ty, tz = self.tool_init_x, self.tool_init_y, self.tool_init_z
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

        Parameters
        ----------
        Action: if translation only, np.array of dim (3,), with (dx,dy,dz).
            If translation_yrot, (dx,dy,dz,dthetay). For more info, see the
            documentation in the constructor.
        """
        act_clip = np.clip(action, a_min=self.action_space.low, a_max=self.action_space.high)
        act_tran = act_clip[:3]

        # Careful: curr_roty should only change _after_ this method, but we change
        # it early. Also this might only work if we ignore collisions?
        if self.action_mode == 'translation':
            assert len(act_clip) == 3, act_clip
            dtheta = 0.
            axis = [0., -1., 0.]
        else:
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

        # I think we do this again to apply global constraints?
        new_tool_state, new_tool_state_tip = self._clip_bounds(
            new_tool_state,
            new_tool_state_tip,
            self.action_low_b,
            self.action_high_b,
        )

        # Update tool states
        self.tool_state = new_tool_state
        self.tool_state_tip = new_tool_state_tip

        # if self._tool_no_collide(new_tool_state) and self._tool_in_bounds(new_tool_state):
        #     # No collision _and_ tool within bounds? Update `self.tool_state`, etc!
        #     self.tool_state = new_tool_state
        #     self.tool_state_tip = new_tool_state_tip
        # else:
        #     # Old state becomes the curent state (we still have to copy!).
        #     self.tool_state[:, 3:6] = self.tool_state[:, :3].copy()
        #     self.tool_state[:, 10:] = self.tool_state[:, 6:10].copy()
        #     self.tool_state_tip[:, 3:6] = self.tool_state_tip[:, :3].copy()
        #     self.tool_state_tip[:, 10:] = self.tool_state_tip[:, 6:10].copy()

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

        # Return updated `states` which later gets assigned to `self.tool_state`.
        return (states, states_tip)

    def _get_collision_bounds(self):
        """Get collision bounds.

        Careful, for the axis_angle, we assume here that we can use same bounds
        as with yrot and translation-only. While a demonstrator might restrict
        itself to yrot even with using axis_angle formulation, a policy could
        learn arbitrary rotations and thus these bounds might not apply
        """
        if (self.action_mode not in ['translation']):
            print('Warning: We have no collision checking.')
            return self.action_low_b, self.action_high_b

        # Using tuned values. Actually I have found that this radius might be
        # too conservative so downscaling this, a hack I know.
        sphere_rad = self.SPHERE_RAD * 0.75  # test with collision alg policy
        abs_x_bound = self.glass_dis_x / 2. - self.border / 2. - sphere_rad
        abs_z_bound = self.glass_dis_z / 2. - self.border / 2. - sphere_rad

        # Not sure if high[1] is ever enforced?
        low  = np.array([-abs_x_bound,      self.border / 2. + sphere_rad, -abs_z_bound])
        high = np.array([ abs_x_bound, self.action_high_b[1] + sphere_rad,  abs_z_bound])
        return low, high

    def _clip_bounds(self, tool_state, tool_state_tip, low, high):
        """Only handling positions here for now.

        Still need to debug with the 4DoF action mode (translation_yrot).
        Note: when we clip we are assuming the 'position' of the tool is at the
        center of the ladle.
        """
        assert len(low) == len(high)
        if len(low) > 3:
            low = low[:3]
            high = high[:3]

        # Position of the new tool tip, this we don't need to approximate.
        tip_x = tool_state_tip[0,0]
        tip_y = tool_state_tip[0,1]
        tip_z = tool_state_tip[0,2]

        # Using our tuned values (to tune, make pyflex shapes at this spot).
        cx = tip_x + (self.SPHERE_RAD * np.cos(self.curr_roty))
        cy = tip_y - self.TIP_TO_CENTER_Y
        cz = tip_z + (self.SPHERE_RAD * np.sin(self.curr_roty))
        curr_pos = np.array([cx, cy, cz])

        # Actually this will sometimes fail, as `tool_state_tip` has updated info
        # but which is not present in the tool center info.
        # tool_center = self._get_tool_center()
        # assert np.array_equal(tool_center, curr_pos), f'{tool_center} {curr_pos}'

        ## Clip tool center pos (old way):
        #clipped_tool_pos = np.clip(tool_state[0, :3], low, high)
        #clip_offset = clipped_tool_pos - tool_state[0, :3]
        #tool_state[0, :3] = clipped_tool_pos

        # Clip tool center pos (new way):
        clipped_tool_pos = np.clip(curr_pos, low, high)
        clip_offset = clipped_tool_pos - curr_pos
        tool_state[0, :3] += clip_offset

        # Offset tool center followers
        tool_state[1, :3] += clip_offset
        tool_state_tip[0, :3] += clip_offset

        return tool_state, tool_state_tip

    def _tool_no_collide(self, new_tool_state, debug=False):
        """Checks if tool is within bounds AND collides (with the glass/box).

        This is a very coarse approximation, until we get a system set up that can do
        better collision checking. NOTE: we need the _new_ tool state here, not the
        `self.tool_state`, and we check indices [0:3] for positions, not [3:6].

        Assumes the box is "zero-centered" which it is (the origin is at the bottom).
        If we use the built-in data/bowl.obj, we have to tune the offset. If we use an
        imported tool that we build, make the origin at the center of the sphere which
        when cut off at the top equals the spoon/ladle. In this metho, that's what the
        'sphere' refers to.

        NOTE(daniel):
            (1) If we adjust the tool scale and tool data is in [0,1], we should
            also adjust the demonstrator and the (tx,tz) offsets and the radius.
            (2) For the tool we load (data 2) we'll assume we can ignore the stick.
            (3) Later could we use PyBullet p.getContactPoints()?
            (4) Also for the custom tool (data 2) we assume the sphere radius was 1
            in Blender, hence we can simply use self.tool_scale as the radius. EDIT:
            ah unfortunately we can't, the tool looks WAY too small for what we set
            the scale. Maybe NVIDIA does something under the hood. Unfortunately that
            means even more hacking. I thought the 'scale' would do the obvious thing?
            (5) Sadly, the 'center' of the tool we imported is at the 'upper left' of
            the tool at its bottom, when I import it. :( More hacking needed.
        """
        if self.action_mode != 'translation':
            print('Warning: We have no collision checking.')
            return True
        T = self.tool_idx

        # Set (tx, ty, tz) to be the center of the sphere formed from the ladle's bowl.
        # TODO(daniel) Hacky constants assume tool scale 0.15 for tool_data [0,1].
        # Similar constants for tool_data [2]. Test with: `policy_move_test_collisions`.
        if self.tool_data in [0, 1]:
            sphere_rad = 0.040
            tx = new_tool_state[T,0] + 0.075
            ty = new_tool_state[T,1] + 0.065
            tz = new_tool_state[T,2] + 0.075
        elif self.tool_data in [2, 3]:
            sphere_rad = 0.050
            tx = new_tool_state[T,0] + 0.090
            ty = new_tool_state[T,1] + 0.080
            tz = new_tool_state[T,2] + 0.090

        # NOTE(daniel): assumes that the dist_{x,z} can be divided by two, and that we
        # have to remove half of the border width, then subtract the sphere radius.
        cond_x = np.abs(tx) < (self.glass_dis_x / 2. - self.border / 2. - sphere_rad)
        cond_y =         ty > (                        self.border / 2. + sphere_rad)
        cond_z = np.abs(tz) < (self.glass_dis_z / 2. - self.border / 2. - sphere_rad)
        if debug and ((not cond_x) or (not cond_y) or (not cond_z)):
            if (not cond_x):
                print(f'x, state: {new_tool_state[0,0:6]}, glassx: {self.glass_dis_x:0.3f}')
            if (not cond_y):
                print(f' y, state: {new_tool_state[0,0:6]}')
            if (not cond_z):
                print(f'  z, state: {new_tool_state[0,0:6]}, glassz: {self.glass_dis_z:0.3f}')
        return cond_x and cond_y and cond_z

    def _tool_in_bounds(self, tool_state):
        """Check if the tool is in bounds.

        I think it suffices to check the current tool we use, and query its position.
        The position is a lower corner of the voxelized grid of the tool but it's a
        good enough approximation. Or maybe the tip is better.
        """
        in_x = self.action_low_b[0] <= tool_state[0,0] <= self.action_high_b[0]
        in_y = self.action_low_b[1] <= tool_state[0,1] <= self.action_high_b[1]
        in_z = self.action_low_b[2] <= tool_state[0,2] <= self.action_high_b[2]
        return in_x and in_y and in_z

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
        config['spheres_env_version'] = self.spheres_env_version
        return config

    def _item_in_bounds(self, item_pos):
        """Checks if item is in bounds and returns x and z conditions.
        Used in the info() methods for the envs, plus algorithmic policies.
        """
        in_bounds_x = np.abs(item_pos[0]) <= (self.glass_dis_x / 2.)
        in_bounds_z = np.abs(item_pos[2]) <= (self.glass_dis_z / 2.)
        return (in_bounds_x, in_bounds_z)

    def _get_tool_center(self):
        """Get the tool center. For the ladle, it's the center of its bowl.

        Added for the spheres env, but we could have done the same for mixed media.
        For mixed media, we put it in `_clip_bounds()`. For some of this we may want
        to make pyflex shapes at this spot. Also, SPHERE_RAD and TIP_TO_CENTER_Y
        depend on the env version (we might change the geometry of the ladle).

        Called in `_get_info()`, but the first time is in `_reset()` before the tip
        has been properly assigned. So the position is (-1,-1,-1)  but that should
        just mean the distance is very large and should not trigger success.

        Careful: sometimes we call this while updating a temporary variable for the
        tool state, but where we haven't assigned it to pyflex.
        """
        curr_shape_states = pyflex.get_shape_states()
        curr_shape_states = curr_shape_states.reshape((-1,14))
        tool_state_tip = curr_shape_states[self.tool_idx_tip, :]

        # Position of the tool tip, this we don't need to approximate.
        tip_x = tool_state_tip[0]
        tip_y = tool_state_tip[1]
        tip_z = tool_state_tip[2]

        # Using our tuned values, get the (to tune, make pyflex shapes at this spot).
        cx = tip_x + (self.SPHERE_RAD * np.cos(self.curr_roty))
        cy = tip_y - self.TIP_TO_CENTER_Y
        cz = tip_z + (self.SPHERE_RAD * np.sin(self.curr_roty))
        return np.array([cx, cy, cz])

    # --------------------------------------------------------------------------- #
    # NOTE(daniel) temporary debugging / testing of various algorithmic policies. #
    # --------------------------------------------------------------------------- #

    def _visualize_action_boundary(self):
        """From action_space.py, `Picker` class, to visualize the range."""
        half_edge = np.array(self.action_high_b - self.action_low_b) / 2.
        center = np.array(self.action_high_b + self.action_low_b) / 2.
        quat = np.array([1., 0., 0., 0.])
        pyflex.add_box(half_edge, center, quat)

    def _debug_tool_properties(self):
        """Debugs the diameter of the bowl in world space.

        Use case: can tune the center position and half edge length to measure
        the diameter of the bowl. Then we can use half of this to get the radius
        and thus if the ladle is rotated in some way about the y axis, we can
        roughly approximate the true center, which makes collision detection a
        lot more reliable.

        Actually after experimenting, if we have a value in half edge, it gets
        doubled (test: try putting 1), so take that into account.
        """
        half_edge = np.array([0.02, 0.01, 0.054])
        center = np.array([
            self.tool_state_tip[0,0],
            self.tool_state_tip[0,1] - 0.128,  # hack
            self.tool_state_tip[0,2] + 0.054,  # hack
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
        elif self.alg_policy == 'ladle_algorithmic_v01':
            # Algorithmic as of July 24.
            assert self.action_mode == 'translation', self.action_mode
            assert self.action_repeat == 8, self.action_repeat
            action = self._ladle_algorithmic_v01(self.inner_step)
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
        """Policy which deliberately moves to test collision checks."""
        dx, dy, dz, drot = 0., 0., 0., 0.
        if 0 <= step < 100:
            dy = -0.0045
        elif 100 <= step < 150:
            dx = -0.0030
        elif 150 <= step < 200:
            dz = -0.0030
        elif 200 <= step < 300:
            dx = 0.0030
        elif 300 <= step < 400:
            dz = 0.0030
            # Put in a few tests like this:
            #if self.action_mode == 'translation_yrot':
            #    dz = 0.
            #    drot = 0.02
        elif 400 <= step < 500:
            dx = -0.0030
        else:
            dy = 0.0045

        if self.action_mode == 'translation':
            action = np.array([dx, dy, dz])
        elif self.action_mode == 'translation_yrot':
            action = np.array([dx, dy, dz, drot])
        return action

    def _ladle_algorithmic_v01(self, step):
        """Move straight to the target ball."""
        return self.AlgPolicyCls.get_action()


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
        self._print = False
        self._magnitude = 0.004

    def reset(self):
        self.state = 0
        self._rest = 0

    def get_action(self):
        """Just move straight to the target."""

        # The tool and item centers (approximations).
        tool_center = self.env._get_tool_center()
        item_center = self.env._get_rigid_pos(item_index=0)[:3]
        if self.env.n_distractors >= 1:
            dist_center = self.env._get_rigid_pos(item_index=1)[:3]

        # Unfortunately distance thresholds have to be tuned carefully.
        direction = item_center - tool_center
        action = (direction / np.linalg.norm(direction)) * self._magnitude

        if self._print:
            distance_item = np.linalg.norm(tool_center - item_center)
            if self.env.n_distractors >= 1:
                distance_dist = np.linalg.norm(tool_center - dist_center)
                print((f'is {self.env.inner_step}, action {action}, '
                       f'distance item {distance_item:0.3f}, '
                       f'distance dist {distance_dist:0.3f}'))
            else:
                print((f'is {self.env.inner_step}, action {action}, '
                       f'distance item {distance_item:0.3f}'))
        return action
