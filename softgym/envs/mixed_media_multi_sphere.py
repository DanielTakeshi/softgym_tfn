import numpy as np
from numpy.random import uniform
import pyflex
import copy
from softgym.envs.mixed_media_env import MixedMediaEnv
from softgym.utils.misc import sample_sphere_points
np.set_printoptions(suppress=True, precision=4, linewidth=150)


class MMMultiSphereEnv(MixedMediaEnv):

    def __init__(self, cached_states_path='MMMultiSphere.pkl',
            n_substeps=2, n_iters=4, inv_dt=100, inv_mass=0.50, sphere_scale=0.060,
            act_noise=0.0, tool_type='sdf', tool_data=2, tool_scale=0.28, **kwargs):
        """A mixed media env with the aim of retrieving 1 of k spheres.

        Should have at least 2 spheres in the scene, ideally this iteractive setting might
        make it harder for a policy. For now this is only going to retrieve 1 item. Any
        other spheres will be labeled as distractors. We also need functionality to find
        the item associated with each particle.

        Env-specific assumptions (also see `MMOneSphereEnv`):
        (1) All items are the same size and thus consist of the same number of particles.
        (2) All items in the water are rigid and not rope/cloth.

        MM env versions for `self.mm_env_version`:
            1: 1 target + 3 distractors [edit: deprecating, please use v04]
            2: 1 target + 1 distractor
            3: 1 target + 2 distractors
            4: 1 target + 3 distractors [use this, version number makes more sense]
        All use 1 target and the same reward function, which only depends on the target
        (and essentially ignores the distractors).
        """
        self.reward_type = 'dense'
        self.name = 'MMMultiSphereEnv'
        self.mm_env_version = 2
        sp = f'_v{str(self.mm_env_version).zfill(2)}.pkl'
        self.cached_states_path = cached_states_path.replace('.pkl', sp)

        # Stuff to tune. For now: same as what's in `MMOneSphere`.
        self.n_substeps = n_substeps
        self.n_iters = n_iters
        self.inv_dt = inv_dt
        self.inv_mass = inv_mass
        self.sphere_scale = sphere_scale
        self.act_noise = act_noise
        self.tool_type = tool_type
        self.tool_data = tool_data
        self.tool_scale = tool_scale
        assert tool_type in ['sdf', 'tmesh', 'rbody'], tool_type
        assert tool_data in [0, 1, 2, 3], tool_data

        # For representations; do this before calling MixedMediaEnv's init.
        self.n_segm_classes = 5 + 1
        self.n_targets = 1
        if self.mm_env_version == 2:
            self.n_distractors = 1
        elif self.mm_env_version == 3:
            self.n_distractors = 2
        elif self.mm_env_version == 4:
            self.n_distractors = 3
        else:
            raise ValueError(self.mm_env_version)
        self.obs_dim_keypt = 8 + 4*self.n_distractors
        self.pc_point_dim = 3 + 3  # (x,y,z, onehot(tool), onehot(targ), onehot(dist))

        # Init MixedMediaEnv which inits FlexEnv.
        super().__init__(**kwargs)

        # Calls FlexEnv method, the first time the C++ Init() gets called.
        self.get_cached_configs_and_states(self.cached_states_path, self.num_variations)

    def get_default_config(self):
        """See supeclass method documentation."""
        config = self._get_superclass_default_config()
        config['item']['n_items'] = 1 + self.n_distractors
        return config

    def generate_env_variation(self, num_variations=1, **kwargs):
        """See docs from MMOneSphereEnv.

        I think we should slightly adjust the range of possible item starting states.
        We don't want items to be sampled on top of each other. Maybe a mesh grid?
        https://stackoverflow.com/questions/36013063/

        Basic idea: discretize the grid space above the water into a 3x3 grid:
            -------
            |x x o|
            |o x o|
            |x o x|
            -------
        Sample spheres (both the target and distractor) in one of those 9 components,
        for example the o's could indicate where we sample 4 items (if that is how
        many we are using). This just assigns the offsets to the config, the actual
        repositioning of items is in `self.set_scene()`. Note: positive x and z is in
        the lower right direction. Positive x is to the right.

        We could add noise on top of all this but I feel like this is sufficient.
        """
        self.cached_configs = []
        self.cached_init_states = []
        config = self.get_default_config()
        _off_x = config['item']['off_x']
        _off_y = config['item']['off_y']
        _off_z = config['item']['off_z']
        _r = config['fluid']['radius']
        config_variations = [copy.deepcopy(config) for _ in range(num_variations)]
        debug_print = False

        # Let's discretize the number of configurations for us to drop an item.
        # Linearize so `coords` = [(x_1,z_1), (x_2,z_1), ..., (x_k,z_k)]
        K = 3
        xx = np.linspace(-0.12, 0.05, num=K)
        zz = np.linspace(-0.12, 0.05, num=K)
        coords = [(xx[i], zz[j]) for i in range(K) for j in range(K)]
        n_items = config['item']['n_items']
        assert n_items <= len(coords), f'{n_items} vs {len(coords)}'

        idx = 0
        while idx < num_variations:
            # Get offsets that we need for the CENTER of the water. These offsets have
            # to be tuned carefully to ensure valid (i.e., above water) init spheres.
            config_variations[idx]['item']['off_x'] = _off_x #+ uniform(low=-0.12, high= 0.05)
            config_variations[idx]['item']['off_y'] = _off_y -0.05  #+ uniform(low=-0.10, high=-0.05)
            config_variations[idx]['item']['off_z'] = _off_z #+ uniform(low=-0.12, high= 0.05)

            # Put these in config variations, mainly to keep things contained here so
            # that all info about numpy randomness in sampling generation is here? Do
            # NOT sample with replacement, otherwise items are going to coincide. The
            # purpose is to provide a unique offset for each of the items, on top of
            # the normal _off_x, _off_z which brings the item to the center of the box.
            # So the above (PLUS the technique of moving particles to the center by
            # increasing the offsets due to the +1,+1 diagonal spacing) is for moving
            # above the center. This includes BOTH the distractor and target items.
            item_coords = np.random.choice(np.arange(len(coords)), size=n_items, replace=False)
            for n in range(n_items):
                # n is the item index, item_coords[n] is the index within coords to use
                xs, zs = coords[item_coords[n]]
                config_variations[idx][f'item_{n}_x'] = xs
                config_variations[idx][f'item_{n}_z'] = zs
            if debug_print:
                print('Sampling the items:')
                for _i,c in enumerate(coords):
                    print(_i, c)
                print(f'item_coords: {item_coords}')
                for key in sorted(list(config_variations[idx].keys())):
                    if 'item_' in key:
                        print(f'{key}:  {config_variations[idx][key]}')

            # ----- Back (mostly) to normal, copied from earlier ----- #
            print(f'\n{self.name}, variation {idx}')
            dim_x = config['fluid']['dim_x']
            dim_z = config['fluid']['dim_z']
            water_radius = _r * config['fluid']['rest_dis_coef']
            m = min(dim_x, dim_z)
            dim_y = int(4 * m)  # Daniel: _adjusted_ for lower dim_y!
            v = dim_x * dim_y * dim_z
            h = v / ((dim_x + 1) * (dim_z + 1)) * water_radius / 3.5
            glass_height = h + 0.1
            config_variations[idx]['fluid']['dim_x'] = dim_x
            config_variations[idx]['fluid']['dim_y'] = dim_y
            config_variations[idx]['fluid']['dim_z'] = dim_z
            config_variations[idx]['glass']['height'] = glass_height
            self.set_scene(config_variations[idx])
            init_state = copy.deepcopy(self.get_state())

            # Protect against sunk item(s), IF we want to do that.
            rigid_pos = self._get_rigid_pos()
            if rigid_pos[1] < self.sunk_height_cutoff:
                print(f'\n\nBad variation! {rigid_pos[1]} < {self.sunk_height_cutoff}')
                print(f'Ignoring re-trying {idx}...\n\n')
                continue

            # Otherwise, add it to the cached configs / states.
            self.cached_configs.append(config_variations[idx])
            self.cached_init_states.append(init_state)
            idx += 1
        return self.cached_configs, self.cached_init_states

    def set_scene(self, config, states=None, print_debug=False, render_debug=False):
        """Constructs the scene by calling PyFlex. See MMOneSphere for details.

        The only nontrivial change is a call to
            all_pos = self._get_item_pos_init(config)
        which is specific to this subclass to handle multiple items.
        """
        super().set_scene(config)

        # Add boxes for the tip, we will later reposition in `_reset()`.
        _idx = self._add_box_tool_tip()
        assert _idx == self.tool_idx_tip, _idx

        # Add boxes to better protect fake tool from flying water pellets.
        self._add_box_boundary()

        # Compute glass params.
        if states is None:
            self._set_glass_params(config["glass"])
        else:
            glass_params = states['glass_params']
            self.border = glass_params['border']
            self.height = glass_params['height']
            self.glass_dis_x = glass_params['glass_dis_x']
            self.glass_dis_z = glass_params['glass_dis_z']
            self.glass_params = glass_params

        # Create glass and move it to be at round or in table, assign to `self`.
        self._create_glass(self.glass_dis_x, self.glass_dis_z, self.height, self.border)
        self.glass_states = self._init_glass_state(
            self.x_center, 0, self.glass_dis_x, self.glass_dis_z, self.height, self.border)
        assert self.glass_states.shape == (5,14)
        self.glass_x = self.x_center  # glass floor center

        # Assign glass shape states (no need for tool now) using `self.wall_idxs`.
        curr_shape_states = pyflex.get_shape_states().reshape((-1,14))
        new_shape_states = np.copy(curr_shape_states)
        new_shape_states[self.wall_idxs, :] = self.glass_states
        pyflex.set_shape_states(new_shape_states)

        # Get the current positions, which are affected by how we init stuff.
        # We might want this info (e.g., indices, etc.) for later.
        self.rigid_idxs = pyflex.get_rigidIndices()
        self.fluid_idxs = [x for x in range(self.particle_num) if x not in self.rigid_idxs]
        self.n_fluid_particles = n_fluid_particles = self.particle_num - len(self.rigid_idxs)
        self._set_particle_to_shape(config['item']['n_items'])
        all_particle_pos = self.get_state()['particle_pos'].reshape((-1, self.dim_position))

        # Initialize scene with w/water moving + respawning in glass, etc.
        if states is None:
            # Move water all inside the glass
            fluid_radius = self.fluid_params['radius'] * self.fluid_params['rest_dis_coef']
            fluid_dis = np.array([1.0 * fluid_radius, fluid_radius * 0.5, 1.0 * fluid_radius])
            lower_x = self.glass_params['glass_x_center'] - self.glass_params['glass_dis_x'] / 2. + self.glass_params['border']
            lower_z = -self.glass_params['glass_dis_z'] / 2 + self.glass_params['border']
            lower_y = self.glass_params['border']
            if self.action_mode in ['sawyer', 'franka']:
                lower_y += 0.56
            lower = np.array([lower_x, lower_y, lower_z])

            # Daniel: iterates rx,ry,rz, which assumes all particles are fluid.
            cnt = 0
            rx = int(self.fluid_params['dim_x'] * 1)
            ry = int(self.fluid_params['dim_y'] * 1)
            rz = int(self.fluid_params['dim_z'] / 1)
            assert rx*ry*rz == n_fluid_particles, f'{rx*ry*rz} vs {n_fluid_particles}'
            for x in range(rx):
                for y in range(ry):
                    for z in range(rz):
                        while cnt in self.rigid_idxs:
                            # skip over indices corresponding to rigid items, only re-scale liquid
                            cnt += 1
                        all_particle_pos[cnt][:3] = lower + np.array([x, y, z]) * fluid_dis
                        cnt += 1

            # Daniel: set positions for _all_ particles.
            pyflex.set_positions(all_particle_pos)
            if print_debug:
                print(f"{self.name}.set_scene(), stabilizing water now ...")
            for _ in range(120):
                pyflex.step()
                if render_debug:
                    pyflex.render()

            # NOTE(daniel) `water_state` includes any rigids.
            state_dic = self.get_state()
            water_state = state_dic['particle_pos'].reshape((-1, self.dim_position))
            in_glass = self._in_glass(
                water_state, self.glass_states, self.border, self.height, return_sum=False)
            not_total_num = np.sum(1 - in_glass)

            # Repeatedly put all water particles outside back in the glass.
            it = 0
            while not_total_num > 0:
                it += 1
                if it >= 4:
                    break

                max_height_now = np.max(water_state[self.fluid_idxs, 1])
                fluid_dis = np.array([1.0 * fluid_radius, fluid_radius * 1, 1.0 * fluid_radius])
                lower_x = self.glass_params['glass_x_center'] - self.glass_params['glass_dis_x'] / 4
                lower_z = -self.glass_params['glass_dis_z'] / 4
                lower_y = max_height_now
                lower = np.array([lower_x, lower_y, lower_z])
                cnt = 0
                dim_x = config['fluid']['dim_x']
                dim_z = config['fluid']['dim_z']
                # Daniel: `water_state` has every index, so ignore any for 'rigid' particles.
                for w_idx in range(len(water_state)):
                    if (w_idx not in self.rigid_idxs) and not in_glass[w_idx]:
                        water_state[w_idx][:3] = lower + \
                            fluid_dis * np.array([cnt % dim_x, cnt // (dim_x * dim_z), (cnt // dim_x) % dim_z])
                        cnt += 1

                if print_debug:
                    print(f"{self.name}.set_scene(), not_total_num: {not_total_num} ...")
                pyflex.set_positions(water_state)
                for _ in range(120):
                    pyflex.step()
                    if render_debug:
                        pyflex.render()

                # NOTE(daniel) `water_state` includes any rigids.
                state_dic = self.get_state()
                water_state = state_dic['particle_pos'].reshape((-1, self.dim_position))
                in_glass = self._in_glass(
                    water_state, self.glass_states, self.border, self.height, return_sum=False)
                not_total_num = np.sum(1 - in_glass)

            # Move the items over the liquid and simulate.
            all_pos = self._get_item_pos_init(config)  # Specific to multi-sphere case!
            pyflex.set_positions(all_pos)
            if print_debug:
                print(f"{self.name}.set_scene() simulating items over liquid ...")
            for _ in range(200):
                pyflex.step()
                if render_debug:
                    pyflex.render()

            # Move tool back above the center of water? Some tuning needed, unfortunately.
            self.tool_state = self._get_init_tool_state()
            curr_shape_states = pyflex.get_shape_states().reshape((-1,14))
            new_shape_states = np.copy(curr_shape_states)
            new_shape_states[:2,:] = self.tool_state
            pyflex.set_shape_states(new_shape_states)
        else:
            # set to passed-in cached init states
            self.set_state(states)

        # Helps with possible segmentation and for detecting bad initial states.
        self.water_height_init = self._get_water_height_init()
        self.sunk_height_cutoff = self.water_height_init / 2.0
        self.glass_height_init = self.height
        assert self.water_height_init < self.glass_height_init, \
            f'{self.water_height_init} vs {self.glass_height_init}'
        if print_debug:
            print(f'{self.name} after set_state, item pos: {self._get_rigid_pos()}')

        # If not loading the cache, we call `set_scene` twice so need to set tool again.
        self.tool_state = self._get_init_tool_state()
        curr_shape_states = pyflex.get_shape_states().reshape((-1,14))
        new_shape_states = np.copy(curr_shape_states)
        new_shape_states[:2, :] = self.tool_state
        pyflex.set_shape_states(new_shape_states)

    def _get_obs(self):
        """Return the observation based on the current flex state.

        See `MMOneSphereEnv` for documentation. Here, the main changes have to do
        with the distractors. Also, we're supporting a new type of point cloud obs,
        `point_cloud_gt_v01`, which allows for sampling points from known pyflex
        positions of the ball, even if it happens to be occluded. This also uses
        the segmentation to reduce code changes (but please ignore the ball segm.
        to save a LOT of compute).
        """
        def get_keypoints():
            pos_tool_tip = self.tool_state_tip[0,:3]  # tool_state_tip is shape (1,14)
            pos_sphere = self._get_rigid_pos()[:3]  # 4D, but we only want 3D
            rad_sphere = self._get_rigid_radius()
            particles = self.get_state()['particle_pos'].reshape((-1, self.dim_position))
            water_state = particles [self.fluid_idxs, :]
            water_bools = self._in_glass(water_state, self.glass_states, self.border,
                    self.height, return_sum=False)
            water_frac = np.sum(water_bools) / len(water_bools)

            # Now the distractors.
            distractors = []
            for k in range(self.n_distractors):
                pos_distractor_k = self._get_rigid_pos(item_index=k+1)[:3]
                rad_distractor_k = self._get_rigid_radius(item_index=k+1)
                distractors.extend([*pos_distractor_k, rad_distractor_k])

            # Concatenate everything.
            obs = np.concatenate(
                    [pos_tool_tip, pos_sphere, [rad_sphere], distractors, [water_frac]])
            assert len(obs) == self.obs_dim_keypt, len(obs)
            return obs

        def get_segm_img(with_depth=False):
            # Get RGB(D) images, possibly from multiple viewpoints. Then segment.
            images_dict = self.segm.query_images()
            segm_img = self.segm.segment(images=images_dict)
            if with_depth:
                return segm_img, images_dict['depth']
            return segm_img

        def get_pointcloud_array():
            # Must be called AFTER segmentation! Then turn `pc` to an array.
            pc = self.segm.get_pointclouds()
            # Also handle one-hot classes with subsampling (tool, targ).
            # If subsampling PC _and_ we have tool flow, we should subsample
            # tool flow (so the tool pts coincide with the PC array's tool pts).
            pc_tool = pc['tool']
            pc_targ = pc['target']
            pc_dist = pc['distractor']
            n1, n2, n3 = len(pc_tool), len(pc_targ), len(pc_dist)
            n_pts = n1 + n2 + n3
            pc_array = np.zeros((max(n_pts, self.max_pts), self.pc_point_dim))
            pc_array[           :n1, :3] = pc_tool
            pc_array[           :n1,  3] = 1.
            pc_array[      n1:n1+n2, :3] = pc_targ
            pc_array[      n1:n1+n2,  4] = 1.
            pc_array[n1+n2:n1+n2+n3, :3] = pc_dist
            pc_array[n1+n2:n1+n2+n3,  5] = 1.

            # For compatibility with tool flow, order so all tool points come first.
            # Should save the idxs we used to inform the tool flow subsampling.
            idxs = np.arange(n_pts)  # including if we don't need to subsample
            if n_pts > self.max_pts:
                idxs = np.sort( np.random.permutation(n_pts)[:self.max_pts] )
                pc_array = pc_array[idxs, :]
            self.segm.set_subsampling_tool_flow(idxs, n_tool=len(pc_tool))

            # Record amount of such points for debugging analysis later.
            self.pcl_dict['tool_raw'] = n1
            self.pcl_dict['targ_raw'] = n2
            self.pcl_dict['dist_raw'] = n3
            self.pcl_dict['tool_subs'] = len(np.where(pc_array[:,3] == 1)[0])
            self.pcl_dict['targ_subs'] = len(np.where(pc_array[:,4] == 1)[0])
            self.pcl_dict['dist_subs'] = len(np.where(pc_array[:,5] == 1)[0])
            return pc_array

        def get_pointcloud_array_gt(pts_sphere):
            # Another variant, now with ball points even if occluded.
            assert self.n_distractors == 1, self.n_distractors  # only tested this
            pc = self.segm.get_pointclouds()
            # If subsampling PCL _and_ we have tool flow, we should subsample
            # tool flow (so the tool pts coincide with the PC array's tool pts).
            pc_tool = pc['tool']

            # Sample sphere points, center at `targ_pos` and `dist_pos`.
            targ_pos = self._get_rigid_pos(item_index=0)[:3]
            dist_pos = self._get_rigid_pos(item_index=1)[:3]
            targ_samp_pts = sample_sphere_points(pts_sphere, radius=self._sphere_radius_PCL)
            dist_samp_pts = sample_sphere_points(pts_sphere, radius=self._sphere_radius_PCL)
            pc_targ = targ_pos + targ_samp_pts
            pc_dist = dist_pos + dist_samp_pts

            # Let's reserve the last `2 * pts_sphere` points for the spheres.
            n1, n2, n3 = len(pc_tool), len(pc_targ), len(pc_dist)
            n_pts = n1 + n2 + n3
            pc_array = np.zeros((max(n_pts, self.max_pts), self.pc_point_dim))
            pc_array[           :n1, :3] = pc_tool
            pc_array[           :n1,  3] = 1.
            pc_array[      n1:n1+n2, :3] = pc_targ
            pc_array[      n1:n1+n2,  4] = 1.
            pc_array[n1+n2:n1+n2+n3, :3] = pc_dist
            pc_array[n1+n2:n1+n2+n3,  5] = 1.

            # For compatibility with tool flow, order so all tool points come first.
            # Should save the idxs we used to inform the tool flow subsampling.
            idxs = np.arange(n_pts)  # including if we don't need to subsample
            if n_pts > self.max_pts:
                # Only subsample the tool (NOT the balls). Offset both n_pts and
                # max_pts by total sphere points, then subsample just the tool.
                new_max_pts = self.max_pts - (2 * pts_sphere)
                assert new_max_pts > 0
                idxs = np.sort( np.random.permutation(n_pts - 2*pts_sphere)[:new_max_pts] )
                idxs = np.concatenate( [idxs, np.arange(n_pts - 2*pts_sphere, n_pts)] )
                pc_array = pc_array[idxs, :]
            self.segm.set_subsampling_tool_flow(idxs, n_tool=len(pc_tool))

            assert len(np.where(pc_array[:,4] == 1)[0]) == pts_sphere
            assert len(np.where(pc_array[:,5] == 1)[0]) == pts_sphere
            return pc_array

        def get_tool_flow():
            return self.segm.get_tool_flow(self.tool_state, self.tool_state_tip)

        # MUST Call `get_segm_img()` before `self.segm.get_pointclouds()`!
        # MUST Call `get_tool_flow()` before `get_pointcloud_array()`!
        if self.observation_mode == 'cam_rgb':
            return self.get_image(self.camera_width, self.camera_height)
        elif self.observation_mode == 'cam_rgbd':
            # 08/18/2022: now RGBD for (width,height,4)-sized image. NOTE(daniel):
            # careful, RGB images in [0,255] (uint8) so check for any division by
            # 255 at runtime. Depth has values in [0,x] where x is 'SoftGym units',
            # usually x is about 1. Depth is (width,height) --> (width,height,1).
            cam_rgb = self.get_image(self.camera_width, self.camera_height)
            _, depth_img = get_segm_img(with_depth=True)
            rgbd_img = np.concatenate(
                (cam_rgb.astype(np.float32), depth_img[...,None]), axis=2)
            return rgbd_img
        elif self.observation_mode == 'key_point':
            return get_keypoints()
        elif self.observation_mode == 'segm':
            return get_segm_img()
        elif self.observation_mode == 'point_cloud':
            _ = get_segm_img()
            return get_pointcloud_array()
        elif self.observation_mode == 'point_cloud_gt_v01':
            _ = get_segm_img()
            return get_pointcloud_array_gt(pts_sphere=250)
        elif self.observation_mode == 'point_cloud_gt_v02':
            _ = get_segm_img()
            return get_pointcloud_array_gt(pts_sphere=600)
        elif self.observation_mode == 'flow':
            _ = get_segm_img()
            tool_flow = get_tool_flow()
            pc_array = get_pointcloud_array()
            return (pc_array, tool_flow)
        elif self.observation_mode == 'combo':
            # Usually for comparing same algorithms with different observations.
            keypts = get_keypoints()
            img_rgb = self.get_image(self.camera_width, self.camera_height)
            segm_img = get_segm_img()
            tool_flow = get_tool_flow()
            pc_array = get_pointcloud_array()
            return (keypts, img_rgb, segm_img, pc_array, tool_flow)
        elif self.observation_mode == 'combo_gt_v01':
            # Usually for comparing same algorithms with different observations.
            # (06/07) add g.t. PCL array (at the end of this tuple). Actually
            # even with this, due to subsampling tool flow we kind of have to
            # pick one of these point clouds, so we may not be able to compare.
            keypts = get_keypoints()
            img_rgb = self.get_image(self.camera_width, self.camera_height)
            segm_img = get_segm_img()
            tool_flow = get_tool_flow()
            pc_array_gt_v01 = get_pointcloud_array_gt(pts_sphere=250)
            return (keypts, img_rgb, segm_img, pc_array_gt_v01, tool_flow)
        elif self.observation_mode == 'combo_gt_v02':
            # Usually for comparing same algorithms with different observations.
            # (07/20) add g.t. PCL array v02 with more ball points.
            keypts = get_keypoints()
            img_rgb = self.get_image(self.camera_width, self.camera_height)
            segm_img = get_segm_img()
            tool_flow = get_tool_flow()
            pc_array_gt_v02 = get_pointcloud_array_gt(pts_sphere=600)
            return (keypts, img_rgb, segm_img, pc_array_gt_v02, tool_flow)
        else:
            raise NotImplementedError()

    def compute_reward(self, obs=None, action=None, set_prev_reward=False):
        """Reward function, both dense and sparse.

        See MMOneSphere for docs.
        For now we _could_ try and see what happens with JUST the reward for getting the
        target item (and completely ignoring distractors), I'd be curious if the policy
        naturally retrieves just the target item alone.
        """
        item_pos = self._get_rigid_pos(item_index=0)
        item_radius = self._get_rigid_radius(item_index=0)
        shape_pos = self.get_state()['shape_pos'].reshape((-1, 14))
        height_tool = shape_pos[self.tool_idx, 1]
        height_item = item_pos[1]
        sparse_reward = (height_item - item_radius*1.5 > height_tool) and \
                (height_item > self.height_cutoff)

        # Dense reward, the target item height (except if out of bounds).
        dense_reward = height_item - (self.water_height_init + 2. * item_radius)

        ## New to this, subtract if we have any other items? TODO(daniel) not now?
        #for k in range(self.n_distractors):
        #    distractor_height = self._get_rigid_pos(item_index=k+1)[1]
        #    distractor_radius = self._get_rigid_radius(item_index=k+1)
        #    distractor_rew = distractor_height - (self.water_height_init + 2. * distractor_radius)
        #    # Also need to enforce in-bounds constraint?
        #    if distractor_rew >= 0:
        #        dense_reward -= distractor_rew

        # Also enforce that the item is within the glass bounds.
        in_bounds_x, in_bounds_z = self._item_in_bounds(item_pos)
        sparse_reward = sparse_reward and in_bounds_x and in_bounds_z
        dense_reward = dense_reward if in_bounds_x and in_bounds_z else -1.

        # If all this is true, then we _still_ need time steps exceeded.
        if sparse_reward:
            self.time_exceeded += 1
        else:
            self.time_exceeded = 0
        sparse_reward = sparse_reward and (self.time_exceeded >= self.time_cutoff)

        # ---------------------------------------------------------------------------- #
        # (Only for analysis) Check the distractor, see if it satisfies condition for
        # this step, and if so increment the counter, if not reset (important!).
        dist_1_pos = self._get_rigid_pos(item_index=1)
        dist_1_radius = self._get_rigid_radius(item_index=1)
        dist_1_height = dist_1_pos[1]
        dist_in_x, dist_in_z = self._item_in_bounds(dist_1_pos)
        dist_1_sparse = ((dist_1_height - dist_1_radius*1.5 > height_tool) and
                         (dist_1_height > self.height_cutoff) and
                         (dist_in_x and dist_in_z))
        if dist_1_sparse:
            self.time_exceeded_dist_1 += 1
        else:
            self.time_exceeded_dist_1 = 0
        # ---------------------------------------------------------------------------- #

        if self.reward_type == 'dense':
            return dense_reward
        elif self.reward_type == 'sparse':
            return int(sparse_reward)
        else:
            raise NotImplementedError(self.reward_type)

    def _get_info(self):
        """Extra info, not used by the agent.

        See MMOneSphere. Only nontrivial change here is tracking height of all items.
        """
        state_dic = self.get_state()
        item_pos = self._get_rigid_pos(item_index=0)
        item_radius = self._get_rigid_radius(item_index=0)
        shape_pos = state_dic['shape_pos'].reshape((-1, 14))
        height_tool = shape_pos[self.tool_idx, 1]
        height_item = item_pos[1]

        # Try to compare the 'bottom' of the tool and the item, where for the item,
        # we get its position, then subtract its radius (edit: doing 1.5x that).
        sparse_reward = (height_item - item_radius*1.5 > height_tool) and \
                (height_item > self.height_cutoff)
        stuck_item = (height_item - item_radius*1.5 <= height_tool) and \
                (height_item > self.height_cutoff)

        # Check the in bounds condition.
        in_bounds_x, in_bounds_z = self._item_in_bounds(item_pos)
        sparse_reward = sparse_reward and in_bounds_x and in_bounds_z

        # Check the time exceeded call again. Do NOT update self.time_exceeded!!
        sparse_reward = sparse_reward and (self.time_exceeded >= self.time_cutoff)

        # Get the water state, assuming we can query fluid_idxs.
        all_pos = state_dic['particle_pos'].reshape((-1, self.dim_position))
        water_state = all_pos[self.fluid_idxs, :]
        water_num = len(water_state)
        in_glass = self._in_glass(water_state, self.glass_states, self.border, self.height)
        out_glass = water_num - in_glass

        # ---------------------------------------------------------------------------- #
        # (Only for analysis) Check if the distractor would satisfy such a condition.
        dist_1_pos = self._get_rigid_pos(item_index=1)
        dist_1_radius = self._get_rigid_radius(item_index=1)
        dist_1_height = dist_1_pos[1]
        dist_in_x, dist_in_z = self._item_in_bounds(dist_1_pos)
        dist_1_sparse = ((dist_1_height - dist_1_radius*1.5 > height_tool) and
                         (dist_1_height > self.height_cutoff) and
                         (dist_in_x and dist_in_z) and
                         (self.time_exceeded_dist_1 >= self.time_cutoff))
        # ---------------------------------------------------------------------------- #

        info = {
            'done': int(sparse_reward) == 1,
            'dist_1_done':     int(dist_1_sparse) == 1,
            'done_and_dist_1': int(sparse_reward and dist_1_sparse),
            'done_no_dist_1':  int(sparse_reward and not dist_1_sparse),
            'no_done_dist_1':  int(not sparse_reward and dist_1_sparse),
            'time_exceeded': self.time_exceeded,
            'item_radius': item_radius,
            'height_item': height_item,
            'height_tool': height_tool,
            'water_inside': in_glass,
            'water_outside': out_glass,
            'in_bounds_x': in_bounds_x,
            'in_bounds_z': in_bounds_z,
            'stuck_item': stuck_item,
            'performance': 0,  # TODO(daniel) check normalized performance in SoftAgent
            'curr_roty': self.curr_roty,  # NOTE(daniel) mainly to evaluate rotation
        }
        if self.observation_mode in ['point_cloud', 'combo'] and self.pcl_dict:
            info['pcl_tool_raw'] = self.pcl_dict['tool_raw']
            info['pcl_targ_raw'] = self.pcl_dict['targ_raw']
            info['pcl_dist_raw'] = self.pcl_dict['dist_raw']
            info['pcl_tool_subs'] = self.pcl_dict['tool_subs']
            info['pcl_targ_subs'] = self.pcl_dict['targ_subs']
            info['pcl_dist_subs'] = self.pcl_dict['dist_subs']

        # New to multi-sphere case, add heights for the other items.
        for k in range(1, self.n_distractors+1):
            item_pos_k = self._get_rigid_pos(item_index=k)
            in_bounds_k_x = np.abs(item_pos_k[0]) <= (self.glass_dis_x / 2.)
            in_bounds_k_z = np.abs(item_pos_k[2]) <= (self.glass_dis_z / 2.)
            info[f'height_item_{k}'] = item_pos_k[1]
            info[f'inbounds_item_{k}'] = in_bounds_k_x and in_bounds_k_z
        return info

    def _get_item_pos_init(self, config):
        """Get adjusted item positions during initialization (for multi-item case).

        During the env creation, we put items in an offset pattern so that we could
        simulate liquid, then move the items over it. Return the adjusted position
        array to be assigned by pyflex in the subclass.

        In C++, items (spheres) get initialized starting at the (2,2) coordinate and
        increasing to (2+i, 2+i) for integer index i. Thus, from the config, get each
        item's off_{x,y,z} keys which show how much to ADD for the FIRST item. For
        any other items later, follow the convention that we add 1 to the (x,z).

        And we have to be careful with the height because otherwise too high means
        that stuff will sink. Maybe we should adjust the way we randomize? I think
        we should do this: sample over (x,z) but ensure that the sampled points are
        not going to be within some radius of each other (in the (x,z) plane). For
        now I'm approximating assuming we can get a mesh grid of samples beforehand.
        But I wonder if that will be enough to avoid items colliding with each other?
        """
        K = len(self.rigid_idxs)
        all_pos = self.get_state()['particle_pos'].reshape((-1, self.dim_position))
        n_items = config['item']['n_items']

        # Shape (K,3) for offsets, but need to adjust!
        offset = np.concatenate(
            (np.ones((K,1)) * config['item']['off_x'],
             np.ones((K,1)) * config['item']['off_y'],
             np.ones((K,1)) * config['item']['off_z'],),
            axis=1)

        # Handle multiple item case, using particle_to_item to help.
        # Actually we are just overriding with item_n_{x,z} because that includes
        # the off_x and off_y values from earlier. Then we need the subtraction
        # by n because we create items in a diagonal pattern: (2,2), (3,3), etc.
        for n in range(n_items):
            particle_idxs = np.where(self.particle_to_item == n)[0]
            offset[particle_idxs, 0] += (config[f'item_{n}_x'] - n)
            offset[particle_idxs, 2] += (config[f'item_{n}_z'] - n)

        # Again assume rigid_idxs starts from 0 (not for this line but for others).
        all_pos[self.rigid_idxs, :3] += offset
        return all_pos


if __name__ == '__main__':
    env = MMMultiSphereEnv(observation_mode='cam_rgb',
                           action_mode='direct',
                           render=True,
                           headless=False,
                           horizon=100,
                           action_repeat=8,
                           render_mode='fluid',
                           deterministic=True,
                           num_variations=1,
                           camera_name='top_down')
    env.reset()
    for i in range(1000):
        pyflex.step()
