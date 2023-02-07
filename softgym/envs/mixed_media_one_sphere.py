import numpy as np
from numpy.random import uniform
import pyflex
import copy
from softgym.envs.mixed_media_env import MixedMediaEnv
from softgym.utils.misc import sample_sphere_points
np.set_printoptions(suppress=True, precision=4, linewidth=150)
DEG_TO_RAD = np.pi / 180.
RAD_TO_DEG = 180 / np.pi


class MMOneSphereEnv(MixedMediaEnv):

    def __init__(self, cached_states_path='MMOneSphere.pkl',
            n_substeps=2, n_iters=4, inv_dt=100, inv_mass=0.50, sphere_scale=0.060,
            act_noise=0.0, tool_type='sdf', tool_data=2, tool_scale=0.28, **kwargs):
        """A mixed media env with the aim of retrieving 1 sphere.

        Should have 1 sphere, and for now it's floating. This env is a simple testbed
        to check if we can simulate this at all, an algorithmic policy should do well.
        Has a 'fake tool' to help with segmentation of the tool.

        Env-specific assumptions:
        (1) That the ladle is visible at the start, particularly its tip.
        (2) That we can initialize a lot of shapes 'out of the way' then move then to a
            'good' starting configuration during `set_scene()` or, later, `_reset()`.
        (3) Shape indices: keep indices 0 and 1 as the shape indices for the real and fake
            tool. Index 2 is for the box we add to the real tool's tip, which is helpful
            if we include rotations of the ladle. Indices 3 and 4 are for the two walls
            which help stop water particles from shooting at the fake tool (thus affecting
            depth image). The last 5 indices are for the glass itself. These are added in
            order and must be consistently indexed for pyflex shapes!
        """
        self.reward_type = 'dense'
        self.name = 'MMOneSphereEnv'
        self.mm_env_version = 2
        sp = f'_v{str(self.mm_env_version).zfill(2)}.pkl'
        self.cached_states_path = cached_states_path.replace('.pkl', sp)

        # Stuff to tune. For now: same as what's in `MMMultiSphere`.
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

        # See `bindings/softgym_scenes/softgym_mixed_media.h`.
        if self.mm_env_version == 1:
            # standard ladle
            assert tool_data == 2, f'{tool_data}, mm: {self.mm_env_version}'
        elif self.mm_env_version == 2:
            # ladle with hole
            if tool_data == 2:
                print('Warning, tool_data is 2, changing to 4')
                tool_data = 4
                self.tool_data = 4
            assert tool_data == 4, f'{tool_data}, mm: {self.mm_env_version}'
        else:
            raise ValueError(self.mm_env_version)

        # For representations; do this before calling MixedMediaEnv's init.
        self.n_segm_classes = 5
        self.n_targets = 1
        self.n_distractors = 0
        self.obs_dim_keypt = 8
        self.pc_point_dim = 3 + 2  # (x,y,z, onehot(tool), onehot(targ))

        # Init MixedMediaEnv which inits FlexEnv then does obs/act modes.
        super().__init__(**kwargs)

        # Calls FlexEnv method, the first time the C++ Init() gets called.
        self.get_cached_configs_and_states(self.cached_states_path, self.num_variations)

    def get_default_config(self):
        """See supeclass method documentation."""
        config = self._get_superclass_default_config()
        config['item']['n_items'] = 1
        return config

    def generate_env_variation(self, num_variations=1, **kwargs):
        """
        Daniel: Called from FlexEnv when it generates env variation (potentially
        after loading a cached config), and then here we can further modify the
        config, and then critically query `set_scene()` which will call PyFlex.
        Increasing dim_x, dim_y, dim_z will increase number of water particles
        because we put that in the fluid configs. But we also scale height based
        on this so the box scales as well.

        Straightforward, # of liquid particles is `dim_x * dim_y * dim_z`.
            (dim_x, dim_y, dim_z) = ( 6, 36,  6), --> 1296 particles (default).
            (dim_x, dim_y, dim_z) = ( 8, 48,  8), --> 3072 particles.
            (dim_x, dim_y, dim_z) = (10, 60, 10), --> 6000 particles.
        When increasing dim_x, dim_z, the dim_y is computed for us, and we
        also probably want to tune glass_height.

        Update 10/24/2021: we generate variations by sampling from dim_x, dim_z. But
        let's change this to sample where the item gets dropped, but otherwise keep
        all the other stuff the same.

        Returns: cached versions of (configs, states).
        """
        self.cached_configs = []
        self.cached_init_states = []
        config = self.get_default_config()
        _off_x = config['item']['off_x']
        _off_y = config['item']['off_y']
        _off_z = config['item']['off_z']
        _r = config['fluid']['radius']
        config_variations = [copy.deepcopy(config) for _ in range(num_variations)]

        idx = 0
        while idx < num_variations:
            # Sample the item (the sphere) position across the water. It's centered
            # at the provided `_off_{x,y,z}` from the default config. These offsets
            # have to be tuned carefully, BTW, to ensure valid init spheres.
            if self.mm_env_version == 1:
                config_variations[idx]['item']['off_x'] = _off_x + uniform(low=-0.12, high= 0.05)
                config_variations[idx]['item']['off_y'] = _off_y + uniform(low=-0.10, high=-0.05)
                config_variations[idx]['item']['off_z'] = _off_z + uniform(low=-0.12, high= 0.05)
            elif self.mm_env_version == 2:
                # Cut high x value and low z value.
                config_variations[idx]['item']['off_x'] = _off_x + uniform(low=-0.12, high=-0.02)
                config_variations[idx]['item']['off_y'] = _off_y + uniform(low=-0.10, high=-0.05)
                config_variations[idx]['item']['off_z'] = _off_z + uniform(low=-0.02, high= 0.05)
                # Should also change the ladle's start. Introduce these new keys.
                assert 'init_x' not in config_variations[idx]['tool']
                assert 'init_y' not in config_variations[idx]['tool']
                assert 'init_z' not in config_variations[idx]['tool']
                config_variations[idx]['tool']['init_x'] = \
                        self.tool_init_x + uniform(low= 0.02, high= 0.04)
                config_variations[idx]['tool']['init_y'] = \
                        self.tool_init_y + uniform(low= 0.00, high= 0.00)
                config_variations[idx]['tool']['init_z'] = \
                        self.tool_init_z + uniform(low=-0.03, high=-0.01)
            else:
                raise ValueError(self.mm_env_version)
            # ---------- Back (mostly) to normal, copied from earlier ---------- #
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
        """Constructs the scene by calling PyFlex.

        Called from `generate_env_variation()` with `states=None`. First, call superclass
        `set_scene` which calls the critical `pyflex.set_scene()` and thus C++ code. If we
        add tools and other shapes in the C++ file as shapes, those are the first shapes,
        the glass stuff happens after.

        Also called from FlexEnv.reset() when we've loaded in a cached state, this will use
        the `states` argument as the config was previously constructed.

        General idea: build shapes, then sample water, then repeatedly put water particles
        back in, then move the item over above the water and drop it. (Why? Water particles
        tend to seep through the walls with high motion, hence putting them back in seems to
        improve simulator stability.) For moving the item (i.e., shere), probably easiest to
        translate all its particles by the same amount? That way it maintains its geometry.

        NOTE(daniel) see documentation above regarding shape indices! Very important!
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
            all_pos = self.get_state()['particle_pos'].reshape((-1, self.dim_position))
            _offset = np.array([config['item']['off_x'],
                                config['item']['off_y'],
                                config['item']['off_z'],
                                0.])
            all_pos[self.rigid_idxs] += _offset
            pyflex.set_positions(all_pos)
            if print_debug:
                print(f"{self.name}.set_scene() simulating items over liquid ...")
            for _ in range(200):
                pyflex.step()
                if render_debug:
                    pyflex.render()

            # Move tool back above the center of water? Some tuning needed, unfortunately.
            self.tool_state = self._get_init_tool_state(new_config=config)
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
        self.tool_state = self._get_init_tool_state(new_config=config)
        curr_shape_states = pyflex.get_shape_states().reshape((-1,14))
        new_shape_states = np.copy(curr_shape_states)
        new_shape_states[:2, :] = self.tool_state
        pyflex.set_shape_states(new_shape_states)

    def _get_obs(self):
        """Return the observation based on the current flex state.

        Called from FlexEnv.reset() and FlexEnv.step(); not called if only generating
        variations to be cached. The obs mode should be independent of the variation;
        given some variation, we should be able to load it and use any obs mode.

        The obs space should probably have an assigned shape (see superclass) that
        makes sense but this is rarely enforced since (a) we don't tend to normalize
        envs, and (b) in SoftAgent code we can detect + enforce correct obs shapes.

        cam_rgb:
            The usual RGB image. See `FlexEnv` for implementation.

        key_point (i.e., 'reduced state' mode):
            A single vector with this information:
                position (x,y,z) of the ladle tip  TODO(daniel) change if rotating.
                position (x,y,z) of the sphere center
                radius of the sphere
                water fraction in cup
            Thus we use 8-D observation, and we will change if the ladle uses rotations.
            The cup (also called box or glass) does not change so we don't need that.
            Is there a better way to get position than `self._get_rigid_pos()`? We also
            use water fraction instead of height, because water might be scooped out by
            the ladle, so using the raw water height is a bit tricky.

        segm (what we hope is better):
            5 classes: nothing/background, glass/box, water, item/sphere, tool.
            Returns an image of dimension (width, height, 5) where each channel is the
            segmentation mask for one item. The precise ordering is:
                img[:, :, 0] = nothing/background
                img[:, :, 1] = glass/box
                img[:, :, 2] = water
                img[:, :, 3] = item/sphere
                img[:, :, 4] = tool
            IMPORTANT: we keep these as separate channels, but because we assume we know
            the tool, the tool will have the full segmentation mask assuming no occlusions.
            The other classes will have masks that assume occlusions. ALSO, these use 0 and
            255 as values, because from inspecting SAC/CURL code, the images don't get
            normalized, so values passed to a network range in (0, 255).

        flow:
            Something new, here we'll also return the point cloud array with it.

        Update (08/17/2022): updating so that we now return depth images. Can use code
        in `softgym.utils.visualization.py` to visualize. Depth images are same size of
        cam_rgb but of type float32, and expressed in 'meters' in SoftGym units, so the
        values are typically bounded in [0, x] where x is around 1 or a little less.
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
            obs = np.concatenate([pos_tool_tip, pos_sphere, [rad_sphere], [water_frac]])
            return obs

        def get_state_info():
            # Just doing 10. We didn't actually collect data with this, we just computed
            # this afterwards, but we might want to regen data just to be safe.
            # Ball pos (3), ladle tip position (3), ladle quaternion (4)
            ball_pos = self._get_rigid_pos()[:3]   # 4D, but we only want 3D
            ladle_tip = self.tool_state_tip[0,:3]  # tool_state_tip is shape (1,14)
            ladle_quat = self.tool_state[0,6:10]   # I think this should work?
            state = np.concatenate((ball_pos, ladle_tip, ladle_quat))
            return state

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
            n1, n2 = len(pc_tool), len(pc_targ)
            n_pts = n1 + n2
            pc_array = np.zeros((max(n_pts, self.max_pts), self.pc_point_dim))
            pc_array[     :n1, :3] = pc_tool
            pc_array[     :n1,  3] = 1.
            pc_array[n1:n1+n2, :3] = pc_targ
            pc_array[n1:n1+n2,  4] = 1.

            # For simplicity, order idxs so all tool points are first, etc.
            # Should save the idxs we used to inform the tool flow subsampling.
            idxs = np.arange(n_pts)  # including if we don't need to subsample
            if n_pts > self.max_pts:
                idxs = np.sort( np.random.permutation(n_pts)[:self.max_pts] )
                pc_array = pc_array[idxs, :]
            self.segm.set_subsampling_tool_flow(idxs, n_tool=len(pc_tool))

            # Record amount of such points for debugging analysis later.
            self.pcl_dict['tool_raw'] = n1
            self.pcl_dict['targ_raw'] = n2
            self.pcl_dict['tool_subs'] = len(np.where(pc_array[:,3] == 1)[0])
            self.pcl_dict['targ_subs'] = len(np.where(pc_array[:,4] == 1)[0])
            return pc_array

        def get_pointcloud_array_gt(pts_sphere):
            # Another variant, now with ball points even if occluded.
            pc = self.segm.get_pointclouds()
            # If subsampling PCL _and_ we have tool flow, we should subsample
            # tool flow (so the tool pts coincide with the PC array's tool pts).
            pc_tool = pc['tool']

            # Sample sphere points, center at `targ_pos` and `dist_pos`.
            targ_pos = self._get_rigid_pos(item_index=0)[:3]
            targ_samp_pts = sample_sphere_points(pts_sphere, radius=self._sphere_radius_PCL)
            pc_targ = targ_pos + targ_samp_pts

            # Let's reserve the last `pts_sphere` points for the one sphere.
            n1, n2 = len(pc_tool), len(pc_targ)
            n_pts = n1 + n2
            pc_array = np.zeros((max(n_pts, self.max_pts), self.pc_point_dim))
            pc_array[     :n1, :3] = pc_tool
            pc_array[     :n1,  3] = 1.
            pc_array[n1:n1+n2, :3] = pc_targ
            pc_array[n1:n1+n2,  4] = 1.

            # For compatibility with tool flow, order so all tool points come first.
            # Should save the idxs we used to inform the tool flow subsampling.
            idxs = np.arange(n_pts)  # including if we don't need to subsample
            if n_pts > self.max_pts:
                # Only subsample the tool (NOT the ball). Offset both n_pts and
                # max_pts by total sphere points, then subsample just the tool.
                new_max_pts = self.max_pts - pts_sphere
                assert new_max_pts > 0
                idxs = np.sort( np.random.permutation(n_pts - pts_sphere)[:new_max_pts] )
                idxs = np.concatenate( [idxs, np.arange(n_pts - pts_sphere, n_pts)] )
                pc_array = pc_array[idxs, :]
            self.segm.set_subsampling_tool_flow(idxs, n_tool=len(pc_tool))

            assert len(np.where(pc_array[:,4] == 1)[0]) == pts_sphere
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
        elif self.observation_mode == 'depth_img':
            _, depth_img = get_segm_img(with_depth=True)
            w,h = depth_img.shape
            new_img = np.zeros([w,h,3])
            new_img[:,:,0] = depth_img
            new_img[:,:,1] = depth_img
            new_img[:,:,2] = depth_img
            return new_img
        elif self.observation_mode == 'depth_segm':
            # 08/24/2022: now depth_segm for another baseline. For this env, use 3
            # channels. First as depth, second and third are the two classes. But
            # make everything in [0,1], and note the nature of `segm_img`. CAREFUL,
            # another design choice is if we should override anything form the tool.
            # I think we should, FYI. Returns (H,W,3) np.float32 image.
            cam_rgb = self.get_image(self.camera_width, self.camera_height)
            segm_img, depth_img = get_segm_img(with_depth=True)
            mask_tool = segm_img[:,:,0]  # binary {0,255} image, see segm code
            mask_item = segm_img[:,:,4]  # binary {0,255} image, see segm code

            # Item should occlude the tool (makes it match the PCL).
            idxs_occlude = np.logical_and(mask_tool, mask_item)
            mask_tool[idxs_occlude] = 0

            # Concatenate and form the image. Must do the same during BC!
            mask_tool = mask_tool.astype(np.float32) / 255.0  # has only {0.0, 1.0}
            mask_item = mask_item.astype(np.float32) / 255.0  # has only {0.0, 1,0}
            depth_segm = np.concatenate(
                    (depth_img[...,None],
                     mask_tool[...,None],
                     mask_item[...,None]), axis=2)

            # Debugging.
            # import cv2, os
            #k = len([x for x in os.listdir('tmp') if 'depth_' in x and '.png' in x])
            #cv2.imwrite(f'tmp/rgb_{str(k).zfill(3)}.png', cam_rgb)
            #cv2.imwrite(f'tmp/depth_{str(k).zfill(3)}.png', (depth_segm[:,:,0] / np.max(depth_segm[:,:,0]) * 255).astype(np.uint8))
            #cv2.imwrite(f'tmp/mask1_{str(k).zfill(3)}.png', (depth_segm[:,:,1] * 255).astype(np.uint8))
            #cv2.imwrite(f'tmp/mask2_{str(k).zfill(3)}.png', (depth_segm[:,:,2] * 255).astype(np.uint8))
            return depth_segm
        elif self.observation_mode == 'rgb_segm_masks':
            # 08/25/2022: similarly now have RGB as first part, then segm.
            cam_rgb = self.get_image(self.camera_width, self.camera_height)
            segm_img, _ = get_segm_img(with_depth=True)
            mask_tool = segm_img[:,:,0]  # binary {0,255} image, see segm code
            mask_item = segm_img[:,:,4]  # binary {0,255} image, see segm code

            # Item should occlude the tool (makes it match the PCL).
            idxs_occlude = np.logical_and(mask_tool, mask_item)
            mask_tool[idxs_occlude] = 0

            # New here, divide image by 255.0 so we get values in [0,1] to align w/others.
            # I think this will make it easier as compared to keeping them on diff scales.
            cam_rgb = cam_rgb.astype(np.float32) / 255.0

            # Concatenate and form the image. Must do the same during BC!
            mask_tool = mask_tool.astype(np.float32) / 255.0  # has only {0.0, 1.0}
            mask_item = mask_item.astype(np.float32) / 255.0  # has only {0.0, 1,0}
            rgb_segm_masks = np.concatenate(
                    (cam_rgb,
                     mask_tool[...,None],
                     mask_item[...,None]), axis=2)
            return rgb_segm_masks
        elif self.observation_mode == 'rgbd_segm_masks':
            # 08/25/2022: similarly now have RGB as first part, then depth, then segm.
            cam_rgb = self.get_image(self.camera_width, self.camera_height)
            segm_img, depth_img = get_segm_img(with_depth=True)
            mask_tool = segm_img[:,:,0]  # binary {0,255} image, see segm code
            mask_item = segm_img[:,:,4]  # binary {0,255} image, see segm code

            # Item should occlude the tool (makes it match the PCL).
            idxs_occlude = np.logical_and(mask_tool, mask_item)
            mask_tool[idxs_occlude] = 0

            # New here, divide image by 255.0 so we get values in [0,1] to align w/others.
            # I think this will make it easier as compared to keeping them on diff scales.
            cam_rgb = cam_rgb.astype(np.float32) / 255.0

            # Concatenate and form the image. Must do the same during BC!
            mask_tool = mask_tool.astype(np.float32) / 255.0  # has only {0.0, 1.0}
            mask_item = mask_item.astype(np.float32) / 255.0  # has only {0.0, 1,0}
            rgbd_segm_masks = np.concatenate(
                    (cam_rgb,
                     depth_img[...,None],
                     mask_tool[...,None],
                     mask_item[...,None]), axis=2)
            return rgbd_segm_masks
        elif self.observation_mode == 'state':
            # New for the CoRL rebuttal, state-based policy baseline.
            return get_state_info()
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
            # UPDATE: adding state info.
            keypts = get_keypoints()
            img_rgb = self.get_image(self.camera_width, self.camera_height)
            segm_img, depth_img = get_segm_img(with_depth=True)
            tool_flow = get_tool_flow()
            pc_array = get_pointcloud_array()
            state = get_state_info()
            return (keypts, img_rgb, segm_img, pc_array, tool_flow, depth_img, state)
        elif self.observation_mode == 'combo_gt_v01':
            # (07/23) Adding based on the multi-sphere case.
            keypts = get_keypoints()
            img_rgb = self.get_image(self.camera_width, self.camera_height)
            segm_img, depth_img = get_segm_img(with_depth=True)
            tool_flow = get_tool_flow()
            pc_array_gt_v01 = get_pointcloud_array_gt(pts_sphere=250)
            return (keypts, img_rgb, segm_img, pc_array_gt_v01, tool_flow, depth_img)
        elif self.observation_mode == 'combo_gt_v02':
            # (07/23) Adding based on the multi-sphere case.
            keypts = get_keypoints()
            img_rgb = self.get_image(self.camera_width, self.camera_height)
            segm_img, depth_img = get_segm_img(with_depth=True)
            tool_flow = get_tool_flow()
            pc_array_gt_v02 = get_pointcloud_array_gt(pts_sphere=600)
            return (keypts, img_rgb, segm_img, pc_array_gt_v02, tool_flow, depth_img)
        else:
            raise NotImplementedError()

    def compute_reward(self, obs=None, action=None, set_prev_reward=False):
        """Reward function, both dense and sparse.

        There are a LOT of caveats / assumptions.
        (1) Assume it's enough for us to detect height cutoff, so that if the item falls
            off, it would do so much earlier.
        (2) Assume we can also compare height of item vs height of bowl.
        (3) For now, not going to penalize for water outside, though we could later.
        (4) Not using any of the inputs, but keeping them here for a consistent API.

        This is also repeated in `_get_info()`. Also we have a hack where we subtract
        the `item_radius` (edit: actually 1.5x that) from `height_item`.

        01/09/2022: Adding dense reward option from Eddie.
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

        # Also enforce that the item is within the glass bounds (except height).
        in_bounds_x, in_bounds_z = self._item_in_bounds(item_pos)
        sparse_reward = sparse_reward and in_bounds_x and in_bounds_z
        dense_reward = dense_reward if in_bounds_x and in_bounds_z else -1.

        # If all this is true, then we _still_ need time steps exceeded.
        if sparse_reward:
            self.time_exceeded += 1
        else:
            self.time_exceeded = 0
        sparse_reward = sparse_reward and (self.time_exceeded >= self.time_cutoff)

        if self.reward_type == 'dense':
            return dense_reward
        elif self.reward_type == 'sparse':
            return int(sparse_reward)
        else:
            raise NotImplementedError(self.reward_type)

    def _get_info(self):
        """Extra info, not used by the agent but used by us for evaluation.

        SoftAgent doesn't support nonscalar info, so for now don't return arrays.

        We are repeating the reward code here to populate stuff in info(). But we do
        NOT want to adjust the time exceeded here! Otherwise that's a double call.
        Also, if `done=True` here then we've met the sparse reward condition.

        NOTE(daniel) as of 11/19/2021, subtracting `item_radius` from `height_item`
        because I noticed cases when the item was stuck to the bottom of the tool, but
        its 'height' was presumed to be above the tool, and the reason is the tool's
        position is at a 'bottom corner' of the 3D voxelized grid where it appears.
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

        info = {
            'done': int(sparse_reward) == 1,
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
            info['pcl_tool_subs'] = self.pcl_dict['tool_subs']
            info['pcl_targ_subs'] = self.pcl_dict['targ_subs']
        return info


if __name__ == '__main__':
    env = MMOneSphereEnv(observation_mode='cam_rgb',
                         action_mode='translation',
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
