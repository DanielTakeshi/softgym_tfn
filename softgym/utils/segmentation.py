"""
For Segmentation, MixedMedia envs.
"""
import os
import scipy
import pyflex
import numpy as np
from pyquaternion import Quaternion
from softgym.utils.camera_projections import get_world_coords

class Segmentation:

    def __init__(self, n_segm_classes, n_targs, n_distr, use_fake_tool=True):
        """Sets up segmentation.

        Mostly populate the class attributes in the `assign_other` and `assign_camera`
        methods after an env reset().
        """
        pyflex_root = os.environ['PYFLEXROOT']
        assert 'PyFlexRobotics' in pyflex_root, pyflex_root
        self.n_segm_classes = n_segm_classes
        assert n_segm_classes in [5, 6], n_segm_classes
        self.n_targs = n_targs
        self.n_distr = n_distr
        self.item_idx_targs = np.arange(self.n_targs)
        self.item_idx_distr = np.arange(self.n_distr) + self.n_targs
        self.use_fake_tool = use_fake_tool

        # Don't change!
        self.tool_idx = 0
        self.tool_idx_fake = 1

        # Some of these values should be tuned. Keep EFFICIENT_DIST=True.
        self.EFFICIENT_DIST = True
        self.PERCENTILE = 66
        self.DIST_PARTICLE = 0.015

        # Necessary for flow calculation
        self.prev_tool_state = None
        self.prev_tip_state = None
        self._sub_tool_idxs = None

        # And pointclouds
        self.target_points = None
        self.distractor_points = None
        self.tool_points = None
        self.prev_tool_points = None

    def assign_other(self, rigid_idxs, fluid_idxs, water_height_init, offset_xval,
            offset_zval, particle_to_item):
        """Note key assumptions about rigid and fluid idxs we need for segmentation!

        KEY ASSUMPTION: we have `self.rigid_boundaries` which tells us the min and max
        RIGID ITEM PARTICLE INDICES for each item. We should have all the target items
        be first, and then the distractor items be at the end.
        """
        self.rigid_idxs = rigid_idxs
        self.fluid_idxs = fluid_idxs
        self.particle_to_item = particle_to_item

        # For handling indices later with segmenting target vs distractor items.
        # Well, we actually only use `self.min_pidx_distr` ...
        self.min_pidx_targs = np.inf
        self.max_pidx_targs = 0
        self.min_pidx_distr = np.inf
        self.max_pidx_distr = 0
        for k in range(self.n_targs + self.n_distr):
            pyflex_indices = np.where(self.particle_to_item == k)[0]
            minv = np.min(pyflex_indices)
            maxv = np.max(pyflex_indices)
            if k < self.n_targs:
                self.min_pidx_targs = min(minv, self.min_pidx_targs)
                self.max_pidx_targs = max(maxv, self.max_pidx_targs)
            else:
                self.min_pidx_distr = min(minv, self.min_pidx_distr)
                self.max_pidx_distr = max(maxv, self.max_pidx_distr)

        self.fluid_start = np.min(self.fluid_idxs)
        self.fluid_end = np.max(self.fluid_idxs)
        assert self.max_pidx_distr < self.fluid_start
        self.water_height_init = water_height_init
        self.offset_xval = offset_xval
        self.offset_zval = offset_zval

        # This is called when we reset, so reset the tool and tip state
        self.prev_tool_state = None
        self.prev_tip_state = None
        self.target_points = None
        self.distractor_points = None
        self.tool_points = None
        self.prev_tool_points = None

    def assign_camera(self, camera_params, camera_name, matrix_world_to_camera):
        """Should be called by the environment during a reset()."""
        self.camera_params = camera_params
        self.camera_width = self.camera_params[camera_name]['width']
        self.camera_height = self.camera_params[camera_name]['height']
        assert self.camera_width == self.camera_height
        self.camera_name = camera_name
        self.matrix_world_to_camera = matrix_world_to_camera

    def get_tool_vec_offset(self):
        """Special method for determining the tool offset from position to tip.

        The reason is the tool 'position' is not at the tool's tip, but we want to do the
        tool rotations from its tip. For now we use the real tool because `get_world_coords`
        is w.r.t. the normal camera matrix, and what's key is that we need to assume the
        ladle tip is actually the highest point (lowest depth). That should be safe to assume
        as I doubt we'll ever _start_ with the ladle tip below something (this is only done
        during initialization).

        NOTE: another key assumption is that we assume the ladle tip is visible...

        NOTE: depth will be 0 if there's nothing at a point, so we need to take the min of
        all _nonzero_depth points to find the ladle tip.
        """
        pyflex.render()
        rgbd = pyflex.render_sensor(0)
        rgbd = np.array(rgbd).reshape(self.camera_height, self.camera_width, 4)
        rgbd = rgbd[::-1, :, :]
        rgb = rgbd[:, :, :3]
        depth = rgbd[:, :, 3]

        # Compute tool position from pyflex.
        tool_state = pyflex.get_shape_states().reshape((-1,14))
        tool_pos = tool_state[self.tool_idx, :3]

        # Compute fake tool's world position of the ladle tip.
        world_coords = get_world_coords(rgb, depth, self.matrix_world_to_camera)
        world_coords = world_coords[:, :, :3]
        depth[np.where(depth == 0)] = 1.0  # turn 0s into 1s so we can take the min
        tx, tz = np.where(depth == np.min(depth))
        tool_tip = world_coords[tx[0], tz[0]]

        # Finally, the vector offset. (Edit: actually return all this info...)
        vec_offset = tool_tip - tool_pos
        stuff = dict(vec_offset=vec_offset, tool_tip=tool_tip, tool_pos=tool_pos)
        return stuff

    def query_images(self):
        """Query RGB(D) images, including with potential camera shifts.

        The purpose of `rgb2` and `depth2` is to get them from a different view,
        so that we can segment out a tool which is copying the original tool.

        NOTE(daniel): we don't actually use the RGB portion of `pyflex.render_sensor()`
        and it also seems a bit stranger / unusual somehow when we load it?
        """
        # We do need pyflex.render().
        pyflex.render()
        rgbd = pyflex.render_sensor(0)

        # Reshape and extract the depth.
        rgbd = np.array(rgbd).reshape(self.camera_height, self.camera_width, 4)
        rgbd = rgbd[::-1, :, :]
        rgb = rgbd[:, :, :3]
        depth = rgbd[:, :, 3]

        # Also, do the same thing but switch the camera location (x-coordinate)!
        camera_pos = pyflex.get_camera_pos()
        camera_pos[0] += self.offset_xval
        camera_pos[2] += self.offset_zval
        pyflex.set_camera_pos(camera_pos)

        # Query the image from another location, and yes we need `pyflex.render()`.
        pyflex.render()
        rgbd2 = pyflex.render_sensor(0)
        rgbd2 = np.array(rgbd2).reshape(self.camera_height, self.camera_width, 4)
        rgbd2 = rgbd2[::-1, :, :]
        rgb2 = rgbd2[:, :, :3]
        depth2 = rgbd2[:, :, 3]

        # Reset camera back to normal.
        camera_pos[0] -= self.offset_xval
        camera_pos[2] -= self.offset_zval
        pyflex.set_camera_pos(camera_pos)

        images = {
            'rgb': rgb,
            'rgb2': rgb2,
            'depth': depth,
            'depth2': depth2,
        }
        return images

    def segment(self, images):
        """Daniel: segment the image. Seems OK but also involves a lot of assumptions:

        (0) Assume we segment outside, glass (i.e., box), water, tool, item(s), and for
        the tool we optionally assume we can segment everything as we control it, even
        for pixels where the tool is actually occluded.

        (1) We have a second (fake) tool that will copy what the normal tool is doing,
        then we segment that with a (fake) second depth camera at the same angle, etc.

        (2) Assumes we know the indexing and assignment of particles to items or fluid.
        We need this to map from (pixel and nearest particle) --> which particle index.

        (3) The segmentation is done at the particle-level, so if relying on the fluid
        viewer, the particles may actually be occupying some pixels that look like they
        are something else (e.g. fluid tends to accumulate more in the scooped up tool
        than suggested from the fluid viewer). Also, the results will differ slightly
        depending on the viewer, I think because the depth camera readings differ.

        (4) Assumes top-down (perspective) camera.

        (5) Water segmentation: while we measure the initial water height, and could assume
        that depth values at or below are considered water, this will NOT reflect water
        that's in the tool when it raises, the water also technically rises when the tool
        enters (and pushes water away), and measuring water particles should capture more
        fine-grained wave effects that might be helpful. If we want to fix a set of pixels
        and assume water 'fills in' we could try something like:

            depth_water = height_camera - (self.water_height_init + 0.02)  # hacky 0.02
            depth_water_idx = depth > depth_water

        which could be faster but also less realistic.

        (6) The order we apply segmentation matters. Generally best to apply item(s)
        segmentation at the end so they can override stuff. For the tool, we might add
        it earlier in segmentation and err on the side of overriding it, because we have
        the separate image which has all the info about the tool. If we overlay the tool
        with others with occlusions (which I don't think we should do) then the glass
        can't be above the tool, but water or items can be.

        (7) If we have multiple items and only want to get part of them, we detect if
        a pixel is in the 'target particle indices' or 'distractor particle indices'.
        """
        rgb = images['rgb']  # only for height/width in `get_world_coords`.
        depth = images['depth']  # everything
        depth2 = images['depth2']  # the (fake) tool only, for full tool segmentation

        # Extract height of the top-down camera, assume we can ignore perspective effects.
        # We might change this manually during deployment, hence should re-query here.
        height_camera = self.camera_params[self.camera_name]['pos'][1]

        # Assume anything above the initial water height is glass OR something we will
        # override after, such as the item or tool. The 0.05 is hacky, to increase glass.
        depth_water = height_camera - (self.water_height_init - 0.05)
        glass_pix = np.logical_and(depth > 0, depth < depth_water)

        # Extract position of all the N particles, items and water. Shape (N,3).
        position_all = pyflex.get_positions().reshape([-1, 4])[:, :3]

        # Determine WORLD coordinates for EACH pixel, shaped into (height,width,3).
        world_coords = get_world_coords(rgb, depth, self.matrix_world_to_camera)
        world_coords = world_coords[:, :, :3]

        # Get tool world coords to ignore water points
        tool_world_coords = get_world_coords(rgb, depth2, self.matrix_world_to_camera)
        tool_world_coords = tool_world_coords[:, :, :3]

        # Compute pairwise distances between pixels' world coords and particles.
        if self.EFFICIENT_DIST:
            # Extract percentile rank of fluid heights (y-coord, idx=1).
            position_fluid = (position_all[self.fluid_idxs])[:, 1]
            fluid_min_height = np.percentile(position_fluid, self.PERCENTILE)

            # Get pyflex indices of the fluid paticles we will consider in `cdist`.
            fluid_above = position_fluid > fluid_min_height
            fluid_above_idx = np.where(fluid_above)[0] + self.fluid_start

            # Here are all the valid PyFlex particle positions to check, filtering out
            # positions of water particles that have y values too low to consider. Note
            # that `position_valid_idx` helps us go from the "cdist index" which is
            # contiguous from 0 to len(position_valid)-1, to the "particle index".
            position_valid_idx = np.concatenate((self.rigid_idxs, fluid_above_idx))
            position_valid = position_all[position_valid_idx]

            # Then do the distances with only a subset of positions (positions_valid)!
            world_coords_l = world_coords.reshape((self.camera_height*self.camera_width, 3))
            # distances = scipy.spatial.distance.cdist(world_coords_l, position_valid)
            kd_tree = scipy.spatial.cKDTree(position_valid)
            dist_world_particle, estim_particle_idx = kd_tree.query(world_coords_l, n_jobs=-1)

            # Map from index used in "cdist" to the original PyFlex index, vectorized.
            estim_particle_idx = position_valid_idx[estim_particle_idx]
            dist_world_particle = dist_world_particle.reshape((self.camera_height, self.camera_width))
        else:
            # Slow because we don't need to check all water particles or all pixels.
            world_coords_l = world_coords.reshape((self.camera_height*self.camera_width, 3))
            distances = scipy.spatial.distance.cdist(world_coords_l, position_all)

            # For each pixel, find closest particle INDEX, shape (height*width,). Should have
            # lots of repetition since different pixels can have the same closest particle.
            estim_particle_idx = np.argmin(distances, axis=1)  # (num_pix, num_parti) matrix

            # For each pixel, extract world coords of its nearest particle (item or fluid).
            nearest_particle_pos = position_all[estim_particle_idx]
            nearest_particle_pos = nearest_particle_pos.reshape(
                (self.camera_height,self.camera_width,3))

            # For each pixel, check distance from its true world coordinate to position of
            # its nearest particle. Pixels can be far from box, or be for the occluding tool,
            # so technically they can have their nearest particle be far away. Use axis=2.
            dist_world_particle = np.linalg.norm(world_coords - nearest_particle_pos, axis=2)

        # If the distance between true world coord vs nearest particle is below some
        # threhsold. This gives us all the pixels that SHOULD be segmented to a particle
        # and helps us ignore the 'outside', 'glass', and 'tool' classes.
        pixels_particles = dist_world_particle < self.DIST_PARTICLE

        # Reshape to map from particle idx to the actual item index for segmentation.
        estim_particle_idx = estim_particle_idx.reshape((self.camera_height,self.camera_width))

        # Determine water indices, assumes that all water indices are AFTER item indices.
        water_pix = np.logical_and(pixels_particles, estim_particle_idx >= self.fluid_start)

        # Segment out the item indices. These contain ALL of the item pixels.
        item_pix = np.logical_and(pixels_particles, estim_particle_idx < self.fluid_start)

        # The target and distractor items. For now we can get away with just the
        # self.min_pidx_distr, if all target indices come BEFORE all distractor indices.
        # Also that value is set to default at np.inf if we have no distractors.
        item_targs_pix = np.logical_and(item_pix, estim_particle_idx < self.min_pidx_distr)
        item_distr_pix = np.logical_and(item_pix, estim_particle_idx >= self.min_pidx_distr)
        if self.n_segm_classes == 5:
            assert np.sum(item_distr_pix) == 0, np.sum(item_distr_pix)

        # Segment the ENTIRE tool, no occlusion.  We save `tool_segm` separately, so
        # do not modify it. It's an upper bound on the number of tool pixels.
        tool_segm = np.logical_and(depth2 > 0, depth2 < height_camera)
        tool_segm = tool_segm.astype(np.uint8)
        tool_pix = tool_segm > 0

        # ------------------------------------------------------------------- #
        # ---------- Begin segmented image! 0 = 'outside/nothing'. ---------- #
        # ------------------------------------------------------------------- #
        img_segm = np.zeros((self.camera_width, self.camera_height)).astype(np.uint8)

        # Segment the glass.
        img_segm[ glass_pix ] = 1

        # Segment the tool, potentially overriding the glass (which can't be above it).
        # Careful, this number is only for indexing if we want an overlaid image, but
        # we usually want these in separate channels. Check that this value isn't used.
        img_segm[ tool_pix ] = -1

        # Segment the water, potentially overriding parts of the glass or the tool.
        img_segm[ water_pix ] = 2

        # Segment the item, potentially overiding any prior stuff.
        img_segm[ item_targs_pix ] = 3
        if self.n_segm_classes == 6:
            img_segm[ item_distr_pix ] = 4

        # Actually return a _binary_ segmented image, but to be consistent with RGB,
        # (1) put values in (0,255) and (2) return a uint8.
        segmented = np.zeros((self.camera_width,
                              self.camera_height,
                              self.n_segm_classes)).astype(np.float32)
        segmented[:, :, 0] = tool_pix       # ALL tool pixels, INCLUDING occluded ones
        segmented[:, :, 1] = img_segm == 0  # outside
        segmented[:, :, 2] = img_segm == 1  # glass
        segmented[:, :, 3] = img_segm == 2  # water
        segmented[:, :, 4] = img_segm == 3  # target item
        if self.n_segm_classes == 6:
            segmented[:, :, 5] = img_segm == 4  # distractor (if any)
        segmented = (segmented * 255.0).astype(np.uint8)

        # Compute pointclouds for later. Here, we now need to consider items
        # occluding tool, otherwise item / dist points might be labeled as tool.
        # The item / dist already have occlusions taken into account.
        self.target_points = world_coords[item_targs_pix]
        self.distractor_points = world_coords[item_distr_pix]
        tool_and_item = np.logical_and(tool_pix, item_targs_pix)
        tool_and_dist = np.logical_and(tool_pix, item_distr_pix)
        tool_and_others = np.logical_or(tool_and_item, tool_and_dist)
        tool_pix_filtered = np.logical_and(~tool_and_others, tool_pix)
        self.prev_tool_points = self.tool_points  # for flow

        tool_coords = tool_world_coords if self.use_fake_tool else world_coords
        self.tool_points = tool_coords[tool_pix_filtered]

        return segmented

    def get_pointclouds(self):
        """MUST BE CALLED AFTER segment!

        This will return ALL such points (not subsampled). Subsampling happens
        during `_get_obs()` of MM envs.
        """
        assert self.target_points is not None, "Must call segment first!"
        return {
            'target': self.target_points,
            'distractor': self.distractor_points,
            'tool': self.tool_points,
        }

    def get_tool_flow(self, tool_state, tip_state):
        """Returns tool flow (i.e., for the ladle we control).

        CAVEAT: This function will return the *previous* tool flow, as
        the observation only contains the previous tool flow information.
        If no previous tool flow exists (when env is reset), return None.

        Note: must assign to prev {tool,tip} state!

        tool_state: (2,14) shape of tool and fake tool.
        tip_state: (1,14) shape of the tip (the tiny box we added).
        """
        assert self.tool_points is not None, "Must call segment first!"

        if self.prev_tool_state is None:
            self.prev_tool_state = tool_state
            self.prev_tip_state = tip_state
            return None

        # Get transformation and rotation deltas
        delta_p = tip_state[0, :3] - self.prev_tip_state[0, :3]

        old_q = self.prev_tool_state[0, 6:10]
        new_q = tool_state[0, 6:10]
        old_quat = Quaternion(w=old_q[3], x=old_q[0], y=old_q[1], z=old_q[2])
        new_quat = Quaternion(w=new_q[3], x=new_q[0], y=new_q[1], z=new_q[2])
        delta_quat = new_quat * old_quat.inverse

        # Compute the tool translation
        n_pts = self.prev_tool_points.shape[0]
        tool_flow = np.zeros_like(self.prev_tool_points)
        tool_flow += delta_p

        # Compute the tool rotation. PyQuaternion rotations are not
        # immediately vectorizable but see the code afterwards for the
        # vectorized version. Requires computing vector from rotation
        # center to the point on the PCL, then doing rotation on that.
        # for i in range(n_pts):
        #     pt = self.prev_tool_points[i]
        #     relative = pt - self.prev_tip_state[0, :3]
        #     relative_rot = delta_quat.rotate(relative)
        #     tool_flow[i] += relative_rot - relative

        delta_quat._normalise()
        dqp = delta_quat.conjugate.q

        relative = self.prev_tool_points - self.prev_tip_state[0, :3]

        vec_mat = np.zeros((n_pts, 4, 4), dtype=self.prev_tool_points.dtype)
        vec_mat[:, 0, 1] = -relative[:, 0]
        vec_mat[:, 0, 2] = -relative[:, 1]
        vec_mat[:, 0, 3] = -relative[:, 2]

        vec_mat[:, 1, 0] = relative[:, 0]
        vec_mat[:, 1, 2] = -relative[:, 2]
        vec_mat[:, 1, 3] = relative[:, 1]

        vec_mat[:, 2, 0] = relative[:, 1]
        vec_mat[:, 2, 1] = relative[:, 2]
        vec_mat[:, 2, 3] = -relative[:, 0]

        vec_mat[:, 3, 0] = relative[:, 2]
        vec_mat[:, 3, 1] = -relative[:, 1]
        vec_mat[:, 3, 2] = relative[:, 0]

        mid = np.matmul(vec_mat, dqp)
        mid = np.expand_dims(mid, axis=-1)

        relative_rot = delta_quat._q_matrix() @ mid
        relative_rot = relative_rot[:, 1:, 0]

        tool_flow += relative_rot - relative

        # Set previous states
        self.prev_tool_state = tool_state
        self.prev_tip_state = tip_state

        # Subsample points based on prior subsampling in MM envs, then return.
        prev_tool_sub = self.prev_tool_points[self._sub_tool_idxs]
        tool_flow_sub = tool_flow[self._sub_tool_idxs]
        return {
            'points': prev_tool_sub,  # (num_tool_pts, 3)
            'flow': tool_flow_sub,  # (num_tool_pts, 3)
        }

    def set_subsampling_tool_flow(self, idxs, n_tool):
        """If we want to subsample the tool flow.

        Possibly easier to handle that here instead of doing it afterwards.
        Should also use `idxs` do this even if we plan to keep everything.
        Should be in order, idxs[i] < idxs[i+1], and anything after n_tool,
        we can assume is an item or distractor.

        Annoying, must be called before we query point clouds elsewhere,
        see comments in MM envs `_get_obs()` methods.
        """
        tool_idxs = np.array([x for x in idxs if x < n_tool])
        self._sub_tool_idxs = tool_idxs


if __name__ == "__main__":
    pass