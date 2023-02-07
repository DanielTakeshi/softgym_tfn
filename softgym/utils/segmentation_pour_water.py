"""
For Segmentation, pouring envs.
"""
import cv2
import os
import pyflex
import numpy as np
from pyquaternion import Quaternion
from softgym.utils.camera_projections import get_world_coords

class SegmentationPourWater:

    def __init__(self, use_fake_tool):
        """Sets up segmentation.

        Populate the class attributes in the `assign_other` and `assign_camera`
        methods after an env reset().

        fake tool is only if we need to duplicate the tool (and probably poured
        box) with a second camera, we would need to do this anyway to track which
        belongs to the tool vs poured box.
        """
        pyflex_root = os.environ['PYFLEXROOT']
        assert 'PyFlexRobotics' not in pyflex_root, pyflex_root
        assert 'PyFlex' in pyflex_root, pyflex_root
        self.use_fake_tool = use_fake_tool
        self._debug_print = False

        # Flow calculation.
        self.prev_tool_state = None
        self._sub_tool_idxs = None

        # Point clouds.
        self.box_targ_pts = None
        self.water_pts = None
        self.tool_points = None
        self.prev_tool_points = None

    def reset(self, off_x, glass_x, poured_x):
        """This is called when we reset.

        Since the target box stays fixed, let's record it here.
        BTW this is really the `assign_other` version of segmentation for MM.
        """
        self.offset_xval = off_x

        # Unlike before we can directly take an image and then get the pixels
        # corresponding to the targets.
        pyflex.render() # We do need pyflex.render().
        rgbd = pyflex.render_sensor(0)

        # Reshape and extract the depth. Nonzero means boxes, right?
        rgbd = np.array(rgbd).reshape(self.camera_height, self.camera_width, 4)
        rgbd = rgbd[::-1, :, :]
        rgb = rgbd[:, :, :3]
        depth = rgbd[:, :, 3]

        # Determine WORLD coordinates for EACH pixel, shaped into (height,width,3).
        world_coords = get_world_coords(rgb, depth, self.matrix_world_to_camera)
        world_coords = world_coords[:, :, :3]

        # Find the largest x for tool, smallest x for target. That gives us
        # the space between the two, and we can use that for a cutoff point.
        # NOTE: I think we have to intersect this with a depth image to tell
        # us which points are nonzero (i.e., boxes) and which are zero (boxes).
        boundary_x = (glass_x + poured_x) / 2.0
        if self._debug_print:
            print(f'Segm, {glass_x:0.3f} vs {poured_x:0.3f}, boundary {boundary_x:0.3f}')
        boxes_tool = world_coords[:,:,0] < boundary_x
        boxes_targ = world_coords[:,:,0] > boundary_x
        boxes_tool = np.logical_and(boxes_tool, depth > 0)
        boxes_targ = np.logical_and(boxes_targ, depth > 0)
        boxes_tool = (boxes_tool * 255).astype(np.uint8)
        boxes_targ = (boxes_targ * 255).astype(np.uint8)

        # Save for later, targ_box_values can be used for point cloud.
        self.targ_box_pix = boxes_targ > 0
        self.targ_box_pcl = world_coords[self.targ_box_pix]
        self.start_depth = depth
        self.start_world_coords = world_coords
        if self._debug_print:
            print(f'Targ box PCL: {self.targ_box_pcl.shape}')

        # Debugging, seems to be as expected though depth image values are
        # a bit strange in some cases (but identifying the right pixels...).
        # Both boxes. Seems like nonzero is what we want?
        if self._debug_print:
            boxes_both = ((depth > 0) * 255).astype(np.uint8)
            boxes_depth = process_depth(depth)
            cv2.imwrite(f'img_boxes_both_depth.png', boxes_depth)
            cv2.imwrite(f'img_boxes_both.png', boxes_both)
            cv2.imwrite(f'img_boxes_targ.png', boxes_targ)
            cv2.imwrite(f'img_boxes_took.png', boxes_tool)

        # Borrowed from MM. Remember, this gets called when env resets.
        self.prev_tool_state = None  # use to ignore flow at time=0
        self.box_targ_pts = None
        self.water_pts = None
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

    def query_images(self):
        """Query RGB(D) images, including with potential camera shifts.

        The purpose of `rgb2` and `depth2` is to get them from a different view,
        so that we can segment out a tool which is copying the original tool.

        NOTE(daniel): we don't actually use the RGB portion of `pyflex.render_sensor()`
        and it also seems a bit stranger / unusual somehow when we load it?

        NOTE(daniel): the PourWater depth data I made does not have water particles
        in it (but the RGBD images have water from the RGB part). But w/PourWater6D,
        the demo data Yufei made will have water particles. The code by default will
        use depth with water particles.

        NOTE(daniel): 08/21/2022 see discussion with Yufei on slack, reverting this
        change for the purposes of the rebuttal.
        """

        # We do need pyflex.render(). But new here, turn fluid rendering on/off.
        pyflex.draw_fluids(False)
        pyflex.render()
        rgbd = pyflex.render_sensor(0)
        pyflex.draw_fluids(True)

        # Reshape and extract the depth.
        rgbd = np.array(rgbd).reshape(self.camera_height, self.camera_width, 4)
        rgbd = rgbd[::-1, :, :]
        rgb = rgbd[:, :, :3]
        depth = rgbd[:, :, 3]

        # What happens if we do this? I think this will give a SEPARATE depth image
        # which we can then return with water. Then we can compare this and depth, and
        # use this water_depth to potentially override other things? I think this will
        # occlude the water target so we should have separate segmentation masks I think.
        pyflex.render()
        water_depth = pyflex.render_sensor(0)
        water_depth = np.array(water_depth).reshape(self.camera_height, self.camera_width, 4)[::-1, :, 3]

        # NOTE(daniel): we are NOT using this!!!
        if self.use_fake_tool:
            # Also, do the same thing but switch the camera location (x-coordinate)!
            camera_pos = pyflex.get_camera_pos()
            camera_pos[0] += self.offset_xval
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
            pyflex.set_camera_pos(camera_pos)

            images = {
                'rgb': rgb,
                'rgb2': rgb2,
                'depth': depth,
                'depth2': depth2,
            }
            self._debug(images)
        else:
            images = {
                'rgb': rgb,
                'depth': depth,
                'water_depth': water_depth,
            }
        return images

    def segment(self, images):
        """Daniel: segment the image. Should be simpler than with mixed media.

        As of 05/31/2022, not using the fake tool, just using the ability to
        turn rendering of water on/off for extracting the boxes.

        08/25/2022: now have water_depth.
        """
        rgb = images['rgb']  # only for height/width in `get_world_coords`.
        depth = images['depth']  # everything
        water_depth = images['water_depth']  # should be same as depth but water occludes.

        # Extract position of all the N particles, water only. Shape (N,3).
        # EDIT: actually I think we should just not segment this. Might be
        # too confusing and we just need these raw positions for the PCL.
        # Yeah, just save positions_all for later!
        positions_all = pyflex.get_positions().reshape([-1,4])[:,:3]

        # Determine WORLD coordinates for EACH pixel, shaped into (height,width,3).
        world_coords = get_world_coords(rgb, depth, self.matrix_world_to_camera)
        world_coords = world_coords[:, :, :3]

        # Segment the boxes, into the tool box vs target box.
        if self.use_fake_tool:
            # Get tool world coords, this (should) ignores water as it's the fake tool.
            depth2 = images['depth2']
            tool_world_coords = get_world_coords(rgb, depth2, self.matrix_world_to_camera)
            tool_world_coords = tool_world_coords[:, :, :3]

            # There might be some bugs due to shooting water pellets, if that's the
            # case then just ignore any points with outlier x values in the point cloud
            # We should do that anyway for water, BTW.
            tool_pix = depth2 > 0
            targ_pix = np.logical_and(np.copy(self.targ_box_pix), ~tool_pix)
        else:
            # We saved the target beforehand. But, target might be overridden.
            both_tools = depth > 0

            # These pixels are the tool, BUT only account for points that do
            # not override the target box (there might be more tool points).
            tool_pix = np.logical_and(both_tools, ~self.targ_box_pix)

            # NOTE(daniel) originally the water was occluding part of the target
            # box, but due to fluid rendering flag, we can turn that on/off. :)

            # Not vectorizing, make it correct first, then vectorize.
            targ_pix = np.copy(self.targ_box_pix)
            xs, ys = np.where(self.targ_box_pix)
            for (xx,yy) in zip(xs,ys):
                curr_xy_depth = depth[xx,yy]
                start_xy_depth = self.start_depth[xx,yy]
                if curr_xy_depth + 1e-6 < start_xy_depth:
                    tool_pix[xx, yy] = True
                    targ_pix[xx, yy] = False

        # ------------------------------------------------------------------- #
        # ---------- Begin segmented image! 0 = 'outside/nothing'. ---------- #
        # ------------------------------------------------------------------- #

        # Try to see if we can get approx. water pixels. I think we can do this by
        # comparing like this because water_depth must have strictly more info:
        water_inds = np.where(water_depth + 1e-7 < depth)
        # If water particles are present they can only decrease depth, otherwise they
        # will just be what was there earlier (the boxes).

        img_segm = np.zeros((self.camera_width, self.camera_height)).astype(np.uint8)
        img_segm[ tool_pix > 0 ] = 1
        img_segm[ targ_pix > 0 ] = 2

        # Actually return a _binary_ segmented image, but to be consistent with RGB,
        # (1) put values in (0,255) and (2) return a uint8.
        segmented = np.zeros((self.camera_width,
                              self.camera_height,
                              4)).astype(np.float32)
        segmented[:, :, 0] = img_segm == 0
        segmented[:, :, 1] = img_segm == 1  # TOOL (CUP WE CONTROL)
        segmented[:, :, 2] = img_segm == 2  # TARG (CUP IS FIXED)

        # AH! Let's do this HERE so we avoid overriding? That way we don't occlude.
        # It will occlude if we save `segm` though.
        img_segm[  water_inds  ] = 3

        segmented[:, :, 3] = img_segm == 3  # WATER
        segmented = (segmented * 255.0).astype(np.uint8)

        if self._debug_print:
            n = len([x for x in os.listdir('.') if 'segm.png' in x])
            _seg = (img_segm * (255./3)).astype(np.uint8)
            _rgb = ((rgb / np.max(rgb)) * 255).astype(np.uint8)
            cv2.imwrite(f'{str(n).zfill(3)}_segm.png', img=_seg)
            cv2.imwrite(f'{str(n).zfill(3)}_real.png', img=_rgb)

        # Compute pointclouds for later. Only tools, we did water earlier.
        self.prev_tool_points = self.tool_points  # for flow, don't need np.copy
        self.tool_points = world_coords[ tool_pix > 0 ]  # the _current_ tool!
        self.box_targ_pts = world_coords[ targ_pix > 0 ]
        self.water_pts = positions_all

        return segmented

    def get_pointclouds(self):
        """Must be called after segment().

        Returns all such points, not subsampled (that happens later).
        """
        assert self.tool_points is not None, "Must call segment first!"
        return {
            'box_tool': self.tool_points,
            'box_targ': self.box_targ_pts,
            'water': self.water_pts,
        }

    def get_tool_flow(self, tool_state):
        """Returns tool flow (i.e., for the box we control).

        CAVEAT: This function will return the *previous* tool flow, as
        the observation only contains the previous tool flow information.
        If no previous tool flow exists (when env is reset), return None.

        Note: must assign to prev {tool,tip} state!

        Unlike with MM, I don't think we need a `tip_state`. Here the
        tool_state can be the bottom of the box (right?). Here we can
        work with the (_,14)-shaped array corresponding to box walls?

        IMPORTANT! 'rotation_bottom' uses index 0 for the shape (the wall)
        which acts as the center of rotation. If changing, then adjust the
        index into `tool_state`.
        """
        assert self.tool_points is not None, "Must call segment first!"

        if self.prev_tool_state is None:
            self.prev_tool_state = tool_state
            return None

        # Get transformation and rotation deltas
        delta_p = tool_state[0, :3] - self.prev_tool_state[0, :3]

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

        # Unlike MM, here the center is actually the tool_state[0, :3].
        relative = self.prev_tool_points - self.prev_tool_state[0, :3]

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

    def _debug(self, images):
        """Save a bunch of images"""
        curr_imgs = sorted(
            [x for x in os.listdir('.') if x[-4:]=='.png' and 'fake' in x]
        )
        n_imgs = str(len(curr_imgs)).zfill(3)
        depth1 = process_depth(images['depth'])
        depth2 = process_depth(images['depth2'])
        cv2.imwrite(f'img_real_{n_imgs}.png', depth1)
        cv2.imwrite(f'img_fake_{n_imgs}.png', depth2)


def process_depth(orig_img, cutoff=None):
    """Make a raw depth image human-readable by putting values in [0,255).

    Careful if the cutoff is in meters or millimeters! I think meters.
    """
    img = orig_img.copy()

    # Useful to turn the background into black into the depth images.
    def depth_to_3ch(img, cutoff):
        w,h = img.shape
        new_img = np.zeros([w,h,3])
        img = img.flatten()
        if cutoff is not None:
            img[img>cutoff] = 0.0
        img = img.reshape([w,h])
        for i in range(3):
            new_img[:,:,i] = img
        return new_img

    def depth_scaled_to_255(img):
        if np.max(img) <= 0.0:
            print('Warning, np.max: {:0.3f}'.format(np.max(img)))
        img = 255.0/np.max(img)*img
        img = np.array(img,dtype=np.uint8)
        for i in range(3):
            img[:,:,i] = cv2.equalizeHist(img[:,:,i])
        return img

    img = depth_to_3ch(img, cutoff)  # all values above 255 turned to white
    img = depth_scaled_to_255(img)   # correct scaling to be in [0,255) now
    return img


if __name__ == "__main__":
    pass
