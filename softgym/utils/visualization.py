import os
import cv2
import numpy as np
from PIL import Image
from moviepy.editor import ImageSequenceClip
from os.path import join
from collections import defaultdict
import matplotlib.pyplot as plt


def save_pointclouds(obs_p, savedir=None, suffix='point_cloud_segm.gif',
        n_views=1, return_np_array=False, pour_water=False):
    """Make matplotlib visualization GIF of the point clouds.

    NOTE: this may be a major compute bottleneck.

    The positions of the point cloud should be w.r.t. world / base frame.
    We use essentially the same code in real with minor adjustments.

    CAUTION / NOTE! Annoyingly, we need to do extra work to align the axes.
    In sim, with the mixed media setup, the y-axis points up, NOT the z-axis.
    Furthermore, in order to get the point cloud to align with RGB images, we
    negate the x-axis in the xz plane. This only works because the xz plane
    is centered at (0,0) for the x and z axes. THOUGHT: should we adjust the
    point cloud data for PointNet++, just so we can 'interpret' it easily?

    Also tune the elevation and azimuth in the GIFs with `plt.show()`. Or
    we can always have multiple versions and stack them side by side, which
    might not be a bad idea.

    07/26/2022: should remove padded rows before subsampling PCLs.

    Parameters
    ----------
    obs_p: a list of point cloud observations for the full episode.
    savedir, suffix: save GIF at os.path.join(savedir, suffix).
    n_views: int, how many PC views? I'm supporting {1,2,3}.
    return_np_array: if True, then we do NOT save but instead return the
        numpy array of frames, for use later. So this method name is a bit
        misleading since we can use it in BC training to save PCLs at test
        time for extra debugging. Also in these cases we should resize the
        image frames, otherwise file sizes can blow up.
    """

    # Assumes that we have 2 or 3 classes. Based on our replay buffer code.
    def remove_zeros(obs):
        tool_idxs = np.where(obs[:,3] == 1)[0]
        targ_idxs = np.where(obs[:,4] == 1)[0]
        if obs.shape[1] == 6:
            dist_idxs = np.where(obs[:,5] == 1)[0]
        else:
            dist_idxs = np.array([])
        n_nonzero_pts = len(tool_idxs) + len(targ_idxs) + len(dist_idxs)

        # Clear out 0s in observation (if any) and in actions (if applicable).
        if n_nonzero_pts < obs.shape[0]:
            nonzero_idxs = np.concatenate(
                    (tool_idxs, targ_idxs, dist_idxs)).astype(np.uint64)
            obs = obs[nonzero_idxs]
        return obs

    # Bells and whistles.
    if not return_np_array:
        assert savedir is not None
    frames = []
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)

    # Multiple versions of the PC at different perspectives, stacked together.
    # Using elev=90 and azim=90 corresponds to the camera view in SoftGym.
    D = 6  # creates (600,600,3) frame
    if pour_water:
        azim = -10
    else:
        azim = 90

    # Set env-dependent ranges.
    if n_views == 1:
        if pour_water:
            view_params = [(45, azim, 111)]
        else:
            view_params = [(90, azim, 111)]
        figsize = (D, D)
    elif n_views == 2:
        view_params = [(10, azim, 121), (90, azim, 122)]
        figsize = (2*D, D)
    elif n_views == 3:
        view_params = [(10, azim, 131), (45, azim, 132), (90, azim, 133)]
        figsize = (3*D, D)
    else:
        raise ValueError(n_views)

    if pour_water:
        xlow, xhigh = -0.05, 0.75
        yhigh, ylow = -0.40, 0.40  # NOTE, reverse the ranges hack
        zlow, zhigh =  0.00, 0.80  # NOTE, this is the y-coord in SoftGym
    else:
        xlow, xhigh = -0.15, 0.15
        ylow, yhigh = -0.15, 0.15
        zlow, zhigh =  0.15, 0.65  # NOTE, this is the y-coord in SoftGym

    for pidx,pcl in enumerate(obs_p):
        fig = plt.figure(figsize=figsize, constrained_layout=True)

        # Put multiple (elev,azim) combos in each figure.
        for (elev, azim, rowcolindex) in view_params:
            ax = fig.add_subplot(rowcolindex, projection='3d')
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlim(xlow, xhigh)
            ax.set_ylim(ylow, yhigh)
            ax.set_zlim(zlow, zhigh)  # this is the y-coord in SoftGym

            # Before subsampling, let's actually remove the rows with 0s.
            pcl = remove_zeros(pcl)

            # Now subsample for visual clarity.
            orig_len = len(pcl)
            if orig_len >= 1500:
                choice = np.random.choice(len(pcl), size=1500, replace=False)
            else:
                choice = np.arange(len(pcl))
            pcl = pcl[choice]

            # Identify segmentation labels and indices. Careful if we change this!
            i_tool = np.where(pcl[:,3] > 0)[0]  # tool is same for both envs
            i_targ = np.where(pcl[:,4] > 0)[0]  # ball for MM, pouree box for PW
            if pcl.shape[1] == 6:
                i_dist = np.where(pcl[:,5] > 0)[0]  # dist for MM, water parts for PW

            # Scatter based on color. Note the y vs z swap (see comments above).
            if pour_water:
                ax.scatter(pcl[i_tool, 0], pcl[i_tool, 2], pcl[i_tool, 1], color='black')
                ax.scatter(pcl[i_targ, 0], pcl[i_targ, 2], pcl[i_targ, 1], color='yellow')
                if pcl.shape[1] == 6:
                    ax.scatter(pcl[i_dist, 0], pcl[i_dist, 2], pcl[i_dist, 1], color='red')
            else:
                ax.scatter(-pcl[i_tool, 0], pcl[i_tool, 2], pcl[i_tool, 1], color='black')
                ax.scatter(-pcl[i_targ, 0], pcl[i_targ, 2], pcl[i_targ, 1], color='yellow')
                if pcl.shape[1] == 6:
                    ax.scatter(-pcl[i_dist, 0], pcl[i_dist, 2], pcl[i_dist, 1], color='red')
            #title = f'Elev: {elev}, Azim: {azim}, t: {str(pidx).zfill(3)}'
            #title = f'{title}\n#tool points: {len(i_tool)}'
            title = f'Time: {str(pidx).zfill(3)}'
            ax.set_title(title, fontsize=23)

        # un-comment to debug elev, azim, etc.
        #plt.show()
        ax.set_axis_off()  # this will turn off axes

        # Get numpy array for figure
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        if return_np_array:
            assert len(frame.shape) == 3 and frame.shape[2] == 3, frame.shape
            frame = cv2.resize(frame, (256,256))
        frames.append(frame)
        plt.close(fig)

    # Use `return_np_array` in case we need to do further processing elsewhere.
    if return_np_array:
        return np.array(frames)
    save_path = os.path.join(savedir, suffix)
    save_numpy_as_gif(np.array(frames), save_path, fps=10)


def make_grid(array, nrow=1, padding=0, pad_value=120):
    """ numpy version of the make_grid function in torch. Dimension of array: NHWC """
    if len(array.shape) == 3:  # In case there is only one channel
        array = np.expand_dims(array, 3)
    N, H, W, C = array.shape
    assert N % nrow == 0
    ncol = N // nrow
    idx = 0
    grid_img = None
    for i in range(nrow):
        row = np.pad(array[idx], [[padding if i == 0 else 0, padding], [padding, padding], [0, 0]], constant_values=pad_value)
        for j in range(1, ncol):
            idx += 1
            cur_img = np.pad(array[idx], [[padding if i == 0 else 0, padding], [0, padding], [0, 0]], constant_values=pad_value)
            row = np.hstack([row, cur_img])
        if i == 0:
            grid_img = row
        else:
            grid_img = np.vstack([grid_img, row])
        idx += 1  # Daniel: needed to avoid repeating GIFs
    return grid_img


def save_numpy_as_gif(array, filename, fps=20, scale=1.0):
    """Creates a gif given a stack of images using moviepy
    Notes
    -----
    works with current Github version of moviepy (not the pip version)
    https://github.com/Zulko/moviepy/commit/d4c9c37bc88261d8ed8b5d9b7c317d13b2cdf62e
    Usage
    -----
    >>> X = randn(100, 64, 64)
    >>> gif('test.gif', X)
    Parameters
    ----------
    filename : string
        The filename of the gif to write to
    array : array_like
        A numpy array that contains a sequence of images
    fps : int
        frames per second (default: 10)
    scale : float
        how much to rescale each image by (default: 1.0)
    """

    # ensure that the file has the .gif extension
    fname, _ = os.path.splitext(filename)
    filename = fname + '.gif'

    # copy into the color dimension if the images are black and white
    if array.ndim == 3:
        array = array[..., np.newaxis] * np.ones(3)

    # make the moviepy clip
    clip = ImageSequenceClip(list(array), fps=fps).resize(scale)
    clip.write_gif(filename, fps=fps)
    return clip


def save_numpy_to_gif_matplotlib(array, filename, interval=50):
    from matplotlib import animation
    from matplotlib import pyplot as plt

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)

    def img_show(i):
        plt.imshow(array[i])
        print("showing image {}".format(i))
        return

    ani = animation.FuncAnimation(fig, img_show, len(array), interval=interval)

    ani.save('{}.mp4'.format(filename))

    import ffmpy
    ff = ffmpy.FFmpeg(
        inputs={"{}.mp4".format(filename): None},
        outputs={"{}.gif".format(filename): None})

    ff.run()


def _process_color(obs_color):
    """Process a raw color image from pyflex.render()."""
    #obs_color = cv2.cvtColor(obs_color, cv2.COLOR_RGB2BGR)
    return obs_color


def _process_rgb(obs_rgb):
    """Process a raw color image from pyflex.render_sensor()."""
    obs_rgb = (obs_rgb / np.max(obs_rgb) * 255).astype(np.uint8)
    #obs_rgb = cv2.cvtColor(obs_rgb, cv2.COLOR_RGB2BGR)
    return obs_rgb


def _process_depth(depth):
    """Process a raw depth image into something human-readable."""
    depth = (depth / np.max(depth) * 255).astype(np.uint8)
    depth_st = np.dstack([depth, depth, depth])
    depth_img = Image.fromarray(depth_st)
    depth_img = np.array(depth_img)
    return depth_img


def _process_segmentation(segm, colors=None):
    """Process a raw segmentation into something human-readable.

    NOTE: assumes segm has 0s, 1s, 2s, etc., up to the number of items we are
    trying to segment. So the normalization here will expand this to the range of
    [0,255] to hopefully increase readability. Actually with more things to segment,
    colors (i.e., number of colors) might be easier.
    """
    if colors is not None:
        assert np.max(segm) < colors, np.max(segm)
        segm = (segm / np.max(segm) * 255.0).astype(np.uint8)
    else:
        segm = (segm / np.max(segm) * 255.0).astype(np.uint8)
    segm_st = np.dstack([segm, segm, segm])
    segm_img = Image.fromarray(segm_st)
    segm_img = np.array(segm_img)
    return segm_img


def save_segmentations(obs_l, frames, savedir):
    """Custom method for testing segmentation for _one_ episode.

    This is meant to be run in conjunction with `examples/demonstrator.py`. The
    image sizes we save are specified with (H,W) here, but the actual resolution
    should be from the 'camera_width' and 'camera_height' parameters.

    :param obs_l: A list equal to the time horizon (plus one) where each item consists
        of a segmented representation, typically of shape (128, 128, n_segm) where the
        number of segmentation classes is `n_segm`. Also for now we only have values
        of 0 and 255 so they are already human readable.
    :param frames: Should be a similar-length list, this is the RGB stuff from the env.
        Actually this might be longer due to some frame skipping.
    :param savedir: Directory to save.
    """
    H, W = 320, 320
    assert len(frames)-1 == 2 * (len(obs_l)-1), f'{len(frames)} vs {len(obs_l)}'
    assert frames[0].shape[0] == obs_l[0].shape[0], \
        f'{frames[0].shape} vs {obs_l[0].shape}, please check `img_size`'

    # Each list consists of one of the segmentation channels we extract.
    segm_channels = defaultdict(list)
    all_segm_frames = []
    n_segm = obs_l[0].shape[2]
    print(f'Saving segmentations with {n_segm} channels!')

    def triplicate(img):
        assert len(img.shape) == 2, img.shape
        w,h = img.shape
        new_img = np.zeros([w,h,3])
        img = img.reshape([w,h])
        for i in range(3):
            new_img[:,:,i] = img
        return new_img

    for idx, obs in enumerate(obs_l):
        assert obs.shape[2] == n_segm, f'{obs.shape} vs {n_segm}'
        for c in range(n_segm):
            segm_channels[f'seg_{c}'].append(obs[:, :, c])
        # Might need to fix grayscale issue?
        #all_segm = np.array([obs[:, :, c] for c in range(n_segm)])
        #all_segm = np.array([make_grid(np.array(f), nrow=1, padding=3) for f in all_segm])
        # Let's add another channel to these.
        current_segms_l = [triplicate(obs[:,:,c]) for c in range(n_segm)]
        concat_segm = np.concatenate(current_segms_l, axis=1)
        all_segm_frames.append(concat_segm)

    all_segm_pth       = join(savedir, 'all_segm.gif')
    all_rgb_frames_pth = join(savedir, 'all_rgb_frames_segm.gif')

    # Make a `combo` which combines the prior segmentation channels.
    # frames.shape: (161, 128, 128, 3)
    # all_segm_frames.shape: (81, 128, 640, 3)
    all_segm_frames = np.array(all_segm_frames)
    frames = np.array(frames)  # rgb
    idxs = np.array([x for x in range(len(frames)) if x % 2 == 0])
    rgb_frames_sub = frames[idxs, :, :, :]  # subsample every 2
    print(f'Trying to concatenate: {rgb_frames_sub.shape} and {all_segm_frames.shape}')
    assert rgb_frames_sub.shape[0] == all_segm_frames.shape[0], \
        f'{rgb_frames_sub.shape} vs {all_segm_frames.shape}'
    rgb_frames_segm = np.concatenate([rgb_frames_sub, all_segm_frames], axis=2)
    save_numpy_as_gif(all_segm_frames, all_segm_pth,       fps=10)
    save_numpy_as_gif(rgb_frames_segm, all_rgb_frames_pth, fps=10)
    print(f'Done saving, check: {savedir}')


if __name__ == '__main__':
    # Older tests from Xingyu / SoftGym.
    if False:
        N = 12
        H = W = 50
        X = np.random.randint(0, 255, size=N * H * W* 3).reshape([N, H, W, 3])
        grid_img = make_grid(X, nrow=3, padding=5)
        cv2.imshow('name', grid_img / 255.)
        cv2.waitKey()

    # --------------------------------------------------------------------------- #
    # Newer visualization / segmentation tests from Daniel Seita et al.
    # --------------------------------------------------------------------------- #
    # Testing the segmentation. This can be for color-only. Overrides normal seg.
    # Might be easier to do this for non-particle images. However, couldn't we also
    # use the task knowledge: with water, we should know depth values, right? Well,
    # ideally ... also recall, OpenCV uses BGR format by default. The images should
    # be saved into BGR format, but when we load them in matplotlib, it's RGB.
    # --------------------------------------------------------------------------- #
    pth = './data_demo/MixedMediaRetrieval_scale_0.06_noise_0.0/0000_no_obs'
    pth = './data_demo/MixedMediaRetrieval_scale_0.06_noise_0.0/0001_YES_obs'

    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import cm
    from matplotlib import colors

    img_pths = sorted([join(pth,x) for x in os.listdir(pth) if '.png' in x])
    for idx,img_pth in enumerate(img_pths):
        img = cv2.imread(img_pth)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Visualize
        r, g, b = cv2.split(img)
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1, projection="3d")

        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        # TODO(daniel) processs / create segmented image
        seg = np.zeros((img.shape[0], img.shape[1], 3))
        seg_pth = img_pth.replace('_bgr_', '_seg_')
        cv2.imwrite(seg_pth, seg)
