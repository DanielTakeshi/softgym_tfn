"""
Meant to be used for when we have saved data from SoftAgent for rollouts from
loading a flow-based policy, and we want to stitch everything together.

Don't forget to divide pcl and flow by 250 since I'm saving raw values.

Update 11/06/2022: I think we have a good video, though I have to make the
sizeref=25 for PourWater. I wonder if this is necessary in general?
"""
import os
import os.path as osp
import sys
import argparse
import pickle
import cv2
import io
from PIL import Image
import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go
import plotly.subplots as make_subplots
import multiprocessing as mp
from collections import defaultdict
from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from softgym.utils.visualization import (
    save_numpy_as_gif, save_segmentations, save_pointclouds
)


def save_flow(obs_f, savedir, pour_water=False):
    """Generates a GIF of flow."""
    if savedir[-4:] != '.gif':
        savedir = savedir+'.gif'
    if pour_water:
        with mp.Pool(mp.cpu_count()) as pool:
            frames = pool.map(_create_flow_frame_pw, obs_f)
        save_numpy_as_gif(np.array(frames), savedir, fps=10)
    else:
        with mp.Pool(mp.cpu_count()) as pool:
            frames = pool.map(_create_flow_frame, obs_f)
        save_numpy_as_gif(np.array(frames), savedir, fps=10)
    return np.array(frames)


def pointcloud(
    T_chart_points: np.ndarray, downsample=5, colors=None, scene="scene",
    name=None, pour_water=False,
) -> go.Scatter3d:
    """Generate point clouds.

    This will have the original source points for the flow. Empirically, seems
    like we might want marker size greater than 3 for figures?

    Also I realized now from debugging pour water that for visualization, we flip
    the z (which turns to y).
    """
    if pour_water:
        marker_dict = {"size": 3}
    else:
        marker_dict = {"size": 3}
    if colors is not None:
        try:
            a = [f"rgb({r}, {g}, {b})" for r, g, b in colors][::downsample]
            marker_dict["color"] = a
        except:
            marker_dict["color"] = colors[::downsample]

    # Annoying, flip x for scooping, flip z for pouring?
    xx = 1 if pour_water else -1
    zz = -1 if pour_water else 1

    return go.Scatter3d(
        x=T_chart_points[0, ::downsample] * xx,
        y=T_chart_points[2, ::downsample] * zz,  # this turns into y
        z=T_chart_points[1, ::downsample],
        mode="markers",
        marker=marker_dict,
        scene=scene,
        name=name,
        showlegend=False,
    )


def _flow_traces_v2(
    pos, flows, sizeref=0.10, scene="scene", flowcolor="red", name="flow",
    pour_water=False,
):
    x_lines = list()
    y_lines = list()
    z_lines = list()

    # normalize flows:
    nonzero_flows = (flows == 0.0).all(axis=-1)
    n_pos = pos[~nonzero_flows]
    n_flows = flows[~nonzero_flows]

    n_dest = n_pos + n_flows * sizeref

    # Annoying, flip x for scooping, flip z for pouring?
    xx = 1 if pour_water else -1
    zz = -1 if pour_water else 1

    for i in range(len(n_pos)):
        x_lines.append(n_pos[i][0] * xx)
        y_lines.append(n_pos[i][2] * zz)
        z_lines.append(n_pos[i][1])
        x_lines.append(n_dest[i][0] * xx)
        y_lines.append(n_dest[i][2] * zz)
        z_lines.append(n_dest[i][1])
        x_lines.append(None)
        y_lines.append(None)
        z_lines.append(None)

    # Decreasing `width` means decreasing thickness of flow lines (I like it).
    lines_trace = go.Scatter3d(
        x=x_lines,
        y=y_lines,
        z=z_lines,
        mode="lines",
        scene=scene,
        line=dict(color=flowcolor, width=8),
        name=name,
        showlegend=False,
    )

    head_trace = go.Scatter3d(
        x=n_dest[:, 0] * xx,
        y=n_dest[:, 2] * zz,
        z=n_dest[:, 1],
        mode="markers",
        marker={"size": 2, "color": "darkred"},
        scene=scene,
        showlegend=False,
    )

    return [lines_trace, head_trace]


def _3d_scene(data):
    # Create a 3D scene which is a cube w/ equal aspect ratio and fits all the data.

    assert data.shape[1] == 3
    # Find the ranges for visualizing.
    mean = data.mean(axis=0)
    max_x = np.abs(data[:, 0] - mean[0]).max()
    max_y = np.abs(data[:, 1] - mean[1]).max()
    max_z = np.abs(data[:, 2] - mean[2]).max()
    all_max = max(max(max_x, max_y), max_z)
    scene = dict(
        xaxis=dict(nticks=10, range=[mean[0] - all_max, mean[0] + all_max]),
        yaxis=dict(nticks=10, range=[mean[1] - all_max, mean[1] + all_max]),
        zaxis=dict(nticks=10, range=[mean[2] - all_max, mean[2] + all_max]),
        aspectratio=dict(x=1, y=1, z=1),
    )
    return scene


def _3d_scene_fixed(x_range, y_range, z_range):
    scene = dict(
        xaxis=dict(nticks=10, range=x_range),
        yaxis=dict(nticks=10, range=y_range),
        zaxis=dict(nticks=10, range=z_range),
        aspectratio=dict(x=1, y=1, z=1),
    )
    return scene


def _create_flow_frame(sample, sizeref=16):
    """For mixed media.

    Update 08/23/2022: some tweaks to allow for making videos for project website
    with aligned RGB images and segmented point clouds. Actually it would probably
    have been better to use z=0.75 or something but then I'd have to redo the demonstrator.
    But maybe try x,y,z all at 1.10 to get a slightly closer-view?
    The sizeref we'll use 15 for now? (Edit: maybe 16)
    """
    pts = sample['points']
    flow = sample['flow']

    # Easiest way to debug: the 'eye' is the camera position.
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.10, y=1.10, z=1.10)
    )

    # Shrink layout, otherwise we get a lot of whitespace.
    layout = go.Layout(
        autosize=True,
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
        width=600,
        height=600,
    )

    # Increase `sizeref` to increase length of red flow vectors for visualization.
    f = go.Figure(layout=layout)
    f.add_trace(
        pointcloud(pts.T, downsample=1, scene="scene1")
    )
    # Update 11/21: downsampling these for the Twitter video.
    ts = _flow_traces_v2(pts[::4], flow[::4], sizeref=sizeref, scene="scene1")
    for t in ts:
        f.add_trace(t)
    f.update_layout(
        # NOTE (08/23) tweaking these for flow visualizations for website.
        scene1=_3d_scene_fixed([-0.40, 0.15], [-0.40, 0.15], [-0.05, 0.60]),
        scene_camera=camera,
    )

    fig_bytes = f.to_image(format="png", engine="kaleido")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)


def _create_flow_frame_pw(sample, sizeref=25):
    """For pour water.

    If making a plot for a paper, should make the x and y coordinate ranges
    be the same to avoid stretching the box. Also, sometimes we might have to
    zoom into images which means likely tuning the downsampling ratio and the
    marker sizes, etc.

    If downsampling we might sometimes want to keep the blue points and just
    downsample flow, for that downsample the input to `_flow_traces_v2()`.

    Update 08/23/2022: don't forget we should flip the y values ... as shown in
    my SoftAgent debugging. Using 600x600 to align them with other GIFs.
    """

    # This camera seems reasonably close to what we see in real.
    # Edit: this is for a 'sideways' view for Pouring. But I want more of an angled view!
    #camera = dict(
    #    up=dict(x=0, y=0, z=1),
    #    center=dict(x=0, y=0, z=0),
    #    eye=dict(x=0.10, y=-2.00, z=0.10)
    #)
    # Easiest way to debug: the 'eye' is the camera position.
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.55, y=-1.55, z=0.75)
    )

    # Actually I sometimes downsample _here_ instead of in the methods.
    #downsample = 3
    pts = sample['points']
    flow = sample['flow']
    #pts = pts[::downsample]
    #flow = flow[::downsample]

    # Shrink layout, otherwise we get a lot of whitespace.
    layout = go.Layout(
        autosize=True,
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0},
        width=600,
        height=600,
    )

    # Increase `sizeref` to increase length of red flow vectors for visualization.
    f = go.Figure(layout=layout)
    f.add_trace(
        pointcloud(pts.T, downsample=1, scene="scene1", pour_water=True)
    )
    ts = _flow_traces_v2(pts[::2], flow[::2], sizeref=sizeref, scene="scene1", pour_water=True)
    for t in ts:
        f.add_trace(t)

    # Amusingly now we don't need to flip the y-axis ranges, I think this is due to
    # setting the eye (see above) in a clever way.
    f.update_layout(
        scene1=_3d_scene_fixed([-0.10, 0.60], [-0.35, 0.35], [0.05, 0.75]),
        scene_camera=camera,
    )
    fig_bytes = f.to_image(format="png", engine="kaleido")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)


def save_and_combine_stuff(rgb_imgs_l, raw_stuff_l, ep, pw=True, dir_tail=''):
    """Create video."""
    rgb_save_pth = osp.join('rollouts', f'ep_{str(ep).zfill(2)}_rgb')
    save_numpy_as_gif(np.array(rgb_imgs_l), rgb_save_pth)

    # We need to downscale. Then later we can upscale flow with `sizeref`.
    scale = 250.

    # Note: point clouds are on the raw scale, we have to downscale later.
    suffix = f'ep_{str(ep).zfill(2)}_pcl'
    pcl_save_pth = osp.join('rollouts', suffix)
    obs_p = [raw_stuff['obs'] for raw_stuff in raw_stuff_l]
    for i in range(len(obs_p)):
        obs_p[i][:,:3] /= scale
    save_pointclouds(
        obs_p=obs_p,
        savedir='rollouts',
        pour_water=pw,
        suffix=suffix,
    )

    # Flow visualizations? Also yes we have to downscale as well.
    flow_save_pth = osp.join('rollouts', f'ep_{str(ep).zfill(2)}_flow')
    xyz_l = [raw_stuff['xyz'] for raw_stuff in raw_stuff_l]
    flow_l = [raw_stuff['flow'] for raw_stuff in raw_stuff_l]
    obs_f = []
    for xyz,flow in zip(xyz_l, flow_l):
        sample = {
            'points': xyz / scale,
            'flow': flow / scale,
        }
        obs_f.append(sample)
    save_flow(obs_f=obs_f, savedir=flow_save_pth, pour_water=pw)

    # Used for loading a GIF and extracting the individual frames.
    def loadGIF(filename):
        gif = cv2.VideoCapture(filename)
        frames = []
        while True:
            ret, cv2im = gif.read()
            if not ret:
                break
            cv2im = cv2.cvtColor(cv2im, cv2.COLOR_RGB2BGR)
            frames.append(cv2im)
        return frames

    # Now with the 3 data directories we made, we should combine the GIFs.
    combo_save_pth = osp.join('rollouts', f'ep_{str(ep).zfill(2)}_ALL')
    im_rgb = loadGIF(rgb_save_pth+'.gif')
    im_pcl = loadGIF(pcl_save_pth+'.gif')
    im_flow = loadGIF(flow_save_pth+'.gif')
    im_rgb = im_rgb[:-1]  # dump last frame
    print(f'  loaded GIFs, len {len(im_rgb)}, {len(im_pcl)}, {len(im_flow)}')
    all_stuff = []
    len_i = min([len(im_rgb), len(im_pcl), len(im_flow)])
    for i in range(len_i):
        im_rgb_i = im_rgb[i]
        im_pcl_i = im_pcl[i]
        im_flow_i = im_flow[i]
        if im_rgb_i.shape[0] != im_pcl_i.shape[0]:
            im_rgb_i = cv2.resize(im_rgb_i, (im_pcl_i.shape[0], im_pcl_i.shape[1]))
        concat_frame = np.hstack( (im_rgb_i,im_pcl_i,im_flow_i) )
        all_stuff.append(concat_frame)
    save_numpy_as_gif(np.array(all_stuff), combo_save_pth)

    # Finally, save a _video_ in .mp4 format so we can share it on the website. The
    # fps is 15 which is reasonable, but since it's .mp4, the user can pause it.
    succeed = dir_tail.split('_')[-1]
    vid_flow_save_pth = f'{combo_save_pth}_succeed_{succeed}.mp4'
    fps = 15
    frame_size = all_stuff[0].shape
    out = cv2.VideoWriter(
            vid_flow_save_pth,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (frame_size[1], frame_size[0])
    )
    for frame in all_stuff:
        frame_cvt = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_cvt)
    out.release()
    print(f'  video: {vid_flow_save_pth}')


def get_rgb_image_seq(data_dir, ep):
    """Extract RGB images from videos."""
    img_dir = [x for x in os.listdir(data_dir) if f'ep_{ep}_cam_' in x]
    assert len(img_dir) == 1, img_dir
    rgb_img_dir = osp.join(data_dir, img_dir[0])
    rgb_imgs_l = sorted(
        [osp.join(rgb_img_dir,x) for x in os.listdir(rgb_img_dir) if '.png' in x]
    )
    print(f'  checking {len(rgb_imgs_l)} RGB images: {rgb_img_dir}')
    return rgb_imgs_l, rgb_img_dir


def get_obs_flow_seq(data_dir, ep):
    """Extract segmented point clouds from videos."""
    raw_data_dir = [x for x in os.listdir(data_dir) if f'ep_{ep}_raw_data_' in x]
    assert len(raw_data_dir) == 1, raw_data_dir
    raw_stuff_dir = osp.join(data_dir, raw_data_dir[0])
    raw_stuff_l = sorted(
        [osp.join(raw_stuff_dir,x) for x in os.listdir(raw_stuff_dir) if '.pkl' in x]
    )
    raw_stuff_pkls = []
    for pkl in raw_stuff_l:
        with open(pkl, 'rb') as fh:
            # keys: 'obs', 'xyz', 'flow'
            data = pickle.load(fh)
        raw_stuff_pkls.append(data)
    return raw_stuff_pkls


def create_video(data_dir, pw=False, num_videos=25):
    """Iterates through all test-time runs, by default 25 from our BC04 settings."""
    for i in range(num_videos):
        print(f'\nOn episode {i}, pour water {pw} ...')
        rgb_imgs_l, rgb_img_dir = get_rgb_image_seq(data_dir, ep=i)
        raw_stuff_l = get_obs_flow_seq(data_dir, ep=i)
        _, dir_tail = osp.split(rgb_img_dir)
        save_and_combine_stuff(rgb_imgs_l, raw_stuff_l, ep=i, pw=pw, dir_tail=dir_tail)


if __name__ == '__main__':
    """Let's stitch things together."""
    p = argparse.ArgumentParser(description='Process some integers.')
    p.add_argument('--env_name', type=str, default='MMOneSphere')
    args = p.parse_args()
    env_kwargs = env_arg_dict[args.env_name]

    # Where we save the videos, etc. Currently we'll be overriding for simplicity.
    if not osp.exists('rollouts'):
        os.makedirs('rollouts', exist_ok=True)

    if args.env_name in ['MMOneSphere']:
        # ScoopBall 6D
        data_dir = osp.join(
            '../softagent_rpad_MM/data/local/',
            'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_load_model_debug',
            'BC04_MMOneSphere_v02_ntrain_0025_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_load_model_debug_2022_11_06_11_59_36_0001',
            'video'
        )
        assert osp.exists(data_dir), data_dir
        assert 'MMOneSphere' in data_dir, data_dir
        create_video(data_dir, pw=False)

    elif args.env_name in ['PourWater', 'PourWater6D']:
        # PourWater 6D
        data_dir = osp.join(
            '../softagent_rpad_MM/data/local/',
            'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_load_model_debug',
            'BC04_PourWater6D_v01_ntrain_0100_PCL_PNet2_svd_pointwise_ee2flow_6DoF_ar_8_hor_100_scalePCL_noScaleTarg_load_model_debug_2022_11_06_12_18_29_0001',
            'video'
        )
        assert osp.exists(data_dir), data_dir
        assert 'PourWater' in data_dir, data_dir
        create_video(data_dir, pw=True)
