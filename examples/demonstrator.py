import os
import os.path as osp
import sys
import argparse
import pickle
import io
from PIL import Image
import numpy as np
import plotly.colors as pc
import plotly.graph_objects as go
import plotly.subplots as make_subplots
import multiprocessing as mp
from collections import defaultdict
from tqdm import tqdm
from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from softgym.utils.visualization import (
    save_numpy_as_gif, save_segmentations, save_pointclouds
)
import random


def save_gif(args, save_subdir, frames, ep_rew, ep_count, obs_l=None,
        info=None, obs_p=None, obs_f=None, obs_p_gt_v01=None, obs_depth=None):
    """Will only save a GIF if using continuous video. Assumes sparse rew.

    Used for `run_mm()` and `run_spheres()` but not `run_pw()`.

    Saves `frames` and `obs_l`, should ideally not override existing data.
    If we are generating data for BC, we don't need to do everything here.

    Parameters
    ----------
    obs_l: list of RGB or segm image obs
    obs_p: list of point cloud obs.
    obs_f: list of flow obs.
    obs_p_gt_v01: list of point cloud obs (ground truth variant).
    """
    save_tail = f'{str(ep_count).zfill(4)}.gif'

    # Try to add more to `save_tail` to make inspecting GIFs easier.
    if info is not None:
        if info['sunk_at_start']:
            save_tail = save_tail.replace('.gif', '_sunkStart.gif')
        if info['height_item'] < 0.1:
            save_tail = save_tail.replace('.gif', '_sunkEnd.gif')
        if info['stuck_item']:
            save_tail = save_tail.replace('.gif', '_stuck.gif')
        if (not info['in_bounds_x']) or (not info['in_bounds_z']):
            save_tail = save_tail.replace('.gif', '_oob.gif')
        if info['done']:
            save_tail = save_tail.replace('.gif', '_sparseYES.gif')
        else:
            save_tail = save_tail.replace('.gif', '_sparseNO.gif')

    # Save the GIF, and exit (if only saving data for BC). If action repeat
    # changed from 8 -> 1, should subsample the frames, though I also set the
    # env length to be 600 instead of 800 (as we weren't using all steps). If
    # we want to let BC go longer, can do that at test time.
    save_tail = save_tail.replace('.gif', f'_rew_{ep_rew:0.2f}.gif')
    save_name = osp.join(save_subdir, save_tail)
    factor = int(len(frames) / 100)
    factor = max(factor, 1)  # due to early termination in spheres envs
    frames_saved = [frames[ff] for ff in range(len(frames)) if ff%factor == 0]

    # Wait, let's now concatenate depth here. Also I dont think we subsample
    # since the `frames` here had more images right? Remember these are the
    # processed images, for an actual deep net we might just pass raw floats?
    # Careful: depth will be in [0,1] so don't divide by 255 as we do for RGB.
    #if obs_depth is not None:
    #    from softgym.utils.visualization import _process_depth
    #    depth_images = [_process_depth(dimg) for dimg in obs_depth]
    #    assert len(depth_images) == len(frames_saved), len(depth_images)
    #    rgb_depth_together = [np.hstack((cimg,dimg))
    #        for (cimg,dimg) in zip(frames_saved,depth_images)]
    #    frames_saved = rgb_depth_together  # override

    save_numpy_as_gif(np.array(frames_saved), save_name)
    print(f'Frames: {len(frames_saved)}, GIF saved to: {save_name}')
    if args.save_data_bc:
        return

    # Now also save observation list for segmentation.
    if obs_l is not None or obs_p is not None or obs_f is not None:
        save_obs_dir = save_name.replace('.gif', '_obs')
        if not os.path.exists(save_obs_dir):
            os.mkdir(save_obs_dir)

    #if obs_l is not None:
    #    save_segmentations(obs_l, frames, savedir=save_obs_dir)

    if obs_p is not None:
        save_pointclouds(obs_p, save_obs_dir, suffix='PCL_segm.gif')

    if obs_p_gt_v01 is not None:
        save_pointclouds(obs_p_gt_v01, save_obs_dir, suffix='PCL_segm_GT_balls.gif')

    if obs_f is not None:
        # Let's try and save some compute time.
        if len(obs_f) > 200:
            obs_f = [obs_f[idx] for idx in range(len(obs_f)) if idx % 6 == 0]
        save_flow(obs_f, savedir=save_obs_dir)


def save_pw_gif(args, save_subdir, frames, ep_rew, ep_count, info=None,
        obs_c=None, obs_s=None, obs_p=None, obs_f=None, obs_depth=None):
    """Will only save a GIF if using continuous video. Assumes sparse rew.

    Saves `frames` and `obs_l`, should ideally not override existing data.
    If we are generating data for BC, we don't need to do everything here.

    Parameters
    ----------
    obs_c: list of RGB images.
    obs_s: list of segmented images.
    obs_p: list of segmented point clouds.
    obs_f: list of tool flow obs.
    """
    save_tail = f'{str(ep_count).zfill(4)}.gif'

    # Try to add more to `save_tail` to make inspecting GIFs easier.
    if info is not None:
        if info['done']:
            save_tail = save_tail.replace('.gif', '_sparseYES.gif')
        else:
            save_tail = save_tail.replace('.gif', '_sparseNO.gif')
        water_in = info['performance']  # frac of water particles inside
        save_tail = save_tail.replace('.gif', f'_{water_in:0.3f}_.gif')

    # Save the GIF, and exit (if only saving data for BC). If action repeat
    # changed from 8 -> 1, should subsample the frames, though I also set the
    # env length to be 600 instead of 800 (as we weren't using all steps). If
    # we want to let BC go longer, can do that at test time.
    save_tail = save_tail.replace('.gif', f'_rew_{ep_rew:0.2f}.gif')
    save_name = osp.join(save_subdir, save_tail)
    factor = int(len(frames) / 100)
    frames_saved = [frames[ff] for ff in range(len(frames)) if ff%factor == 0]

    # Wait, let's now concatenate depth here. Also I dont think we subsample
    # since the `frames` here had more images right? Remember these are the
    # processed images, for an actual deep net we might just pass raw floats?
    # Careful: depth will be in [0,1] so don't divide by 255 as we do for RGB.
    #if obs_depth is not None:
    #    from softgym.utils.visualization import _process_depth
    #    depth_images = [_process_depth(dimg) for dimg in obs_depth]
    #    assert len(depth_images) == len(frames_saved), len(depth_images)
    #    rgb_depth_together = [np.hstack((cimg,dimg))
    #        for (cimg,dimg) in zip(frames_saved,depth_images)]
    #    frames_saved = rgb_depth_together  # override

    save_numpy_as_gif(np.array(frames_saved), save_name)
    print(f'Frames: {len(frames_saved)}, GIF saved to: {save_name}')
    if args.save_data_bc:
        return

    # Now also save observation list for segmentation.
    if obs_s is not None or obs_p is not None or obs_f is not None:
        save_obs_dir = save_name.replace('.gif', '_obs')
        if not os.path.exists(save_obs_dir):
            os.mkdir(save_obs_dir)

    if obs_p is not None and (not args.collect_visuals):
        pcl_frames = save_pointclouds(obs_p, savedir=save_obs_dir, pour_water=True)

    if obs_f is not None:
        # Let's try and save some compute time.
        if len(obs_f) > 200:
            obs_f = [obs_f[idx] for idx in range(len(obs_f)) if idx % 6 == 0]
        if args.collect_visuals:
            # Save as separate set of html files (easier for figures).
            save_flow_html(obs_f, obs_p=obs_p, savedir=save_obs_dir, pour_water=True)
        else:
            # Save as single numpy array (for GIFs).
            frames_flow = save_flow(obs_f, savedir=save_obs_dir, pour_water=True)


def run_spheres(args, env):
    """Take steps through spheres env, possibly save BC data."""

    # Assign the algorithmic policy choice (if desired), OK if None (it will ignore).
    assert args.env_name in ['SpheresLadle']
    assert hasattr(env, 'spheres_env_version') and hasattr(env, 'alg_policy')
    env.set_alg_policy(args.alg_policy)

    # Create directory for storing GIFs, etc.
    if not os.path.exists(args.save_video_dir):
        os.mkdir(args.save_video_dir)
    save_subdir = get_video_subdir(args, env)

    # Record info for reporting later. Detect sunk items at the start and end of episodes.
    ep_success = 0
    ep_rewards = []
    info_stats = {'stuck_item': 0, 'oob': 0, 'sunk_item_start': 0, 'sunk_item_end': 0}
    n_success_bc = 1100
    config_idxs = []

    # Save both `frames` (for continuous video) and `obs` (for testing segmentation).
    for e in range(args.num_variations):
        data_bc = defaultdict(list)
        this_ep_success = False

        # Each call to the reset will sample a specific config ID.
        obs = env.reset(config_id=e)
        SUNK_HEIGHT = env.get_sunk_height_cutoff()
        frames = [env.get_image(args.img_size, args.img_size)]
        obs_l = [obs]
        ep_rew = 0
        ep_start_sunk = False
        ep_start_sunk_d = 0
        print()
        print(100*'=')
        print(f'----- Episode {e+1} -----')
        print(100*'=')

        for t in range(env.horizon):
            # New change, moving this to MM env's method, see documentation there.
            action_raw, action_scaled = env.get_random_or_alg_action()

            # Add time-aligned BC data, separate from `obs_l` but with same data.
            if args.save_data_bc:
                data_bc['obs'].append(obs)
                data_bc['act_raw'].append(action_raw)
                data_bc['act_scaled'].append(action_scaled)

            # THEN take steps in the env as usual.
            obs, reward, done, info = env.step(action_scaled,
                    record_continuous_video=args.record_continuous_video,
                    img_size=args.img_size)

            # Record info.
            if args.record_continuous_video:
                frames.extend(info['flex_env_recorded_frames'])
            ep_rew += reward
            obs_l.append(obs)

            # Special to mixed media, we are strangely getting some sunk items.
            if t == 0:
                if info['height_item'] < SUNK_HEIGHT:
                    ep_start_sunk = True
                _k = 0
                while f'height_item_{_k}' in info.keys():
                    if info[f'height_item_{_k}'] < SUNK_HEIGHT:
                        ep_start_sunk_d += 1
                    _k += 1

            # New, early termination.
            if done and env.terminate_early:
                print(f'terminating at t={t}')
                break

        # For MM, `info['done']` checks the sparse reward condition. It is only True
        # if that condition has been met; we check it here at the END of the episode.
        if info['done']:
            ep_success += 1
            this_ep_success = True
        ep_rewards.append(ep_rew)

        # Record extra stuff from end of episode for detailed analysis.
        info['in_bounds_x'] = True  # compatibility with GIFs
        info['in_bounds_z'] = True  # compatibility with GIFs
        info['sunk_at_start'] = False
        info_stats[f'distractors_sunk_{str(e).zfill(4)}'] = ep_start_sunk_d
        if ep_start_sunk:
            info_stats['sunk_item_start'] += 1
            info['sunk_at_start'] = True  # Add to normal `info`.
        if info['height_item'] < 0.1:
            # The item should normally be starting at ~0.2, so if it's like this,
            # something strange happened with the initial data generation.
            info_stats['sunk_item_end'] += 1
        if info['stuck_item']:
            info_stats['stuck_item'] += 1

        # Save GIFs if we want to test segmentation, but we should also ensure the
        # recorded frames (img_size) is the same size as the camera width.
        print(f'Finished {e+1} episodes')
        print(f'  ep success? {ep_success} (cumulative), {this_ep_success} (this one)')
        print(f'  ep reward: {ep_rew:0.2f} (just this episode)')
        if args.obs_mode == 'cam_rgb':
            save_gif(args, save_subdir, frames, ep_rew=ep_rew, ep_count=e, obs_l=None, info=info)
        elif args.obs_mode in ['point_cloud', 'point_cloud_gt_v01', 'point_cloud_gt_v02']:
            save_gif(args, save_subdir, frames, ep_rew=ep_rew, ep_count=e, obs_p=obs_l, info=info)
        elif (args.obs_mode == 'segm') and (args.camera_width == args.img_size):
            save_gif(args, save_subdir, frames, ep_rew=ep_rew, ep_count=e, obs_l=obs_l, info=info)
        elif args.obs_mode == 'flow':
            obs_f = [o[1] for o in obs_l[1:]]
            save_gif(args, save_subdir, frames, ep_rew=ep_rew, ep_count=e, obs_l=None, info=info,
                obs_p=None, obs_f=obs_f)
        elif args.obs_mode in ['combo', 'combo_gt_v01', 'combo_gt_v02']:
            # I think this case is due to something with segmentation.
            if args.camera_width == args.img_size:
                # Save everything, mainly for debugging or to compare observations. First
                # two in each obs_l tuple are keypts (not used) and RGB (save separately).
                obs_s = [o[2] for o in obs_l]  # segmented, use obs_s instead of obs_l
                obs_p = [o[3] for o in obs_l]  # pc_array
                obs_f = [o[4] for o in obs_l[1:]]  # flow (tool)
                if len(obs_l[0]) > 5:
                    obs_p_gt_v01 = [o[5] for o in obs_l]  # pc_array (ground truth variant)
                else:
                    obs_p_gt_v01 = None
                save_gif(args, save_subdir, frames, ep_rew=ep_rew, ep_count=e, info=info,
                    obs_l=obs_s, obs_p=obs_p, obs_f=obs_f, obs_p_gt_v01=obs_p_gt_v01)
            else:
                pass

        # Save BC data for this episode, if filtering only do successful ones. We do
        # save GIFs of failures, mainly helps to check that we correctly ignore. :)
        if args.save_data_bc:
            if args.filtered:
                record_data = this_ep_success
                if record_data:
                    config_idxs.append(e)
            else:
                record_data = True

            if record_data:
                n_eps = f'{str(e).zfill(4)}'
                n_obs = len(data_bc['obs'])
                data_path = osp.join(save_subdir, f'BC_{n_eps}_{n_obs}.pkl')
                with open(data_path, 'wb') as fh:
                    pickle.dump(data_bc, fh)

                if args.filtered and (len(config_idxs) == n_success_bc):
                    print(f'Exiting due to len(config_idxs): {len(config_idxs)}')
                    last_ep_bc = e
                    break

    # Collect statistics on results and save to a text file.
    print(f'\n----- Finished {args.num_variations} variations! -----')
    results_txt = osp.join(save_subdir, 'results.txt')
    stuck = info_stats['stuck_item']
    sunk_start = info_stats['sunk_item_start']
    sunk_end = info_stats['sunk_item_end']
    oob = info_stats['oob']
    with open(results_txt, 'w') as tf:
        print(f'Success:    {ep_success} / {args.num_variations}', file=tf)
        if args.save_data_bc:
            print(f'Though we only did {last_ep_bc} configs due to BC', file=tf)
        mean = np.mean(ep_rewards)
        std = np.std(ep_rewards)
        median = np.median(ep_rewards)
        print(f'Ep Rews:  {mean:0.2f} +/- {std:0.1f}', file=tf)
        print(f'  median: {median:0.2f}', file=tf)
        print(f'Stuck items?     {stuck} / {args.num_variations}', file=tf)
        print(f'OOB item?          {oob} / {args.num_variations}', file=tf)
        print(f'Sunk start? {sunk_start} / {args.num_variations}', file=tf)
        print(f'Sunk end?     {sunk_end} / {args.num_variations}', file=tf)
        _e = 0
        # Only print the non-zero ones to save space.
        while f'distractors_sunk_{str(_e).zfill(4)}' in info_stats.keys():
            dsunk = info_stats[f'distractors_sunk_{str(_e).zfill(4)}']
            if dsunk > 0:
                print(f'sunk_distractors_start {str(_e).zfill(4)}: {dsunk}', file=tf)
            _e += 1
    print(f'See results in: {results_txt}')

    # If filtering BC data, probably should save the successful config indices.
    # When doing BC later, we load data from this file to filter the configs.
    if args.save_data_bc and args.filtered:
        BC_results_txt = osp.join(save_subdir, 'BC_data.txt')
        with open(BC_results_txt, 'w') as tf:
            for e in config_idxs:
                print(e, file=tf)
        print(f'Also see BC successful configs in: {BC_results_txt}')


def run_mm(args, env):
    """Take steps through mixed media env, possibly save BC data."""

    # Assign the algorithmic policy choice (if desired), OK if None (it will ignore).
    assert args.env_name in ['MMOneSphere', 'MMMultiSphere']
    assert hasattr(env, 'mm_env_version') and hasattr(env, 'alg_policy')
    env.set_alg_policy(args.alg_policy)

    # Create directory for storing GIFs, etc.
    if not os.path.exists(args.save_video_dir):
        os.mkdir(args.save_video_dir)
    save_subdir = get_video_subdir(args, env)

    # Record info for reporting later. Detect sunk items at the start and end of episodes.
    ep_success = 0
    ep_rewards = []
    info_stats = {'stuck_item': 0, 'oob': 0, 'sunk_item_start': 0,
        'sunk_item_end': 0, 'num_done_but_few_pcl_targ': 0}

    # For behavioral cloning, we want 1000 data points and to record successes.
    # EDIT: maybe not, might as well just get as many as we can? For BC#03 we might
    # just get 500 episodes, with up to 400 for training? Can use 2000 stored configs.
    # With act repeat=1 we get a lot more info per episode.
    if args.action_repeat == 1:
        n_success_bc = 500
    elif args.action_repeat == 8:
        n_success_bc = 1100
    config_idxs = []

    # Save both `frames` (for continuous video) and `obs` (for testing segmentation).
    for e in range(args.num_variations):
        data_bc = defaultdict(list)
        this_ep_success = False

        # Each call to the reset will sample a specific config ID.
        obs = env.reset(config_id=e)
        SUNK_HEIGHT = env.get_sunk_height_cutoff()
        frames = [env.get_image(args.img_size, args.img_size)]
        obs_l = [obs]
        ep_rew = 0
        ep_start_sunk = False
        ep_start_sunk_d = 0
        print()
        print(100*'=')
        print(f'----- Episode {e+1} -----')
        print(100*'=')

        for t in range(env.horizon):
            # New change, moving this to MM env's method, see documentation there.
            action_raw, action_scaled = env.get_random_or_alg_action()

            # Add time-aligned BC data, separate from `obs_l` but with same data.
            if args.save_data_bc:
                data_bc['obs'].append(obs)
                # eddieli 08/16/2022, adding this to track global rotation
                data_bc['tool_state'].append(env.tool_state)
                data_bc['act_raw'].append(action_raw)
                data_bc['act_scaled'].append(action_scaled)

            # THEN take steps in the env as usual.
            obs, reward, _, info = env.step(action_scaled,
                    record_continuous_video=args.record_continuous_video,
                    img_size=args.img_size)

            # Record info.
            if args.record_continuous_video:
                frames.extend(info['flex_env_recorded_frames'])
            ep_rew += reward
            obs_l.append(obs)

            # Special to mixed media, we are strangely getting some sunk items.
            if t == 0:
                if info['height_item'] < SUNK_HEIGHT:
                    ep_start_sunk = True
                _k = 0
                while f'height_item_{_k}' in info.keys():
                    if info[f'height_item_{_k}'] < SUNK_HEIGHT:
                        ep_start_sunk_d += 1
                    _k += 1

        # UPDATE(05/11/2022): for BC03 check if the point cloud has enough of
        # target item visible after lifting (ASSUMING info says it's done). If
        # ball isn't visible (submerged in water yet lifted) then make it False.
        # Only for improving quality of BC data! Maybe 4 points is enough?
        if args.obs_mode == 'combo' and args.save_data_bc:
            # Track this to judge demonstrator quality; we have this marked
            # as a 'success', but are getting rid of this for BC data.
            if info['done'] and (info['pcl_targ_subs'] < 4):
                info_stats['num_done_but_few_pcl_targ'] += 1
            info['done'] = info['done'] and (info['pcl_targ_subs'] >= 4)

        # For MM, `info['done']` checks the sparse reward condition. It is only True
        # if that condition has been met; we check it here at the END of the episode.
        if info['done']:
            # Actually this can happen with repeated attempts (with BC#03 data).
            #assert ep_rew > 0, f'If done, should not have {ep_rew} be <= 0?'
            ep_success += 1
            this_ep_success = True
        ep_rewards.append(ep_rew)

        # Record extra stuff from end of episode for detailed analysis.
        info['sunk_at_start'] = False
        info_stats[f'distractors_sunk_{str(e).zfill(4)}'] = ep_start_sunk_d
        if ep_start_sunk:
            info_stats['sunk_item_start'] += 1
            info['sunk_at_start'] = True  # Add to normal `info`.
        if info['height_item'] < 0.1:
            # The item should normally be starting at ~0.2, so if it's like this,
            # something strange happened with the initial data generation.
            info_stats['sunk_item_end'] += 1
        if info['stuck_item']:
            info_stats['stuck_item'] += 1
        if (not info['in_bounds_x']) or (not info['in_bounds_z']):
            info_stats['oob'] += 1

        # Save GIFs if we want to test segmentation, but we should also ensure the
        # recorded frames (img_size) is the same size as the camera width.
        print(f'Finished {e+1} episodes')
        print(f'  ep success? {ep_success} (cumulative), {this_ep_success} (this one)')
        print(f'  ep reward: {ep_rew:0.2f} (just this episode)')
        print('  done but few PCLs: {}'.format(info_stats['num_done_but_few_pcl_targ']))
        if args.obs_mode == 'cam_rgb':
            save_gif(args, save_subdir, frames, ep_rew=ep_rew, ep_count=e, obs_l=None, info=info)
        elif args.obs_mode == 'point_cloud':
            save_gif(args, save_subdir, frames, ep_rew=ep_rew, ep_count=e, obs_p=obs_l, info=info)
        elif (args.obs_mode == 'segm') and (args.camera_width == args.img_size):
            save_gif(args, save_subdir, frames, ep_rew=ep_rew, ep_count=e, obs_l=obs_l, info=info)
        elif args.obs_mode == 'flow':
            obs_f = [o[1] for o in obs_l[1:]]
            save_gif(args, save_subdir, frames, ep_rew=ep_rew, ep_count=e, obs_l=None, info=info,
                obs_p=None, obs_f=obs_f)
        elif args.obs_mode in ['combo', 'combo_gt_v01', 'combo_gt_v02']:
            # I think this case is due to something with segmentation.
            if args.camera_width == args.img_size:
                # Save everything, mainly for debugging or to compare observations. First
                # two in each obs_l tuple are keypts (not used) and RGB (save separately).
                obs_s = [o[2] for o in obs_l]  # segmented, use obs_s instead of obs_l
                obs_p = [o[3] for o in obs_l]  # pc_array (could be gt)
                obs_f = [o[4] for o in obs_l[1:]]  # flow (tool)
                obs_p_gt_v01 = None
                if len(obs_l[0]) > 5:
                    #obs_p_gt_v01 = [o[5] for o in obs_l]  # pc_array (ground truth variant)
                    obs_depth = [o[5] for o in obs_l]  # wait, this is depth NOT gt PCLs!!
                else:
                    obs_depth = None
                save_gif(args, save_subdir, frames, ep_rew=ep_rew, ep_count=e, info=info,
                    obs_l=obs_s, obs_p=obs_p, obs_f=obs_f, obs_p_gt_v01=obs_p_gt_v01,
                    obs_depth=obs_depth)
            else:
                pass

        # Save BC data for this episode, if filtering only do successful ones. We do
        # save GIFs of failures, mainly helps to check that we correctly ignore. :)
        if args.save_data_bc:
            if args.filtered:
                record_data = this_ep_success
                if record_data:
                    config_idxs.append(e)
            else:
                record_data = True

            if record_data:
                n_eps = f'{str(e).zfill(4)}'
                n_obs = len(data_bc['obs'])
                data_path = osp.join(save_subdir, f'BC_{n_eps}_{n_obs}.pkl')
                with open(data_path, 'wb') as fh:
                    pickle.dump(data_bc, fh)

                if args.filtered and (len(config_idxs) == n_success_bc):
                    print(f'Exiting due to len(config_idxs): {len(config_idxs)}')
                    last_ep_bc = e
                    break

    # Collect statistics on results and save to a text file.
    print(f'\n----- Finished {args.num_variations} variations! -----')
    results_txt = osp.join(save_subdir, 'results.txt')
    stuck = info_stats['stuck_item']
    sunk_start = info_stats['sunk_item_start']
    sunk_end = info_stats['sunk_item_end']
    oob = info_stats['oob']
    done_but_few_pcl = info_stats['num_done_but_few_pcl_targ']
    with open(results_txt, 'w') as tf:
        print(f'Success:    {ep_success} / {args.num_variations}', file=tf)
        if args.save_data_bc:
            print(f'Though we only did {last_ep_bc} configs due to BC', file=tf)
            print(f'# done but few PCLs (not collected): {done_but_few_pcl}', file=tf)
        mean = np.mean(ep_rewards)
        std = np.std(ep_rewards)
        median = np.median(ep_rewards)
        print(f'Ep Rews:  {mean:0.2f} +/- {std:0.1f}', file=tf)
        print(f'  median: {median:0.2f}', file=tf)
        print(f'Stuck items?     {stuck} / {args.num_variations}', file=tf)
        print(f'OOB item?          {oob} / {args.num_variations}', file=tf)
        print(f'Sunk start? {sunk_start} / {args.num_variations}', file=tf)
        print(f'Sunk end?     {sunk_end} / {args.num_variations}', file=tf)
        _e = 0
        # Only print the non-zero ones to save space.
        while f'distractors_sunk_{str(_e).zfill(4)}' in info_stats.keys():
            dsunk = info_stats[f'distractors_sunk_{str(_e).zfill(4)}']
            if dsunk > 0:
                print(f'sunk_distractors_start {str(_e).zfill(4)}: {dsunk}', file=tf)
            _e += 1
    print(f'See results in: {results_txt}')

    # If filtering BC data, probably should save the successful config indices.
    # When doing BC later, we load data from this file to filter the configs.
    if args.save_data_bc and args.filtered:
        BC_results_txt = osp.join(save_subdir, 'BC_data.txt')
        with open(BC_results_txt, 'w') as tf:
            for e in config_idxs:
                print(e, file=tf)
        print(f'Also see BC successful configs in: {BC_results_txt}')


def run_pw(args, env):
    """Take steps through pour water env, possibly save BC data."""

    # Assign the algorithmic policy choice (if desired), OK if None (it will ignore).
    # assert args.env_name in ['PourWater']
    assert hasattr(env, 'pw_env_version') and hasattr(env, 'alg_policy')
    env.set_alg_policy(args.alg_policy)

    # Create directory for storing GIFs, etc.
    if not os.path.exists(args.save_video_dir):
        os.mkdir(args.save_video_dir)
    save_subdir = get_video_subdir(args, env)

    # Record info for reporting later.
    ep_success = 0
    ep_rewards = []
    info_stats = {} # we might not need this for pour water

    # For behavioral cloning, we want 1000 data points and to record successes.
    # EDIT: maybe not, might as well just get as many as we can? For BC#03 we might
    # just get 500 episodes, with up to 400 for training? Can use 2000 stored configs.
    # With act repeat=1 we get a lot more info per episode.
    if args.action_repeat == 1:
        n_success_bc = 500
    elif args.action_repeat == 8:
        n_success_bc = 1100
    config_idxs = []

    # Save both `frames` (for continuous video) and `obs` (for testing segmentation).
    last_ep_bc = -1
    for e in range(args.num_variations):
        data_bc = defaultdict(list)
        this_ep_success = False

        # Each call to the reset will sample a specific config ID.
        obs = env.reset(config_id=e)
        frames = [env.get_image(args.img_size, args.img_size)]
        obs_l = [obs]
        ep_rew = 0
        print()
        print(100*'=')
        print(f'----- Episode {e+1} -----')
        print(100*'=')

        for t in range(env.horizon):
            # New change, moving this to MM env's method, see documentation there.
            action_raw, action_scaled = env.get_random_or_alg_action()

            # Add time-aligned BC data, separate from `obs_l` but with same data.
            if args.save_data_bc:
                data_bc['obs'].append(obs)
                data_bc['act_raw'].append(action_raw)
                data_bc['act_scaled'].append(action_scaled)

            # THEN take steps in the env as usual.
            obs, reward, _, info = env.step(action_scaled,
                    record_continuous_video=args.record_continuous_video,
                    img_size=args.img_size)

            # Record info.
            if args.record_continuous_video:
                frames.extend(info['flex_env_recorded_frames'])
            ep_rew += reward
            obs_l.append(obs)

        # For MM, `info['done']` checks the sparse reward condition. It is only True
        # if that condition has been met; we check it here at the END of the episode.
        # Same thing happens with pour water.
        if info['done']:
            ep_success += 1
            this_ep_success = True
        ep_rewards.append(ep_rew)

        # Save GIFs if we want to test segmentation, but we should also ensure the
        # recorded frames (img_size) is the same size as the camera width.
        print(f'Finished {e+1} episodes')
        print(f'  ep success? {ep_success} (cumulative), {this_ep_success} (this one)')
        print(f'  ep reward: {ep_rew:0.2f} (just this episode)')
        if args.obs_mode == 'cam_rgb':
            save_pw_gif(args, save_subdir, frames, ep_rew=ep_rew, ep_count=e,
                    obs_c=None, info=info)
        elif args.obs_mode == 'combo':
            # The first part in each tuple are the shape states.
            obs_c = [o[1] for o in obs_l]  # RGB image (c)
            obs_s = [o[2] for o in obs_l]  # segmented (s)
            obs_p = [o[3] for o in obs_l]  # PCL array (p)
            obs_f = [o[4] for o in obs_l[1:]]  # flow (f)
            if len(obs_l[0]) > 5:
                #obs_p_gt_v01 = [o[5] for o in obs_l]  # pc_array (ground truth variant)
                obs_depth = [o[5] for o in obs_l]  # wait, this is depth NOT gt PCLs!!
            else:
                obs_depth = None
            save_pw_gif(args, save_subdir, frames, ep_rew=ep_rew, ep_count=e,
                    obs_c=obs_c, obs_s=obs_s, obs_p=obs_p, obs_f=obs_f, info=info,
                    obs_depth=obs_depth)

        # Save BC data for this episode, if filtering only do successful ones. We do
        # save GIFs of failures, mainly helps to check that we correctly ignore. :)
        if args.save_data_bc:
            if args.filtered:
                record_data = this_ep_success
                if record_data:
                    config_idxs.append(e)
            else:
                record_data = True

            if record_data:
                n_eps = f'{str(e).zfill(4)}'
                n_obs = len(data_bc['obs'])
                data_path = osp.join(save_subdir, f'BC_{n_eps}_{n_obs}.pkl')
                with open(data_path, 'wb') as fh:
                    pickle.dump(data_bc, fh)

                if args.filtered and (len(config_idxs) == n_success_bc):
                    print(f'Exiting due to len(config_idxs): {len(config_idxs)}')
                    last_ep_bc = e
                    break

    # Collect statistics on results and save to a text file.
    print(f'\n----- Finished {args.num_variations} variations! -----')
    results_txt = osp.join(save_subdir, 'results.txt')
    with open(results_txt, 'w') as tf:
        print(f'Success:    {ep_success} / {args.num_variations}', file=tf)
        if args.save_data_bc:
            print(f'Though we only did {last_ep_bc} configs due to BC', file=tf)
        mean = np.mean(ep_rewards)
        std = np.std(ep_rewards)
        median = np.median(ep_rewards)
        print(f'Ep Rews:  {mean:0.2f} +/- {std:0.1f}', file=tf)
        print(f'  median: {median:0.2f}', file=tf)
        _e = 0
    print(f'See results in: {results_txt}')

    # If filtering BC data, probably should save the successful config indices.
    # When doing BC later, we load data from this file to filter the configs.
    if args.save_data_bc and args.filtered:
        BC_results_txt = osp.join(save_subdir, 'BC_data.txt')
        with open(BC_results_txt, 'w') as tf:
            for e in config_idxs:
                print(e, file=tf)
        print(f'Also see BC successful configs in: {BC_results_txt}')


def main():
    """Daniel: same as `random_env.py` but now running a demonstrator.
    See the `bash_scripts/` folder for usage.

    Supports MixedMedia envs and PourWater envs.

    Use `record_continuous_video` to visualize successes / failures.
    We do need the `num_variations` (X) to be equal.
    Sphere scale should range from 0.05 to 0.1.

    The `img_size` here is only for querying frames that we save for the GIF. It can be a
    different size from the `camera_width` and `camera_height` values used for the native
    observation, which are 720x720 by default and are what we see with `pyflex.render()`.

    In the CoRL 2020 paper, SoftGym used 128x128 images.
    To clear up confusion: we have two types of 'tool' variables.
        tool_type: how we load it (SDF, Triangle Mesh, Rigid Body)
        tool_data: the data file we use for loading it.
    The latter will be an integer since we may get a lot of possibilities.
    The former just has 3 options so we can use a fixed set of strings.
    """
    p = argparse.ArgumentParser(description='Process some integers.')
    p.add_argument('--env_name', type=str, default='MMOneSphere')
    p.add_argument('--obs_mode', type=str, default='cam_rgb')
    p.add_argument('--act_mode', type=str, default='translation')
    p.add_argument('--camera_name', type=str, default='top_down')
    p.add_argument('--camera_width', type=int, default=256)
    p.add_argument('--camera_height', type=int, default=256)
    p.add_argument('--headless', type=int, default=0)
    p.add_argument('--num_variations', type=int, default=1)
    p.add_argument('--save_video_dir', type=str, default='./data_demo/')
    p.add_argument('--img_size', type=int, default=256, help='Size of the recorded videos')
    p.add_argument('--render_mode', type=str, default='particle', help='Or fluid mode')
    p.add_argument('--record_continuous_video', action='store_true', default=False)
    p.add_argument('--use_cached_states', action='store_true', default=False)
    p.add_argument('--save_cached_states', action='store_true', default=False)
    p.add_argument('--alg_policy', type=str, default=None, help='if None, then random pol.')
    p.add_argument('--save_data_bc', action='store_true', default=False)
    p.add_argument('--filtered', action='store_true', default=False)
    p.add_argument('--action_repeat', type=int, default=8) # CAREFUL, use 1 or 8.
    # NOTE(daniel): will remove these later. Please don't adjust these.
    p.add_argument('--n_substeps', type=int, default=2, help='see FleX docs')
    p.add_argument('--n_iters', type=int, default=4, help='see FleX docs')
    p.add_argument('--inv_dt', type=float, default=100, help='dt=1/inv_dt, physics sim step')
    p.add_argument('--inv_mass', type=float, default=0.50, help='inv mass for sphere')
    p.add_argument('--sphere_scale', type=str, default=0.060, help='temporary')
    p.add_argument('--act_noise', type=float, default=0.0, help='temporary')
    p.add_argument('--tool_type', type=str, default='sdf', help='temporary')
    p.add_argument('--tool_data', type=int, default=2, help='2 is what we normally used')
    p.add_argument('--tool_scale', type=float, default=0.28, help='temporary')
    p.add_argument('--collect_visuals', action='store_true', default=False)
    args = p.parse_args()
    env_kwargs = env_arg_dict[args.env_name]

    # If just collecting visuals for a paper we want to keep the seed FIXED.
    if args.collect_visuals:
        np.random.seed(0)
        random.seed(0)

    # See `softgym_mixed_media.h` for mapping the integer data to the file name.
    assert args.tool_data in [0, 1, 2, 3, 4]
    assert args.tool_type in ['sdf', 'tmesh', 'rbody']

    # If BC, we might want to save as many different obs as possible.
    if args.save_data_bc:
        assert args.obs_mode in ['combo', 'combo_gt_v01', 'combo_gt_v02'], \
                'If using BC, should save different obs.'

    # Generate and save the initial states for running this environment for the first time
    env_kwargs['use_cached_states'] = args.use_cached_states
    env_kwargs['save_cached_states'] = args.save_cached_states
    env_kwargs['observation_mode'] = args.obs_mode
    env_kwargs['action_mode'] = args.act_mode
    env_kwargs['num_variations'] = args.num_variations
    env_kwargs['headless'] = args.headless
    env_kwargs['render_mode'] = args.render_mode
    env_kwargs['camera_width'] = args.camera_width
    env_kwargs['camera_height'] = args.camera_height
    env_kwargs['camera_name'] = args.camera_name  # note: override this for SpheresLadle
    env_kwargs['render'] = True
    env_kwargs['action_repeat'] = args.action_repeat
    if args.action_repeat == 1:
        env_kwargs['horizon'] = 600
    elif args.action_repeat == 8:
        env_kwargs['horizon'] = 100
    else:
        raise ValueError(args.action_repeat)

    # Adjust cached states path. If changing, change `save_subdir`. NOTE(daniel):
    # making special cases based on variations, normally we do NOT mess with this.
    # Edit: actually I think we now should always do this.
    if args.num_variations not in [1000]:
        env_kwargs['cached_states_path'] = (f'{args.env_name}_'
            f'nVars_{str(args.num_variations).zfill(4)}.pkl')

    # If `use_cached_states=False`, then we construct `num_variations` starting states
    # first during env construction. Otherwise, no need to regenerate starting states.
    if not env_kwargs['use_cached_states']:
        print('Waiting to generate environment variations.')

    # We don't really use these custom args anymore, should simplify later.
    if args.env_name in ['MMOneSphere', 'MMMultiSphere']:
        env = normalize(SOFTGYM_ENVS[args.env_name](
            n_substeps=args.n_substeps,
            n_iters=args.n_iters,
            inv_dt=args.inv_dt,
            inv_mass=args.inv_mass,
            sphere_scale=args.sphere_scale,
            act_noise=args.act_noise,
            tool_type=args.tool_type,
            tool_data=args.tool_data,
            tool_scale=args.tool_scale,
            **env_kwargs))
    elif args.env_name in ['PourWater', 'SpheresLadle', "PourWater6D"]:
        env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))
    else:
        raise ValueError(args.env_name)

    # Assume if we're saving the cached states that we exit ASAP. This is optional, we
    # can just continue and the code works since it saves configs in the FlexEnv `env`.
    if args.save_cached_states:
        print('Done with saving cached states. Exit now.')
        sys.exit()

    # Actually run the env.
    if args.env_name in ['MMOneSphere', 'MMMultiSphere']:
        run_mm(args, env)
    elif args.env_name in ['SpheresLadle']:
        run_spheres(args, env)
    elif args.env_name in ['PourWater', 'PourWater6D']:
        run_pw(args, env)
    else:
        raise ValueError(args.env_name)


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
        marker_dict = {"size": 4}
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
        marker={"size": 3, "color": "darkred"},
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


def _create_flow_frame(sample):
    """For mixed media.

    Update 08/23/2022: some tweaks to allow for making videos for project website
    with aligned RGB images and segmented point clouds.
    """
    pts = sample['points']
    flow = sample['flow']

    # Easiest way to debug: the 'eye' is the camera position.
    camera = dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.25, y=1.25, z=1.25)
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
    ts = _flow_traces_v2(pts, flow, sizeref=2., scene="scene1")
    for t in ts:
        f.add_trace(t)
    f.update_layout(
        # NOTE (08/23) tweaking these for flow visualizations for website.
        scene1=_3d_scene_fixed([-0.40, 0.12], [-0.40, 0.12], [0.00, 0.60]),
        scene_camera=camera,
    )

    fig_bytes = f.to_image(format="png", engine="kaleido")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)


def _create_flow_frame_pw(sample, return_f=False, sizeref=4, debug_print=True):
    """For pour water.

    If making a plot for a paper, should make the x and y coordinate ranges
    be the same to avoid stretching the box. Also, sometimes we might have to
    zoom into images which means likely tuning the downsampling ratio and the
    marker sizes, etc.

    If downsampling we might sometimes want to keep the blue points and just
    downsample flow, for that downsample the input to `_flow_traces_v2()`.

    08/23/2022: don't forget we should flip the y values ... as shown in
        my SoftAgent debugging. Using 600x600 to align them with other GIFs.
    11/09/2022: tweaks to get this teaser figure updated. Also this includes tool
        points in the sample which I thought we had handled??
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
    pts = sample['points']
    flow = sample['flow']
    #pts = pts[::downsample]
    #flow = flow[::downsample]
    if debug_print:
        print(f'Before downsampling, pts: {pts.shape}, flow: {flow.shape}')

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

    # Actually we'll downsample _here_, just for flow traces.
    df = 10  # Maybe 10 for the translation only and 4-5 for rotation?
    # Using rotation might require longer sizeref like 7 instead of, say, 4?
    ts = _flow_traces_v2(pts[::df], flow[::df], sizeref=sizeref, scene="scene1", pour_water=True)
    for t in ts:
        f.add_trace(t)

    # Amusingly now we don't need to flip the y-axis ranges, I think this is due to
    # setting the eye (see above) in a clever way.
    f.update_layout(
        scene1=_3d_scene_fixed([-0.10, 0.70], [-0.40, 0.40], [0.0, 0.80]),
        scene_camera=camera,
    )

    if return_f:
        return f

    fig_bytes = f.to_image(format="png", engine="kaleido")
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)


def save_flow(obs_f, savedir, pour_water=False):
    if pour_water:
        with mp.Pool(mp.cpu_count()) as pool:
            frames = pool.map(_create_flow_frame_pw, obs_f)
        save_numpy_as_gif(np.array(frames), osp.join(savedir, 'flow_pw.gif'), fps=10)
    else:
        with mp.Pool(mp.cpu_count()) as pool:
            frames = pool.map(_create_flow_frame, obs_f)
        save_numpy_as_gif(np.array(frames), osp.join(savedir, 'flow_mm.gif'), fps=10)
    return np.array(frames)


def save_flow_html(obs_f, obs_p, savedir, pour_water=True):
    """Just save the flow html.
    The `obs_p` is solely as another sanity check.
    """
    if pour_water:
        for ff in range(len(obs_f)):
            # Sanity check.
            segm_pcl = obs_p[ff]
            sample_f = obs_f[ff]
            assert sample_f['points'].shape[0] == len(np.where(segm_pcl[:,3] == 1.)[0]), \
                '{} and {}'.format(sample_f['points'].shape, segm_pcl[:,3])
            # Now actually create the frame.
            fig = _create_flow_frame_pw(obs_f[ff], return_f=True)
            htmldir = osp.join(savedir, f'flow_{str(ff).zfill(3)}.html')
            fig.write_html(htmldir)
    sys.exit()


def get_video_subdir(args, env):
    """Get the directory where we save videos/GIFs and other stats."""
    suffix = f'{args.env_name}'
    if hasattr(env, 'mm_env_version'):
        suffix = f'{suffix}_v{str(env.mm_env_version).zfill(2)}'
    if hasattr(env, 'pw_env_version'):
        suffix = f'{suffix}_v{str(env.pw_env_version).zfill(2)}'
    if hasattr(env, 'spheres_env_version'):
        suffix = f'{suffix}_v{str(env.spheres_env_version).zfill(2)}'
    if args.save_data_bc:
        if args.filtered:
            suffix = f'{suffix}_BClone_filtered_wDepth'
        else:
            suffix = f'{suffix}_BClone_UNfiltered_wDepth'
    if hasattr(env, 'alg_policy'):
        suffix = f'{suffix}_{env.alg_policy}'
    suffix = f'{suffix}_nVars_{args.num_variations}_obs_{args.obs_mode}'
    suffix = f'{suffix}_act_{args.act_mode}'
    save_subdir = osp.join(args.save_video_dir, suffix)
    if not os.path.exists(save_subdir):
        os.mkdir(save_subdir)
    return save_subdir


def print_info(info):
    """Debugging."""
    ignore = ['flex_env_recorded_frames', 'performance']
    for key in info:
        if key not in ignore:
            print(f'  {key}:  {info[key]}')


if __name__ == '__main__':
    main()
