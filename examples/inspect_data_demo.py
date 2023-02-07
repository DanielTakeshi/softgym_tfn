"""Use to inspect the data we might generate.

Expected usage is to complement the BC package in SoftAgent by using this
code to inspect things like action magnitudes, etc. Some code may be shared,
especially among the replay buffer loading.
"""
import os
from os.path import join
import pickle
import numpy as np
np.set_printoptions(suppress=True, precision=5, edgeitems=20, linewidth=200)
import matplotlib.pyplot as plt


def get_obs_tool_flow(pcl, tool_flow):
    """
    NOTE(daniel): this is to get (obs,act) encoded correctly + consistently.
    If `pcl` is segmented point cloud from time t-1, and `tool_flow`
    is the flow from time t, then their tool points shoud coincide.
    """
    pcl_tool = pcl[:,3] == 1
    tf_pts = tool_flow['points']
    tf_flow = tool_flow['flow']
    n_tool_pts_obs = np.sum(pcl_tool)
    n_tool_pts_flow = tf_pts.shape[0]
    # First shapes only equal if: (a) fewer than max pts or (b) no item/distr.
    assert tf_pts.shape[0] <= pcl.shape[0], f'{tf_pts.shape}, {pcl.shape}'
    assert tf_pts.shape == tf_flow.shape, f'{tf_pts.shape}, {tf_flow.shape}'
    assert n_tool_pts_obs == n_tool_pts_flow, f'{n_tool_pts_obs}, {n_tool_pts_flow}'
    assert np.array_equal(pcl[:n_tool_pts_obs,:3], tf_pts)  # yay :)
    a = np.zeros((pcl.shape[0], 3))  # all non-tool point rows get 0s
    a[:n_tool_pts_obs] = tf_flow   # actually encode flow for BC purposes
    return (pcl, a)


DATA = {
    'obs': [],
    'act': [],
    'info': [],
}


def add(obs, action, info, scale_targets=False, scale_pcl_flow=True, scale_pcl_val=250):
    action_ub = np.array([0.004, 0.004, 0.004])

    if scale_pcl_flow:
        ## Should be fine, just expressing it in different units.
        #obs[:, :3] *= self.scale_pcl_val  # first 3 columns are xyz
        action *= scale_pcl_val  # applies for EE translations and flow
    elif scale_targets:
        # Scale transl. parts to (-1,1), assumes symmetrical bounds!
        # Either action is just (3,) for EE or (N,3) for flow.
        if len(action.shape) == 1:
            action[:3] = action[:3] / action_ub[:3]
        else:
            action[:,:3] = action[:,:3] / action_ub[:3]

    DATA['obs'].append(obs)
    DATA['act'].append(action)
    DATA['info'].append(info)


def inspect(bc_data_dir, ep_len, encoder_type='pn++', action_type='ee',
        action_mode='translation', action_repeat=8, ignore_valid=False):
    """TODO"""
    if action_repeat == 1:
        ep_len = 600
        ep_part_to_use = ep_len - 1
        max_train_demo = 400
    elif action_repeat == 8:
        ep_len = 100
        if ('_v02_' in bc_data_dir) or ('_v04_' in bc_data_dir):
            ep_part_to_use = int(0.75 * ep_len)  # for {v02,v04}
        else:
            ep_part_to_use = ep_len - 1  # for {v08}
        max_train_demo = 1000
    else:
        raise ValueError(action_repeat)

    # Load paths, mostly following replay buffer code.
    pkl_paths = sorted([
        join(data_dir,x) for x in os.listdir(data_dir)
            if x[-4:] == '.pkl' and 'BC' in x])
    n_train_demos = 100
    n_valid_demos = 100
    valid_config_start = 1000

    # If filtering, load file which specifies config indices to keep. This is
    # later used in SoftGym since we'll have more configs and need to subsample.
    print(f'Loading {len(pkl_paths)} configs (i.e., episodes) from: {data_dir}')
    filt_fname = join(data_dir, 'BC_data.txt')
    assert os.path.exists(filt_fname), f'{filt_fname}'
    with open(filt_fname, 'rb') as fh:
        config_idxs = [int(l.rstrip()) for l in fh]
    filtered_config_idxs = config_idxs

    # Handle train and valid _config_ indexes (we only want filtered ones).
    train_config_idxs = filtered_config_idxs[:n_train_demos]
    valid_config_idxs = filtered_config_idxs[
            max_train_demo : max_train_demo + n_valid_demos]
    print(f'First {n_train_demos} idxs of starting configs are training.')
    print(f'Config at filtered index {max_train_demo} is first valid episode.')
    # These give the indices for the original (including filtered) set of configs.
    print(f'Train configs (start,end), (inclusive,inclusive): '
        f'{train_config_idxs[0]}, {train_config_idxs[-1]}')
    print(f'Valid configs (start,end), (inclusive,inclusive): '
        f'{valid_config_idxs[0]}, {valid_config_idxs[-1]}')

    # Iterate through filtered paths, only keeping what we need. We later use
    # indices to restrict the sampling.
    for pidx,pkl_pth in enumerate(pkl_paths):
        if pidx % 50 == 0:
            print(f'  checking episode/config idx: {pidx}')

        # Handle train / valid logic limits.
        if pidx == n_train_demos:
            print(f'  finished {pidx} demos, done collecting train')
            if pidx < max_train_demo:
                continue
        elif n_train_demos < pidx < max_train_demo:
            continue
        if pidx == max_train_demo:
            print(f'  now on {pidx}, start of valid demos')
        if pidx == max_train_demo + n_valid_demos:
            print(f'  on {pidx}, exit now after {n_valid_demos} valid demos')
            break
        if ignore_valid:
            if pidx >= valid_config_start:
                continue

        # Each 'data' is one episode, with 'obs' and 'act' keys.
        with open(pkl_pth, 'rb') as fh:
            data = pickle.load(fh)
        act_key = 'act_raw'
        len_o = len(data['obs'])
        len_a = len(data[act_key])  # TODO(daniel) use act_scaled?
        if len_a == 0:
            # We changed keys from act -> {act_raw,act_scaled} 04/26
            len_a = len(data['act'])
            act_key = 'act'
            print(f'FYI, using outdated key for actions: {act_key}')
        assert len_o == len_a, f'{len_o} vs {len_a}'
        assert len_o == ep_len, len_o

        # Add each (obs,act) from this episode into the data buffer.
        # The `obs` is actually a tuple, so extract appropriate item.
        for t in range(ep_part_to_use):
            obs_tuple = data['obs'][t]
            act_raw = data[act_key][t]

            if encoder_type == 'pixel':
                obs = np.transpose(obs_tuple[1], (2,0,1))
            elif encoder_type == 'segm':
                obs = np.transpose(obs_tuple[2], (2,0,1))
            elif encoder_type == 'pn++':
                obs = obs_tuple[3]

            if action_type == 'ee':
                # If action repeat > 1, this ee is repeated that many times.
                assert action_mode == 'translation', action_mode
                act = act_raw
            elif action_type == 'flow':
                assert (action_mode in ['translation',
                    'translation_axis_angle']), action_mode
                # Want the _next_ observation at time t+1. Careful, for action
                # repeat, flow only considers before/after ALL the repetitions.
                obs_tuple_next = data['obs'][t+1]
                tool_flow = obs_tuple_next[4]
                obs, act = get_obs_tool_flow(obs, tool_flow)
                # Careful, if using rotations, cannot just divide by act repeat.
                # With translations, need to divide flow to make it like ee since
                # we only see the stuff before / after ALL act repetitions.
                if action_mode == 'translation':
                    act = act / action_repeat
                elif action_mode == 'translation_axis_angle':
                    assert action_repeat == 1
            else:
                raise NotImplementedError(action_type)

            # Pull keypoints from observation
            keypoints = obs_tuple[0]

            add(obs=obs, action=act, info=keypoints)

    # --------------------------------------------------------------------------- #
    print(f'Done loading BC data.')
    obses = np.array(DATA['obs'])
    acts = np.array(DATA['act'])
    print(f'  obs: {obses.shape}')
    print(f'  act: {acts.shape}')
    print(f'    min, max: {np.min(acts,axis=0)}, {np.max(acts,axis=0)}')
    print(f'    medi:     {np.median(acts,axis=0)}')
    print(f'    mean:     {np.mean(acts,axis=0)}')
    print(f'    std:      {np.std(acts,axis=0)}\n')
    # --------------------------------------------------------------------------- #

    # Might also want to plot some stuff. For translation with ee actions:
    if action_type == 'ee':
        _plot_acts_ee(acts)
    elif action_type == 'eepose':
        _plot_acts_eepose(acts)
    _plot_obs_data(obses)


def _plot_acts_ee(acts):
    figname = 'fig_acts_ee.png'
    titlesize = 32
    ticksize = 28
    legendsize = 23
    nrows, ncols = 1, 3
    fig, ax = plt.subplots(nrows, ncols, squeeze=False, figsize=(11*ncols, 7*nrows))
    ax[0,0].set_title(f'$\Delta$x', size=titlesize)
    ax[0,1].set_title(f'$\Delta$y', size=titlesize)
    ax[0,2].set_title(f'$\Delta$z', size=titlesize)
    ax[0,0].hist(acts[:,0], label='x')
    ax[0,1].hist(acts[:,1], label='y')
    ax[0,2].hist(acts[:,2], label='z')
    ax[0,0].set_xlim([-1,1])
    ax[0,1].set_xlim([-1,1])
    ax[0,2].set_xlim([-1,1])
    ax[0,0].set_yscale('log')
    ax[0,2].set_yscale('log')
    for r in range(nrows):
        for c in range(ncols):
            leg = ax[r,c].legend(loc="best", ncol=1, prop={'size':legendsize})
            for legobj in leg.legendHandles:
                legobj.set_linewidth(5.0)
            ax[r,c].tick_params(axis='x', labelsize=ticksize)
            ax[r,c].tick_params(axis='y', labelsize=ticksize)
    plt.tight_layout()
    plt.savefig(figname)
    print(f'See plot: {figname}')


def _plot_acts_eepose(acts):
    figname = 'fig_acts_eepose.png'
    titlesize = 32
    ticksize = 28
    legendsize = 23
    nrows, ncols = 1, 3
    fig, ax = plt.subplots(nrows, ncols, squeeze=False, figsize=(11*ncols, 7*nrows))
    ax[0,0].set_title(f'$\Delta$x', size=titlesize)
    ax[0,1].set_title(f'$\Delta$y', size=titlesize)
    ax[0,2].set_title(f'$\Delta$z', size=titlesize)
    ax[0,0].hist(acts[:,0], label='x')
    ax[0,1].hist(acts[:,1], label='y')
    ax[0,2].hist(acts[:,2], label='z')
    ax[0,0].set_xlim([-1,1])
    ax[0,1].set_xlim([-1,1])
    ax[0,2].set_xlim([-1,1])
    for r in range(nrows):
        for c in range(ncols):
            leg = ax[r,c].legend(loc="best", ncol=1, prop={'size':legendsize})
            for legobj in leg.legendHandles:
                legobj.set_linewidth(5.0)
            ax[r,c].tick_params(axis='x', labelsize=ticksize)
            ax[r,c].tick_params(axis='y', labelsize=ticksize)
    plt.tight_layout()
    plt.savefig(figname)
    print(f'See plot: {figname}')


def _plot_obs_data(obs):
    # Extract all nonzero points in PCLs and stack them.
    obs_nonzero = []
    for ob in obs:
        idxs_tool = np.where(ob[:,3] == 1)[0]
        idxs_targ = np.where(ob[:,4] == 1)[0]
        idxs = np.concatenate((idxs_targ, idxs_tool))
        ob = ob[idxs]
        obs_nonzero.append(ob)
    # (N,5) where N = sum of ALL nonzero in all points
    obs_nonzero = np.vstack(obs_nonzero)

    figname = 'fig_obs_data.png'
    titlesize = 32
    ticksize = 28
    legendsize = 23
    nrows, ncols = 1, 3
    fig, ax = plt.subplots(nrows, ncols, squeeze=False, figsize=(11*ncols, 7*nrows))
    ax[0,0].set_title(f'PCL $x$ coords', size=titlesize)
    ax[0,1].set_title(f'PCL $y$ coords', size=titlesize)
    ax[0,2].set_title(f'PCL $z$ coords', size=titlesize)
    ax[0,0].hist(obs_nonzero[:,0], bins=30, rwidth=0.95, label='x')
    ax[0,1].hist(obs_nonzero[:,1], bins=30, rwidth=0.95, label='y')
    ax[0,2].hist(obs_nonzero[:,2], bins=30, rwidth=0.95, label='z')
    #ax[0,0].set_xlim([-1,1])
    #ax[0,1].set_xlim([-1,1])
    #ax[0,2].set_xlim([-1,1])
    for r in range(nrows):
        for c in range(ncols):
            leg = ax[r,c].legend(loc="best", ncol=1, prop={'size':legendsize})
            for legobj in leg.legendHandles:
                legobj.set_linewidth(5.0)
            ax[r,c].tick_params(axis='x', labelsize=ticksize)
            ax[r,c].tick_params(axis='y', labelsize=ticksize)
    plt.tight_layout()
    plt.savefig(figname)
    print(f'See plot: {figname}')


if __name__ == "__main__":
    # For action_repeat=1 the lengths of episodes are 600.
    HEAD = '/data/seita/softgym_mm/data_demo'
    #data_dir = join(HEAD, 'MMOneSphere_v01_BClone_filtered_ladle_algorithmic_v02_nVars_2000_obs_combo_act_translation')
    data_dir = join(HEAD, 'MMOneSphere_v01_BClone_filtered_ladle_algorithmic_v08_nVars_2000_obs_combo_act_translation')
    assert os.path.exists(data_dir), data_dir

    ep_len = 100
    action_repeat = 8
    action_mode = 'translation'
    encoder_type = 'pn++'
    action_type = 'ee'
    ignore_valid = True

    inspect(data_dir, ep_len, encoder_type, action_type, action_mode,
            action_repeat, ignore_valid)