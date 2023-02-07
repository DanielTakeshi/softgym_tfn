"""Double check the segmentation for our datasets."""
import os
from os.path import join
import pickle
import numpy as np
np.set_printoptions(suppress=True, precision=5, edgeitems=20, linewidth=200)
from softgym.utils import visualization


def inspect_pouring_6D(
        bc_data_dir,
        filtered=True,
        n_train_demos=100,
        n_valid_demos=25,
        max_train_demo=1000,
        ep_len=100,
        ep_part_to_use=99,
    ):
    """See `BehavioralCloningData._load_from_data()` from SoftAgent.

    Some of the arguments here assume PourWater6D values (e.g., `ep_part_to_use`).
    Instead of saving to a replay buffer, just save the point clouds, and then plot
    them in matplotlib to visualize the segmentation.
    """

    # New stuff for BC to handle filtering, since we filter and track the configs
    # which were successful, and should only evaluate on these at test time.
    first_idx_train = 0   # (s,a) pair, inclusive
    last_idx_train = -1   # (s,a) pair, exclusive
    first_idx_valid = -1  # (s,a) pair, inclusive
    last_idx_valid = -1   # (s,a) pair, exclusive
    _train_config_idxs = None  # (filtered) train configs only
    _valid_config_idxs = None  # (filtered) valid configs only
    idx = 0

    def get_obs_tool_flow(pcl, tool_flow):
        # NOTE(daniel): this is to get (obs,act) encoded correctly + consistently.
        # If `pcl` is segmented point cloud from time t-1, and `tool_flow`
        # is the flow from time t, then their tool points shoud coincide.
        # We will always provide (2000,d)-sized PCLs; in training, can resize.
        pcl_tool = pcl[:,3] == 1
        tf_pts = tool_flow['points']
        tf_flow = tool_flow['flow']
        n_tool_pts_obs = np.sum(pcl_tool)
        n_tool_pts_flow = tf_pts.shape[0]
        # First shapes only equal if: (a) fewer than max pts or (b) no item/distr.
        assert tf_pts.shape[0] <= pcl.shape[0], f'{tf_pts.shape}, {pcl.shape}'
        # assert tf_pts.shape == tf_flow.shape, f'{tf_pts.shape}, {tf_flow.shape}'
        assert n_tool_pts_obs == n_tool_pts_flow, f'{n_tool_pts_obs}, {n_tool_pts_flow}'
        assert np.array_equal(pcl[:n_tool_pts_obs,:3], tf_pts)  # yay :)
        flow_dim = tf_flow.shape[1]
        a = np.zeros((pcl.shape[0], flow_dim))  # all non-tool point rows get 0s
        a[:n_tool_pts_obs] = tf_flow   # actually encode flow for BC purposes
        return (pcl, a)

    # Load pickle paths into list. One item is one demonstration.
    print(f'\nLoading data for Behavioral Cloning: {bc_data_dir}')
    pkl_paths = sorted([
        join(bc_data_dir,x) for x in os.listdir(bc_data_dir)
            if x[-4:] == '.pkl' and 'BC' in x])

    # If filtering, load file which specifies config indices to keep. This is
    # later used in SoftGym since we'll have more configs and need to subsample.
    print(f'Loading {len(pkl_paths)} configs (i.e., episodes) from data.')
    if filtered:
        filt_fname = join(bc_data_dir, 'BC_data.txt')
        assert os.path.exists(filt_fname), f'{filt_fname}'
        with open(filt_fname, 'rb') as fh:
            config_idxs = [int(l.rstrip()) for l in fh]
        _filtered_config_idxs = config_idxs
    else:
        _filtered_config_idxs = [i for i in range(len(pkl_paths))]

    # Handle train and valid _config_ indexes (we only want filtered ones).
    _train_config_idxs = _filtered_config_idxs[:n_train_demos]
    _valid_config_idxs = _filtered_config_idxs[
            max_train_demo : max_train_demo + n_valid_demos]
    print(f'First {n_train_demos} idxs of starting configs are training.')
    print(f'Config at filtered index {max_train_demo} is first valid episode.')
    # These give the indices for the original (including filtered) set of configs.
    print(f'Train configs (start,end), (inclusive,inclusive): '
        f'{_train_config_idxs[0]}, {_train_config_idxs[-1]}')
    print(f'Valid configs (start,end), (inclusive,inclusive): '
        f'{_valid_config_idxs[0]}, {_valid_config_idxs[-1]}')

    ## Action bounds. Careful about rotations. NOTE(daniel): ignore in this script.
    #assert self.action_type in ALL_ACTS, self.action_type
    #print(f'Action type: {self.action_type}. Act bounds:')
    #print(f'  lower: {self.action_lb}')
    #print(f'  upper: {self.action_ub}')
    n_diff_ee_flow = 0

    # Iterate through filtered paths, only keeping what we need. We later use
    # indices to restrict the sampling.
    for pidx,pkl_pth in enumerate(pkl_paths):
        DATA = []
        if pidx % 50 == 0:
            print(f'  checking episode/config idx: {pidx} on idx {idx}')

        # Handle train / valid logic limits.
        if pidx == n_train_demos:
            print(f'  finished {pidx} demos, done w/train at idx {idx}')
            last_idx_train = idx
            if pidx < max_train_demo:
                continue
        elif n_train_demos < pidx < max_train_demo:
            continue
        if pidx == max_train_demo:
            print(f'  now on {pidx}, start of valid demos')
            first_idx_valid = idx
        if pidx == max_train_demo + n_valid_demos:
            print(f'  on {pidx}, exit now after {n_valid_demos} valid demos')
            break

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

        # Add each (obs,act) from this episode into the data buffer.
        # The `obs` is actually a tuple, so extract appropriate item.
        for t in range(ep_part_to_use):
            obs_tuple = data['obs'][t]
            act_raw = data[act_key][t]

            # Keep this much simpler, just care about `obs`, the pcl array.
            # This has the position and the segmentation information.
            obs = obs_tuple[3]
            DATA.append(obs)

        # Check this episode and plot?
        visualization.save_pointclouds(
            obs_p=DATA,
            savedir='tmp',  # create a `tmp/` if you don't have it
            suffix=f'pointcloud_segm_{str(pidx).zfill(3)}.gif',
            n_views=3,
            return_np_array=False,
            pour_water=True,
        )


if __name__ == "__main__":
    # Make sure this is a PourWater6D dataset! Change according to your machine.
    HEAD = 'data_demo'
    data_dir = join(HEAD, 'PourWater6D_v01_BClone_filtered_wDepth_pw_algo_v02_nVars_2000_obs_combo_act_translation_axis_angle_withWaterFrac')
    assert os.path.exists(data_dir), data_dir

    # Inspect by going through the observations and plotting them.
    inspect_pouring_6D(data_dir)
