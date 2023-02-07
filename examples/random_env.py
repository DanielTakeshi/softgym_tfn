import sys
import os
import os.path as osp
import argparse
import numpy as np
import random
from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from softgym.utils.visualization import save_numpy_as_gif


def main():
    # NOTE(daniel): if using Mixed Media, please use `demonstrator.py` instead.
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--env_name', type=str, default='ClothDrop')
    parser.add_argument('--headless', type=int, default=0, help='Whether to run the environment with headless rendering')
    parser.add_argument('--num_variations', type=int, default=1, help='Number of environment variations to be generated')
    parser.add_argument('--save_video_dir', type=str, default='./data/', help='Path to the saved video')
    parser.add_argument('--img_size', type=int, default=256, help='Size of the recorded videos')
    # Daniel: other stuff for tuning. We should figure out another way to handle seeding.
    parser.add_argument('--n_substeps', type=int, default=2, help='see FleX docs')
    parser.add_argument('--n_iters', type=int, default=4, help='see FleX docs')
    parser.add_argument('--inv_dt', type=float, default=100, help='dt=1/inv_dt, physics sim step')
    parser.add_argument('--inv_mass', type=float, default=1.0, help='inv mass for sphere')
    parser.add_argument('--render_mode', type=str, default='particle', help='Or fluid mode')
    parser.add_argument('--other', type=str, default='', help='Used to add to suffix')
    parser.add_argument('--seed', type=int, default=0, help='random seed (TODO check)')
    parser.add_argument('--camera', type=str, default='default_camera')
    args = parser.parse_args()
    env_kwargs = env_arg_dict[args.env_name]

    # Seeding (temporary). TODO(daniel) check if this is working.
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Generate and save the initial states for running this environment for the first time
    env_kwargs['use_cached_states'] = False
    env_kwargs['save_cached_states'] = False
    env_kwargs['num_variations'] = args.num_variations
    env_kwargs['render'] = True
    env_kwargs['headless'] = args.headless
    env_kwargs['render_mode'] = args.render_mode
    env_kwargs['camera_name'] = args.camera

    if not env_kwargs['use_cached_states']:
        print('Waiting to generate environment variations. May take 1 minute for each variation...')
    if 'MM' in args.env_name:
        print('Please use demonstrator.py script')
        sys.exit()
    else:
        env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))
    env.reset()
    frames = [env.get_image(args.img_size, args.img_size)]

    # Daniel: random policy.
    for i in range(env.horizon):
        action = env.action_space.sample()
        # By default, the environments will apply action repetition. The option
        # of record_continuous_video provides rendering of all intermediate
        # frames. Only use this option for visualization as it increases computation.
        _, _, _, info = env.step(action, record_continuous_video=True, img_size=args.img_size)
        frames.extend(info['flex_env_recorded_frames'])

    if args.save_video_dir is not None:
        save_name = osp.join(args.save_video_dir, args.env_name + '.gif')
        if args.env_name == 'MixedMediaRetrieval':
            # Just for this test env, add more info and try avoid overriding GIFs.
            save_name = (
                    f'{save_name[:-4]}_'
                    f'ss_{str(args.n_substeps).zfill(2)}_'
                    f'itrs_{str(args.n_iters).zfill(2)}_'
                    f'invmass_{str(args.inv_mass).zfill(1)}_'
                    f'render_{args.render_mode}'
            )
            if args.other != '':
                save_name = f'{save_name}_{args.other}'
            _, save_name_tail = osp.split(save_name)
            gif_files = sorted([x for x in os.listdir(args.save_video_dir)
                    if save_name_tail in x and x[-4:]=='.gif'])
            n_gifs = len(gif_files)
            save_name = f'{save_name}_{str(n_gifs).zfill(4)}.gif'
        save_numpy_as_gif(np.array(frames), save_name)
        print('Video generated and save to {}'.format(save_name))


if __name__ == '__main__':
    main()
