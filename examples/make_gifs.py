import os
from os.path import join
import imageio
import argparse
import numpy as np
import sys


def merge_gifs(args):
    """Merge GIFs assuming same lengths. Ignore any with 'merged' keyword."""
    files = [join('data',x) for x in os.listdir('data')
        if args.env in x and args.param in x and 'merged' not in x]
    files = sorted(files)
    gifs = []
    gif_length = -1
    for f in files:
        g = imageio.get_reader(f)
        print(f'{f}, frames: {g.get_length()}')
        gifs.append(g)
        if gif_length == -1:
            gif_length = g.get_length()
        else:
            if g.get_length() != gif_length:
                print(f'Warning! Different lengths!')
                print(f'We will exit now, but we could just take the minimum.')
                sys.exit()

    # New GIF writer. Default fps=10.
    new_gname = join('data', f'{args.env}_{args.param}_merged_{len(gifs)}x.gif')
    new_gif = imageio.get_writer(new_gname, fps=30)

    print('Creating the GIF...')
    for _ in range(gif_length):
        # Get the frame from the first gif.
        new_image = gifs[0].get_next_data()

        # Iterate through the rest.
        for g in gifs[1:]:
            frame = g.get_next_data()
            new_image = np.hstack((new_image, frame))
        new_gif.append_data(new_image)

    # Handle closing.
    for g in gifs:
        g.close()
    new_gif.close()
    print(f'Please see {new_gname} for the updated GIF.')


def merge_gifs_list(args, data):
    """Merge GIFs from this particular list."""
    files = sorted(data['files'])
    gifs = []
    gif_length = -1
    for f in files:
        g = imageio.get_reader(f)
        print(f'{f}, frames: {g.get_length()}')
        gifs.append(g)
        if gif_length == -1:
            gif_length = g.get_length()
        else:
            if g.get_length() != gif_length:
                print(f'Warning! Different lengths!')
                print(f'We will exit now, but we could just take the minimum.')
                sys.exit()

    # New GIF writer. Default fps=10.
    gif_name = data['name']
    new_gname = join('data', f'{args.env}_{gif_name}_merged_{len(gifs)}x.gif')
    new_gif = imageio.get_writer(new_gname, fps=24)

    print('Creating the GIF...')
    for _ in range(gif_length):
        # Get the frame from the first gif.
        new_image = gifs[0].get_next_data()

        # Iterate through the rest.
        for g in gifs[1:]:
            frame = g.get_next_data()
            new_image = np.hstack((new_image, frame))
        new_gif.append_data(new_image)

    # Handle closing.
    for g in gifs:
        g.close()
    new_gif.close()
    print(f'Please see {new_gname} for the updated GIF.')


if __name__ == "__main__":
    # Example: python examples/misc_utils.py --env PassWater --param radius
    p = argparse.ArgumentParser(description='Misc stuff.')
    p.add_argument('--env', type=str, help='The SoftGym env')
    p.add_argument('--param', type=str, help='Filter by keyword')
    args = p.parse_args()
    assert args.env is not None

    # For a given single param
    #merge_gifs(args)

    # ------------------------------------------------------------------------ #
    # Might be easier just to hack up if we can paste file names, see method.  #
    # Example: python examples/misc_utils.py --env MixedMediaRetrieval
    # ------------------------------------------------------------------------ #
    data_tune_substeps = {
        'name': 'tune_substeps',
        'files': [
            'MixedMediaRetrieval_ss_01_itrs_04_invdt_100_render_particle.gif',
            'MixedMediaRetrieval_ss_02_itrs_04_invdt_100_render_particle.gif',
            'MixedMediaRetrieval_ss_03_itrs_04_invdt_100_render_particle.gif',
            'MixedMediaRetrieval_ss_05_itrs_04_invdt_100_render_particle.gif',
            'MixedMediaRetrieval_ss_08_itrs_04_invdt_100_render_particle.gif',
            'MixedMediaRetrieval_ss_10_itrs_04_invdt_100_render_particle.gif',
            'MixedMediaRetrieval_ss_20_itrs_04_invdt_100_render_particle.gif',
        ]
    }
    data_tune_iterations = {
        'name': 'tune_iterations',
        'files': [
            'MixedMediaRetrieval_ss_02_itrs_01_invdt_100_render_particle.gif',
            'MixedMediaRetrieval_ss_02_itrs_02_invdt_100_render_particle.gif',
            'MixedMediaRetrieval_ss_02_itrs_04_invdt_100_render_particle.gif',
            'MixedMediaRetrieval_ss_02_itrs_06_invdt_100_render_particle.gif',
            'MixedMediaRetrieval_ss_02_itrs_08_invdt_100_render_particle.gif',
            'MixedMediaRetrieval_ss_02_itrs_10_invdt_100_render_particle.gif',
            'MixedMediaRetrieval_ss_02_itrs_20_invdt_100_render_particle.gif',
        ]
    }
    data_tune_substeps_fluid_1_sphere = {
        'name': 'tune_substeps_fluid_1_sphere',
        'files': [
            'MixedMediaRetrieval_ss_01_itrs_04_invdt_100_render_fluid.gif',
            'MixedMediaRetrieval_ss_02_itrs_04_invdt_100_render_fluid.gif',
            'MixedMediaRetrieval_ss_03_itrs_04_invdt_100_render_fluid.gif',
            'MixedMediaRetrieval_ss_05_itrs_04_invdt_100_render_fluid.gif',
            'MixedMediaRetrieval_ss_08_itrs_04_invdt_100_render_fluid.gif',
            'MixedMediaRetrieval_ss_10_itrs_04_invdt_100_render_fluid.gif',
            'MixedMediaRetrieval_ss_20_itrs_04_invdt_100_render_fluid.gif',
        ]
    }
    data_tune_iterations_fluid_1_sphere = {
        'name': 'tune_iterations_fluid_1_sphere',
        'files': [
            'MixedMediaRetrieval_ss_02_itrs_01_invdt_100_render_fluid.gif',
            'MixedMediaRetrieval_ss_02_itrs_02_invdt_100_render_fluid.gif',
            'MixedMediaRetrieval_ss_02_itrs_04_invdt_100_render_fluid.gif',
            'MixedMediaRetrieval_ss_02_itrs_06_invdt_100_render_fluid.gif',
            'MixedMediaRetrieval_ss_02_itrs_08_invdt_100_render_fluid.gif',
            'MixedMediaRetrieval_ss_02_itrs_10_invdt_100_render_fluid.gif',
            'MixedMediaRetrieval_ss_02_itrs_20_invdt_100_render_fluid.gif',
        ]
    }
    data_tune_substeps_liquid = {
        'name': 'tune_substeps_liquid_particle',
        'files': [
            'MixedMediaRetrieval_ss_01_itrs_04_invdt_100_render_particle.gif',
            'MixedMediaRetrieval_ss_02_itrs_04_invdt_100_render_particle.gif',
            'MixedMediaRetrieval_ss_03_itrs_04_invdt_100_render_particle.gif',
            'MixedMediaRetrieval_ss_05_itrs_04_invdt_100_render_particle.gif',
            'MixedMediaRetrieval_ss_08_itrs_04_invdt_100_render_particle.gif',
            'MixedMediaRetrieval_ss_10_itrs_04_invdt_100_render_particle.gif',
            'MixedMediaRetrieval_ss_20_itrs_04_invdt_100_render_particle.gif',
        ]
    }
    data_tune_collision_distance = {
        'name': 'tune_collision_distance',
        'files': [
            'MixedMediaRetrieval_collisionDistance_0.0003.gif',
            'MixedMediaRetrieval_collisionDistance_0.0010.gif',
            'MixedMediaRetrieval_collisionDistance_0.0033.gif',
            'MixedMediaRetrieval_collisionDistance_0.0100.gif',
        ]
    }
    data_tune_shape_collision_distance = {
        'name': 'tune_shape_collision_distance',
        'files': [
            'MixedMediaRetrieval_ss_02_itrs_04_invdt_100_render_particle_shapecollision_0.0000.gif',
            'MixedMediaRetrieval_ss_02_itrs_04_invdt_100_render_particle_shapecollision_0.0010.gif',
            'MixedMediaRetrieval_ss_02_itrs_04_invdt_100_render_particle_shapecollision_0.0100.gif',
            'MixedMediaRetrieval_ss_02_itrs_04_invdt_100_render_particle_shapecollision_0.1000.gif',
        ]
    }
    data_tune_item_invmass = {
        'name': 'tune_item_invmass',
        'files': [
            'MixedMediaRetrieval_ss_02_itrs_04_invmass_0.1_render_particle.gif',
            'MixedMediaRetrieval_ss_02_itrs_04_invmass_0.5_render_particle.gif',
            'MixedMediaRetrieval_ss_02_itrs_04_invmass_0.75_render_particle.gif',
            'MixedMediaRetrieval_ss_02_itrs_04_invmass_1.0_render_particle.gif',
            'MixedMediaRetrieval_ss_02_itrs_04_invmass_2.0_render_particle.gif',
        ]
    }
    data_tune_mm_demo = {
        'name': 'tune_mm_demo',
        'files': [
            'MixedMediaRetrieval_demo_01.gif',
            'MixedMediaRetrieval_demo_02.gif',
            'MixedMediaRetrieval_demo_03.gif',
            'MixedMediaRetrieval_demo_04.gif',
        ]
    }

    # Prepend 'data/' to the file names.
    def adjust_files(config):
        new_files = [join('data',x) for x in config['files']]
        config['files'] = new_files
        return config

    #merge_gifs_list(args, data=adjust_files(data_tune_substeps))
    #merge_gifs_list(args, data=adjust_files(data_tune_iterations))
    #merge_gifs_list(args, data=adjust_files(data_tune_substeps_fluid_1_sphere))
    #merge_gifs_list(args, data=adjust_files(data_tune_iterations_fluid_1_sphere))
    #merge_gifs_list(args, data=adjust_files(data_tune_substeps_liquid))
    #merge_gifs_list(args, data=adjust_files(data_tune_collision_distance))
    #merge_gifs_list(args, data=adjust_files(data_tune_shape_collision_distance))
    #merge_gifs_list(args, data=adjust_files(data_tune_item_invmass))
    merge_gifs_list(args, data=adjust_files(data_tune_mm_demo))