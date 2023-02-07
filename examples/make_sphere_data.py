import torch
import pickle
import argparse

def sample_sphere_points(n_points, radius=1.0):
    points = torch.randn(n_points, 3)
    points /= torch.linalg.norm(points, axis=1).view(-1, 1)
    points *= radius
    return points

def random_so3(num):
    return torch.linalg.qr(torch.randn(num, 3, 3)).Q

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('-n', type=int, default=10000, help="number of training exmaples")
    ap.add_argument('-d', type=float, default=0.2, help="distance between spheres")
    ap.add_argument('-r', type=float, default=0.025, help="radius of spheres")
    ap.add_argument('--output_dir', type=str, help="output directory")
    args = ap.parse_args()

    # Generate two spheres, separated by distance d on the x-axis
    left_sphere = sample_sphere_points(400, radius=args.r)
    left_sphere[:, 0] -= args.d / 2
    right_sphere = sample_sphere_points(400, radius=args.r)
    right_sphere[:, 0] += args.d / 2

    all_points = torch.cat([left_sphere, right_sphere], dim=0) # (800, 3)

    # Generate sphere flow (pointing towards each other)
    # NOTE (eddie): One other way we could do it is by having one hot vectors that indicate a pointing sphere and a pointed sphere, but doing it this way instead for ease of code
    initial_flow = torch.zeros_like(all_points)
    initial_flow[:400, 0] = 1.0
    initial_flow[400:, 0] = -1.0

    # Generate random rotation matrices
    rotations = random_so3(args.n)

    # Generate dataset
    dataset_pts = torch.einsum('bij,kj->bki', rotations, all_points) # (args.n, 800, 3)
    dataset_flow = torch.einsum('bij,kj->bki', rotations, initial_flow) # (args.n, 800, 3)

    # Save dataset
    with open(f'{args.output_dir}/dummy_{args.d}d_{args.r}r.pkl', 'wb') as f:
        pickle.dump({'points': dataset_pts.numpy(), 'flow': dataset_flow.numpy()}, f)
