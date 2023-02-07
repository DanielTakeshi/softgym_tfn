import numpy as np
from pyquaternion import Quaternion


def sample_sphere_points(n_points, radius):
    """Use for ground truth sphere point clouds."""
    points = np.random.randn(n_points, 3)
    norm = np.linalg.norm(points, axis=1)[:,None] + 1e-7
    points = (points / norm) * radius
    return points


def rotation_2d_around_center(pt, center, theta):
    """
    2d rotation on 3d vectors by ignoring y factor
    :param pt:
    :param center:
    :return:
    """
    pt = pt.copy()
    pt = pt - center
    x, y, z = pt
    new_pt = np.array([np.cos(theta) * x - np.sin(theta) * z, y, np.sin(theta) * x + np.cos(theta) * z]) + center
    return new_pt


def extend_along_center(pt, center, add_dist, min_dist, max_dist):
    pt = pt.copy()
    curr_dist = np.linalg.norm(pt - center)
    pt = pt - center
    new_dist = min(max(min_dist, curr_dist + add_dist), max_dist)
    pt = pt * (new_dist / curr_dist)
    pt = pt + center
    return pt


def vectorized_range(start, end):
    """Return an array of NxD, iterating from the start to the end."""
    N = int(np.max(end - start)) + 1
    idxes = np.floor(np.arange(N) * (end - start)[:, None] / N + start[:, None]).astype('int')
    return idxes


def vectorized_meshgrid(vec_x, vec_y):
    """vec_x in NxK, vec_y in NxD. Return xx in Nx(KxD) and yy in Nx(DxK)"""
    N, K, D = vec_x.shape[0], vec_x.shape[1], vec_y.shape[1]
    vec_x = np.tile(vec_x[:, None, :], [1, D, 1]).reshape(N, -1)
    vec_y = np.tile(vec_y[:, :, None], [1, 1, K]).reshape(N, -1)
    return vec_x, vec_y


def rotate_rigid_object(center, axis, angle, pos=None, relative=None):
    """Rotate a rigid object (e.g. shape in flex).

    NOTE(daniel): misleading name we can use shapes, and are not restricted to
    PyFlexRobotics Rigid Bodies. This is used in PourWater to rotate the box, and
    the reason is the box consists of 5 walls, so if we just naively rotate all
    of them, they will rotate about their centers. But, the object should have a
    common center of rotation, so if a wall is not centered at the origin, its
    position must be adjusted!

    We might also need this in cases when we rotate an item (e.g., a tool ladle)
    and want to adjust where PyFlex 'thinks' its origin should be.

    pos: np.ndarray 3x1, [x, y, z] coordinate of the object.
    relative: relative coordinate of the object to center.
    center: rotation center.
    axis: rotation axis.
    angle: rotation angle in radius.
    TODO: add rotation of coordinates (NOTE(daniel): ?)

    Returns
    -------
    The updated _position_ of the shape object.
    """
    if relative is None:
        relative = pos - center
    quat = Quaternion(axis=axis, angle=angle)
    after_rotate = quat.rotate(relative)
    return after_rotate + center


def quatFromAxisAngle(axis, angle):
    """Given rotation axis and angle, return quaternion that represents such rotation.

    NOTE(daniel): I assumed there might be some numerical precision issues:
    https://math.stackexchange.com/questions/291110/
    but I doubt we will ever run into this?
    Returns in (x,y,z,w) form. Note that this differs from Quaternion() which will
    return in (w,x,y,z) form.

    By the way, if we are doing full 6 DoF grasping, then we will want to compute
    the appropriate `axis` beforehand, right? In SoftGym we tend to just need one
    'euler angle' (if we even use rotations at all) so the `axis` is typically
    something simple like [0,0,-1].
    """
    axis /= np.linalg.norm(axis)
    half = angle * 0.5
    w = np.cos(half)
    sin_theta_over_two = np.sin(half)
    axis *= sin_theta_over_two
    quat = np.array([axis[0], axis[1], axis[2], w])
    return quat


def print_dict(d, indent=0):
    """Pretty print a dict."""
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            print_dict(value, indent+1)
        elif isinstance(value, float):
            vstr = f'{value:0.3f}'
            print('\t' * (indent+1) + vstr)
        else:
            print('\t' * (indent+1) + str(value))