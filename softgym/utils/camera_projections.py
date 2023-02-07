"""
Use this script to handle stuff like camera-to-world code, etc.
Requires PyFlexRobotics and the use of a depth camera.

From Zixuan Huang et al., with extra documentation by Daniel Seita.
"""
import numpy as np


def intrinsic_from_fov(height, width, fov=90):
    """Basic Pinhole Camera Model.

    See derivations / explanations:
    https://en.wikipedia.org/wiki/Camera_resectioning
    https://szeliski.org/Book/ (Section 2.1.4)

    These references tend to include a skew coefficient (gamma) in the intrinsics
    matrix. Here, we set that value to 0 (which is typical) and I think that's because
    in simulation we can assume the sensor is perfectly aligned with the optical axis.

    Parameters
    ----------
    height, width: pixel values of sensor (camera).
    fov: field of view.

    Returns
    -------
    4x4 camera intrinsics matrix.
    """
    px, py = (width / 2, height / 2)
    hfov = fov / 360. * 2. * np.pi
    fx = width / (2. * np.tan(hfov / 2.))

    vfov = 2. * np.arctan(np.tan(hfov / 2) * height / width)
    fy = height / (2. * np.tan(vfov / 2.))

    return np.array([[fx, 0, px, 0.],
                     [0, fy, py, 0.],
                     [0, 0, 1., 0.],
                     [0., 0., 0., 1.]])


def get_rotation_matrix(angle, axis):
    """Get axis-angle representation of rotation matrices.

    Returns a 4x4 rotation matrix to be used with homogeneous coordinates.
    You can find this formula in computer vision and robotics textbooks, e.g.,
    Modern Robotics (Park and Lynch), Chapter 3 on rigid body transformations.
    Note: Modern Robotics actually transposes this rotation matrix, so it's not
    exactly here, but the transpose of a rotation matrix (its inverse in this
    special case) is also a rotation matrix -- though I'd like to double check.
    TODO(daniel)
    """
    axis = axis / np.linalg.norm(axis)
    s = np.sin(angle)
    c = np.cos(angle)

    m = np.zeros((4, 4))

    m[0][0] = axis[0] * axis[0] + (1.0 - axis[0] * axis[0]) * c
    m[0][1] = axis[0] * axis[1] * (1.0 - c) - axis[2] * s
    m[0][2] = axis[0] * axis[2] * (1.0 - c) + axis[1] * s
    m[0][3] = 0.0

    m[1][0] = axis[0] * axis[1] * (1.0 - c) + axis[2] * s
    m[1][1] = axis[1] * axis[1] + (1.0 - axis[1] * axis[1]) * c
    m[1][2] = axis[1] * axis[2] * (1.0 - c) - axis[0] * s
    m[1][3] = 0.0

    m[2][0] = axis[0] * axis[2] * (1.0 - c) - axis[1] * s
    m[2][1] = axis[1] * axis[2] * (1.0 - c) + axis[0] * s
    m[2][2] = axis[2] * axis[2] + (1.0 - axis[2] * axis[2]) * c
    m[2][3] = 0.0

    m[3][0] = 0.0
    m[3][1] = 0.0
    m[3][2] = 0.0
    m[3][3] = 1.0

    return m


def get_matrix_world_to_camera(camera_param):
    """Create matrix for image -> world transforms.

    Parameters
    ----------
    camera_param is a dictionary in the common format used in SoftGym:
        {'pos': cam_pos,
         'angle': cam_angle,
         'width': self.camera_width,
         'height': self.camera_height}
    """
    cam_x, cam_y, cam_z = camera_param['pos'][0], camera_param['pos'][1], \
                          camera_param['pos'][2]
    cam_x_angle, cam_y_angle, cam_z_angle = camera_param['angle'][0], \
                                            camera_param['angle'][1], \
                                            camera_param['angle'][2]

    # get rotation matrix: from world to camera
    matrix1 = get_rotation_matrix(- cam_x_angle, [0, 1, 0])
    matrix2 = get_rotation_matrix(- cam_y_angle - np.pi, [1, 0, 0])
    rotation_matrix = matrix2 @ matrix1

    # get translation matrix: from world to camera
    translation_matrix = np.eye(4)
    translation_matrix[0][3] = - cam_x
    translation_matrix[1][3] = - cam_y
    translation_matrix[2][3] = - cam_z

    return rotation_matrix @ translation_matrix


def get_world_coords(rgb, depth, matrix_world_to_camera):
    """From SoftAgent RPAD, like uv_to_world_pos but with full images as input.

    This should vectorize so I think it's faster.
    Modifying this to directly pass in the `matrix_world_to_camera`.
    """
    height, width, _ = rgb.shape
    K = intrinsic_from_fov(height, width, 45)

    # Apply back-projection: K_inv @ pixels * depth
    u0 = K[0, 2]
    v0 = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    x = np.linspace(0, width - 1, width).astype(np.float)
    y = np.linspace(0, height - 1, height).astype(np.float)
    u, v = np.meshgrid(x, y)
    one = np.ones((height, width, 1))
    x = (u - u0) * depth / fx
    y = (v - v0) * depth / fy
    z = depth
    cam_coords = np.dstack([x, y, z, one])

    # convert the camera coordinate back to the world coordinate
    # using the rotation and translation matrix
    cam_coords = cam_coords.reshape((-1, 4)).transpose()  # 4 x (height x width)
    world_coords = np.linalg.inv(matrix_world_to_camera) @ cam_coords  # 4 x (height x width)
    world_coords = world_coords.transpose().reshape((height, width, 4))

    return world_coords

def get_pointcloud(depth, matrix_world_to_camera):
    """
    Equivalent to get_world_coords, but ignores 0-depth points
    """
    height, width = depth.shape
    u, v = depth.nonzero()
    z = depth[u, v]
    K = intrinsic_from_fov(height, width, 45).astype(np.float32)
    x0 = K[0, 2]
    y0 = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]
    one = np.ones(u.shape, np.float32)

    x = (v - x0) * z / fx
    y = (u - y0) * z / fy
    cam_coords = np.stack([x, y, z, one], axis=1)
    cam2world = np.linalg.inv(matrix_world_to_camera).T
    world_coords = cam_coords @ cam2world
    return world_coords[:, :3].astype(np.float32)


def uv_to_world_pos(u, v, z, camera_params):
    """Transform from image coordinates and depth to world coordinates.

    Parameters
    ----------
    u, v: image coordinates
    z: depth value
    camera_params: dict with usual format from SoftGym.
    """
    height, width = camera_params['height'], camera_params['width']
    K = intrinsic_from_fov(height, width, 45)  # TODO(daniel) is 45 right?
    x0 = K[0, 2]
    y0 = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]
    matrix_world_to_camera = get_matrix_world_to_camera(camera_params)
    one = np.ones(u.shape)

    x = (v - x0) * z / fx
    y = (u - y0) * z / fy
    cam_coords = np.stack([x, y, z, one], axis=1)
    cam2world = np.linalg.inv(matrix_world_to_camera).T
    world_coords = cam_coords @ cam2world
    return world_coords


def world_to_uv(matrix_world_to_camera, world_coordinate, height=360, width=360):
    """Transform from world coordinates to image pixels.

    Same as the `project_to_image` method:
    https://github.com/Xingyu-Lin/softagent_rpad/blob/master/VCD/camera_utils.py

    Parameters
    ----------
    matrix_world_to_camera: should be from `get_matrix_world_to_camera()`.
    world_coordinate: np.array, shape (n x 3), specifying world coordinates, i.e.,
        the coordinate we can get from FleX.

    Returns
    -------
    (u,v): specifies (x,y) coords, `u` and `v` are each np.arrays, shape (n,).
        To use it directly with a numpy array such as img[uv], we might have to
        transpose it. Unfortunately I always get confused about the right way.
    """
    world_coordinate = np.concatenate(
        [world_coordinate, np.ones((len(world_coordinate), 1))], axis=1)  # n x 4
    camera_coordinate = matrix_world_to_camera @ world_coordinate.T  # 3 x n
    camera_coordinate = camera_coordinate.T  # n x 3
    K = intrinsic_from_fov(height, width, 45)  # TODO(daniel) is 45 right?

    u0 = K[0, 2]
    v0 = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    # Convert to ints because we want the pixel coordinates.
    x, y, depth = camera_coordinate[:, 0], camera_coordinate[:, 1], camera_coordinate[:, 2]
    u = (x * fx / depth + u0).astype("int")
    v = (y * fy / depth + v0).astype("int")
    return u, v