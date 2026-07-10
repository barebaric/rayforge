import numpy as np


def euler_rotation_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """
    Build a 3x3 rotation matrix from Euler angles in degrees.

    The rotation order is X -> Y -> Z (extrinsic Tait-Bryan angles).
    """
    ax, ay, az = np.radians(rx), np.radians(ry), np.radians(rz)
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    return np.array(
        [
            [cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
            [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
            [-sy, sx * cy, cx * cy],
        ],
        dtype=np.float64,
    )
