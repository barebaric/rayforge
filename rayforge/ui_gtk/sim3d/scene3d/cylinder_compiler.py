"""
Pure-numpy cylinder mesh generation for texture mapping.

Generates vertex arrays that map a texture onto a cylinder surface
by transforming [0,1] texture coordinates through a grid matrix into
local cylinder space, then wrapping into Z/Y planes.

The cylinder always runs along X in the output vertex data.
"""

import numpy as np

GRID_S = 8
GRID_T = 64


def generate_cylinder_vertices(
    grid_matrix: np.ndarray,
    diameter: float,
    grid_s: int = GRID_S,
    grid_t: int = GRID_T,
) -> np.ndarray:
    radius = diameter / 2.0

    i_vals = np.arange(grid_s, dtype=np.float32)
    j_vals = np.arange(grid_t, dtype=np.float32)
    lx0 = i_vals / grid_s
    lx1 = (i_vals + 1) / grid_s
    ly0 = j_vals / grid_t
    ly1 = (j_vals + 1) / grid_t

    lx_grid, ly_grid = np.meshgrid(lx0, ly1, indexing="ij")
    lx1_grid, _ = np.meshgrid(lx1, ly1, indexing="ij")
    _, ly0_grid = np.meshgrid(lx0, ly0, indexing="ij")

    def _transform_points(lx, ly):
        shape = lx.shape
        ones = np.ones(shape, dtype=np.float32)
        pts = np.stack(
            [
                lx.ravel(),
                ly.ravel(),
                np.zeros(lx.size, dtype=np.float32),
                ones.ravel(),
            ],
            axis=-1,
        )
        p_cyl = pts @ grid_matrix.T
        theta = np.radians(p_cyl[:, 1])
        col_cyl = p_cyl[:, 0].reshape(shape)
        col_sin = (radius * np.sin(theta)).reshape(shape)
        col_cos = (radius * np.cos(theta)).reshape(shape)
        return col_cyl, col_sin, col_cos

    x00, y00, z00 = _transform_points(lx_grid, ly0_grid)
    x10, y10, z10 = _transform_points(lx1_grid, ly0_grid)
    x01, y01, z01 = _transform_points(lx_grid, ly_grid)
    x11, y11, z11 = _transform_points(lx1_grid, ly_grid)

    s0 = lx_grid
    s1 = lx1_grid
    t0 = 1.0 - ly0_grid
    t1 = 1.0 - ly_grid

    s0_f = s0.flatten()
    s1_f = s1.flatten()
    t0_f = t0.flatten()
    t1_f = t1.flatten()
    x00_f, y00_f, z00_f = x00.flatten(), y00.flatten(), z00.flatten()
    x10_f, y10_f, z10_f = x10.flatten(), y10.flatten(), z10.flatten()
    x01_f, y01_f, z01_f = x01.flatten(), y01.flatten(), z01.flatten()
    x11_f, y11_f, z11_f = x11.flatten(), y11.flatten(), z11.flatten()

    n = s0_f.size
    vertices = np.empty(n * 30, dtype=np.float32)

    vertices[0::30] = x00_f
    vertices[1::30] = y00_f
    vertices[2::30] = z00_f
    vertices[3::30] = s0_f
    vertices[4::30] = t0_f
    vertices[5::30] = x10_f
    vertices[6::30] = y10_f
    vertices[7::30] = z10_f
    vertices[8::30] = s1_f
    vertices[9::30] = t0_f
    vertices[10::30] = x01_f
    vertices[11::30] = y01_f
    vertices[12::30] = z01_f
    vertices[13::30] = s0_f
    vertices[14::30] = t1_f

    vertices[15::30] = x10_f
    vertices[16::30] = y10_f
    vertices[17::30] = z10_f
    vertices[18::30] = s1_f
    vertices[19::30] = t0_f
    vertices[20::30] = x11_f
    vertices[21::30] = y11_f
    vertices[22::30] = z11_f
    vertices[23::30] = s1_f
    vertices[24::30] = t1_f
    vertices[25::30] = x01_f
    vertices[26::30] = y01_f
    vertices[27::30] = z01_f
    vertices[28::30] = s0_f
    vertices[29::30] = t1_f

    return vertices
