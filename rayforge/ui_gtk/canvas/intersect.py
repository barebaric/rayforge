from typing import Tuple, List
from gi.repository import Graphene


def obb_intersects_aabb(
    obb_corners: List[Tuple[float, float]], aabb: Graphene.Rect
) -> bool:
    """
    Checks if an Oriented Bounding Box (OBB) intersects with an Axis-Aligned
    Bounding Box (AABB) using the Separating Axis Theorem (SAT).

    An intersection occurs if the projections of the two shapes overlap on all
    potential separating axes. The axes to test are the normals of the edges
    of both shapes.
    """

    def project(polygon_corners, axis):
        """
        Projects a polygon onto an axis and returns the min/max projection.
        """
        min_p = float("inf")
        max_p = float("-inf")
        for p in polygon_corners:
            # Vector dot product
            projection = p[0] * axis[0] + p[1] * axis[1]
            min_p = min(min_p, projection)
            max_p = max(max_p, projection)
        return min_p, max_p

    aabb_corners = [
        (aabb.get_x(), aabb.get_y()),
        (aabb.get_x() + aabb.get_width(), aabb.get_y()),
        (aabb.get_x() + aabb.get_width(), aabb.get_y() + aabb.get_height()),
        (aabb.get_x(), aabb.get_y() + aabb.get_height()),
    ]

    # The axes to test are the unique normals of the edges.
    # For an AABB, the normals are the world axes.
    # For an OBB (rectangle), there are two unique normals.
    edge1 = (
        obb_corners[1][0] - obb_corners[0][0],
        obb_corners[1][1] - obb_corners[0][1],
    )
    normal1 = (-edge1[1], edge1[0])

    edge2 = (
        obb_corners[3][0] - obb_corners[0][0],
        obb_corners[3][1] - obb_corners[0][1],
    )
    normal2 = (-edge2[1], edge2[0])

    axes_to_test = [(1, 0), (0, 1), normal1, normal2]

    for axis in axes_to_test:
        # Ensure axis is not a zero vector (can happen with zero-sized
        # elements)
        if axis[0] == 0 and axis[1] == 0:
            continue

        min_p1, max_p1 = project(obb_corners, axis)
        min_p2, max_p2 = project(aabb_corners, axis)

        # Check for separation: if one projection doesn't overlap,
        # there's a separating axis.
        if max_p1 < min_p2 or max_p2 < min_p1:
            return False  # A separating axis was found

    # If no separating axis was found after checking all axes, the
    # polygons must be intersecting.
    return True
