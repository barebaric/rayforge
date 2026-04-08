from typing import List, NamedTuple, Tuple, Union


class Rect3D(NamedTuple):
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float


Point = Tuple[float, float]
Point3D = Tuple[float, float, float]
Point2DOr3D = Union[Point, Point3D]
CubicBezier = Tuple[Point, Point, Point, Point]
Rect = Tuple[float, float, float, float]
Polygon = List[Point]
Polygon3D = List[Point3D]
Edge = Tuple[Point, Point]
IntPoint = Tuple[int, int]
IntPolygon = List[IntPoint]
