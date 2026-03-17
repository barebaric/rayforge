from typing import List, Tuple, Union

Point = Tuple[float, float]
Point3D = Tuple[float, float, float]
Point2DOr3D = Union[Point, Point3D]
Rect = Tuple[float, float, float, float]
Polygon = List[Point]
Edge = Tuple[Point, Point]
IntPoint = Tuple[int, int]
IntPolygon = List[IntPoint]
