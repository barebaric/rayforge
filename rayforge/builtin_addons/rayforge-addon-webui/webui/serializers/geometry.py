"""
Geometry serialization for the web UI.

Workpiece outlines are flattened from ``WorkPiece.get_world_geometry()``
(WORLD space, mm, Y-up) into polyline lists; job ops are flattened into
attributed polyline segments for the toolpath preview.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from rayforge.core.stock import StockItem
    from rayforge.core.workpiece import WorkPiece

DEFAULT_TOLERANCE = 0.05


def _rev(item: "WorkPiece | StockItem") -> int:
    return int(item.geometry_revision) + int(item.transform_revision)


def _flatten(geometry, tolerance: float) -> list[list[float]]:
    """Geometry contours as flat ``[x, y, x, y, ...]`` polylines."""
    linearized = geometry.linearize(
        tolerance if tolerance > 0 else DEFAULT_TOLERANCE
    )
    if tolerance > 0:
        linearized = linearized.simplify(tolerance)
    polylines = []
    for segment in linearized.segments():
        flat: list[float] = []
        for point in segment:
            flat.append(round(float(point[0]), 4))
            flat.append(round(float(point[1]), 4))
        polylines.append(flat)
    return polylines


def workpiece_geometry(
    workpiece: "WorkPiece", tolerance: float = 0.0
) -> dict[str, Any]:
    """
    The workpiece's world-space vector geometry as polylines.

    Returns ``{"rev", "polylines", "outline"}``; ``polylines`` entries
    are flat ``[x, y, ...]`` lists in WORLD mm and ``outline`` is
    ``[x, y, w, h]``. Geometry-free workpieces yield empty polylines.
    """
    return _item_geometry(workpiece, workpiece.get_world_geometry(), tolerance)


def stock_geometry(
    stock: "StockItem", tolerance: float = 0.0
) -> dict[str, Any]:
    """
    The stock item's pristine world-space vector geometry as polylines,
    in the same shape as :func:`workpiece_geometry`.
    """
    return _item_geometry(stock, stock.get_world_geometry(), tolerance)


def _item_geometry(
    item: "WorkPiece | StockItem",
    geometry,
    tolerance: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "rev": _rev(item),
        "polylines": [],
        "outline": None,
    }
    if geometry is None or geometry.is_empty():
        return result
    x_min, y_min, x_max, y_max = geometry.rect()
    result["outline"] = [
        round(float(x_min), 4),
        round(float(y_min), 4),
        round(float(x_max - x_min), 4),
        round(float(y_max - y_min), 4),
    ]
    result["polylines"] = _flatten(geometry, tolerance)
    return result


def _scan_power(command: dict[str, Any]) -> float | None:
    values = command.get("power_values")
    if not values:
        return None
    return round(sum(values) / (len(values) * 255.0), 4)


def ops_segments(ops, tolerance: float = 0.0) -> dict[str, Any]:
    """
    Flatten job ops into attributed polyline segments.

    Curves are linearized first, so every segment is a plain polyline.
    Each segment carries the active power (0-1) and feed rate at the
    time it was emitted, plus a ``cutting`` flag (False for travel
    moves, True for LINE_TO and SCAN_LINE segments).
    """
    flat = ops.copy()
    flat.linearize_all()
    commands = flat.to_dict()["commands"]

    segments: list[dict[str, Any]] = []
    power: float | None = None
    rate: float | None = None
    current: dict[str, Any] | None = None
    position = (0.0, 0.0)

    def new_segment(x: float, y: float, cutting: bool) -> dict[str, Any]:
        segment = {
            "points": [[round(x, 4), round(y, 4)]],
            "power": power,
            "rate": rate,
            "cutting": cutting,
        }
        segments.append(segment)
        return segment

    for command in commands:
        cmd_type = command["type"]
        if cmd_type == "SET_POWER":
            power = command.get("power")
            if current is not None:
                current["power"] = power
        elif cmd_type == "SET_FEED_RATE":
            rate = command.get("speed")
            if current is not None:
                current["rate"] = rate
        elif cmd_type in ("MOVE_TO", "LINE_TO", "SCAN_LINE"):
            end_x, end_y, _ = command["end"]
            cutting = cmd_type != "MOVE_TO"
            if not cutting or current is None or not current["cutting"]:
                if cutting:
                    current = new_segment(
                        position[0], position[1], cutting=True
                    )
                else:
                    current = new_segment(end_x, end_y, cutting=False)
            current["points"].append([round(end_x, 4), round(end_y, 4)])
            position = (end_x, end_y)
            if cmd_type == "SCAN_LINE":
                scan_power = _scan_power(command)
                if scan_power is not None:
                    current["power"] = scan_power

    return {
        "segments": segments,
        "estimated_seconds": ops.estimate_time(),
        "distance_mm": ops.distance(),
    }
