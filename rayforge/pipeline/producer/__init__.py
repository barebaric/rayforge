# flake8: noqa:F401
import inspect
from .base import OpsProducer
from .outline import OutlineTracer
from .edge import EdgeTracer
from .rasterize import Rasterizer

producer_by_name = dict([
    (name, obj) for name, obj in locals().items()
    if inspect.isclass(obj) and
    issubclass(obj, OpsProducer) and
    not inspect.isabstract(obj)
])

__all__ = [
    "OpsProducer",
    "OutlineTracer",
    "EdgeTracer",
    "Rasterizer",
    "producer_by_name",
]
