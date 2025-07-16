# flake8: noqa:F401
import inspect
from .producer import OpsProducer
from .outline import OutlineTracer
from .edge import EdgeTracer
from .rasterize import Rasterizer

producer_by_name = dict(
    [(name, obj) for name, obj in list(locals().items())
     if not name.startswith('_') or inspect.ismodule(obj)]
)
