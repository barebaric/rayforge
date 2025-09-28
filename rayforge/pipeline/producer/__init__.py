# flake8: noqa:F401
import inspect
from .base import OpsProducer, PipelineArtifact
from .depth import DepthEngraver
from .edge import EdgeTracer
from .frame import FrameProducer
from .rasterize import Rasterizer
from .shrinkwrap import ShrinkWrapProducer

producer_by_name = dict(
    [
        (name, obj)
        for name, obj in locals().items()
        if inspect.isclass(obj)
        and issubclass(obj, OpsProducer)
        and not inspect.isabstract(obj)
    ]
)

__all__ = [
    "OpsProducer",
    "DepthEngraver",
    "EdgeTracer",
    "FrameProducer",
    "Rasterizer",
    "ShrinkWrapProducer",
    "producer_by_name",
]
