# flake8: noqa:F401
import inspect
from .base import OpsTransformer, ExecutionPhase
from .crop_transformer import CropTransformer
from .multipass_transformer import MultiPassTransformer
from .optimize_transformer import Optimize
from .overscan_transformer import OverscanTransformer
from .smooth_transformer import Smooth
from .tabs_transformer import TabOpsTransformer

transformer_by_name = dict(
    (name, obj)
    for name, obj in locals().items()
    if inspect.isclass(obj)
    and issubclass(obj, OpsTransformer)
    and not inspect.isabstract(obj)
)

__all__ = [
    "CropTransformer",
    "OpsTransformer",
    "ExecutionPhase",
    "MultiPassTransformer",
    "Optimize",
    "OverscanTransformer",
    "Smooth",
    "TabOpsTransformer",
    "transformer_by_name",
]
