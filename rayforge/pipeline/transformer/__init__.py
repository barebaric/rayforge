# flake8: noqa:F401
import inspect
from .base import OpsTransformer, ExecutionPhase

transformer_by_name = dict(
    (name, obj)
    for name, obj in locals().items()
    if inspect.isclass(obj)
    and issubclass(obj, OpsTransformer)
    and not inspect.isabstract(obj)
)

__all__ = [
    "OpsTransformer",
    "ExecutionPhase",
    "transformer_by_name",
]
