import inspect
from .colorfilter import KeepColor
from .grayscale import ToGrayscale
from .modifier import Modifier
from .transparency import MakeTransparent

modifier_by_name = dict(
    (name, obj) for name, obj in locals().items()
    if inspect.isclass(obj) and
    issubclass(obj, Modifier) and
    not inspect.isabstract(obj)
)

__all__ = [
    "Modifier",
    "KeepColor",
    "ToGrayscale",
    "MakeTransparent",
    "modifier_by_name",
]
