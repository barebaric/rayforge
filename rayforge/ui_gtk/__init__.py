__all__ = [
    "WorkSurface",
    "DotElement",
]


def __getattr__(name):
    if name == "WorkSurface":
        from .canvas2d.surface import WorkSurface

        return WorkSurface
    if name == "DotElement":
        from .canvas2d.elements.dot import DotElement

        return DotElement
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
