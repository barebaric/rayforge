"""Compatibility shim for the historical ``sketcher.core`` import path."""

from sketcher.template_functions import (
    TemplateFunction,
    get_template_functions,
    register_template_function,
    unregister_template_function,
)

__all__ = [
    "TemplateFunction",
    "get_template_functions",
    "register_template_function",
    "unregister_template_function",
]
