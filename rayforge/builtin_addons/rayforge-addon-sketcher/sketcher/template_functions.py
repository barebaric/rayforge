"""Compatibility shim for the core template-function registry.

The registry now lives in :mod:`rayforge.core.template_functions` so
barcode content and sketch text boxes share it without a dependency on
the sketcher addon.  This module keeps the historical import path
working for scripts and addons.
"""

from rayforge.core.template_functions import (
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
