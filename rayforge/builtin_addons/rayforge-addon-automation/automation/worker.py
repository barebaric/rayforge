"""
Backend entry point for the automation addon.

Registers steps with the main application.
"""

from rayforge.core.hooks import hookimpl

from .steps import CommandStep

ADDON_NAME = "automation"


@hookimpl
def register_steps(step_registry):
    """Register steps with the step registry."""
    step_registry.register(CommandStep, addon_name=ADDON_NAME)
