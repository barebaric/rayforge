"""Settings groups (backward-compat re-exports).

The groups moved to
``rayforge/ui_gtk/doceditor/post_processor/groups/``; this module
re-exports them for existing importers.
"""

from ...post_processor.groups import (
    ExpanderHost,
    PlaceholderSettingsGroup,
    TransformerSettingsGroup,
)

__all__ = [
    "ExpanderHost",
    "PlaceholderSettingsGroup",
    "TransformerSettingsGroup",
]
