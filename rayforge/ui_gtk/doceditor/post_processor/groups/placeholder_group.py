"""Error display for a missing transformer widget."""

from gettext import gettext as _
from typing import TYPE_CHECKING

from gi.repository import Adw

from .....pipeline.transformer.base import OpsTransformer
from .transformer_group import (
    ExpanderHost,
    TransformerSettingsGroup,
)

if TYPE_CHECKING:
    from rayforge.core.step import Step


class PlaceholderSettingsGroup(TransformerSettingsGroup):
    """
    Error display for missing transformer widget.

    This group is shown when a step's transformer type is not available.
    """

    def __init__(
        self,
        title: str,
        transformer: OpsTransformer,
        page: ExpanderHost,
        *,
        step: "Step | None" = None,
        **kwargs,
    ):
        super().__init__(title, transformer, page, step=step, **kwargs)

        transformer_type = type(transformer).__name__

        error_row = Adw.ActionRow(
            title=_("This feature is not available."),
            subtitle=_(
                "The required component '{}' could not be found. "
                "The document can still be saved."
            ).format(transformer_type),
        )
        error_row.add_css_class("error")
        self.add(error_row)
