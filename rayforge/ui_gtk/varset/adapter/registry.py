import logging
from gettext import gettext as _
from typing import Dict, Optional, Tuple, Type

from gi.repository import Adw

from ....core.varset import (
    BaudrateVar,
    BoolVar,
    ChoiceVar,
    FloatVar,
    HostnameVar,
    IntVar,
    PortVar,
    SerialPortVar,
    SliderFloatVar,
    SpeedVar,
    TextAreaVar,
    Var,
)
from .base import RowAdapter, escape_title
from .combo import BaudRateAdapter, ComboAdapter, SerialPortAdapter
from .entry import EntryAdapter, HostnameAdapter
from .slider import SliderAdapter
from .speed import SpeedRowAdapter
from .spin_row import SpinRowAdapter
from .switch import SwitchAdapter
from .textarea import TextAreaAdapter

logger = logging.getLogger(__name__)

_ADAPTER_MAP: Dict[Type[Var], Type[RowAdapter]] = {
    TextAreaVar: TextAreaAdapter,
    SliderFloatVar: SliderAdapter,
    SpeedVar: SpeedRowAdapter,
    BaudrateVar: BaudRateAdapter,
    PortVar: SpinRowAdapter,
    SerialPortVar: SerialPortAdapter,
    HostnameVar: HostnameAdapter,
    ChoiceVar: ComboAdapter,
    FloatVar: SpinRowAdapter,
    IntVar: SpinRowAdapter,
    BoolVar: SwitchAdapter,
}

_FALLBACK_MAP: Dict[type, Type[RowAdapter]] = {
    int: SpinRowAdapter,
    float: SpinRowAdapter,
    bool: SwitchAdapter,
    str: EntryAdapter,
}


def create_row_for_var(
    var: Var, target_property: str = "value"
) -> Tuple[Adw.PreferencesRow, Optional[RowAdapter]]:
    """
    Creates a PreferencesRow and RowAdapter for the given Var.

    Dispatches to the adapter class registered in _ADAPTER_MAP,
    falling back to var.var_type if no specific adapter is found.
    """
    adapter_cls: Optional[Type[RowAdapter]] = None

    for var_class, cls in _ADAPTER_MAP.items():
        if isinstance(var, var_class):
            adapter_cls = cls
            break

    if adapter_cls is None:
        adapter_cls = _FALLBACK_MAP.get(var.var_type)

    if adapter_cls is not None:
        return adapter_cls.create(var, target_property)

    logger.warning(
        "No UI widget defined for Var with key '%s' and type %s",
        var.key,
        type(var),
    )
    row = Adw.ActionRow(
        title=escape_title(var.label),
        subtitle=_("Unsupported type: {t}").format(t=type(var).__name__),
        sensitive=False,
    )
    return row, None
