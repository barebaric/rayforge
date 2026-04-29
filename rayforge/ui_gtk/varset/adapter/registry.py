import logging
from gettext import gettext as _
from typing import Dict, Optional, Tuple, Type

from gi.repository import Adw

from ....core.varset import Var
from .base import RowAdapter, _ADAPTER_REGISTRY, escape_title
from .entry import EntryAdapter
from .spin_row import SpinRowAdapter
from .switch import SwitchAdapter

# Trigger @register_adapter decorators. Every adapter module must be
# imported here so that its class registers itself in _ADAPTER_REGISTRY.
from .appkey import AppKeyAdapter
from .combo import BaudRateAdapter, ComboAdapter, SerialPortAdapter
from .entry import HostnameAdapter
from .oauth import OAuthFlowAdapter
from .slider import SliderAdapter
from .speed import SpeedRowAdapter
from .textarea import TextAreaAdapter

_ALL_ADAPTERS = (
    AppKeyAdapter,
    BaudRateAdapter,
    ComboAdapter,
    SerialPortAdapter,
    HostnameAdapter,
    OAuthFlowAdapter,
    SliderAdapter,
    SpeedRowAdapter,
    TextAreaAdapter,
)

logger = logging.getLogger(__name__)

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

    Walks the Var's MRO to find the most specific registered adapter.
    Falls back to var.var_type if no adapter is registered for any
    class in the hierarchy.
    """
    adapter_cls: Optional[Type[RowAdapter]] = None

    for cls in type(var).__mro__:
        if cls in _ADAPTER_REGISTRY:
            adapter_cls = _ADAPTER_REGISTRY[cls]
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
