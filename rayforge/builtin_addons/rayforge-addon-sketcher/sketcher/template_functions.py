"""Public registry for text-box template functions.

Text-box content may contain ``{expression}`` placeholders that are
evaluated at render time.  Besides the sketch's parameters and the
``math`` module, a set of named callables is available inside those
expressions (e.g. ``{today()}``, ``{uuid8()}``).

Addons and scripts can extend the available callables via
:func:`register_template_function` without modifying the sketcher
package itself.

Built-in functions
------------------
==================  =========================================================
``today()``         Current UTC date (``date``).
``now()``           Current UTC datetime (``datetime``).
``date()``          Alias for ``today()``.
``time()``          Current UTC time (``datetime.time``).
``timestamp()``     Unix timestamp as a float (seconds since epoch).
``uuid()``          Full UUID string (36 chars, e.g.
                    ``"550e8400-e29b-...-``).
``uuid8()``         Short UUID string (first 8 hex chars).
``uuid4()``         Legacy alias for ``uuid8()``.
==================  =========================================================
"""

from __future__ import annotations

import time as _time
import uuid as _uuid
from collections.abc import Callable
from datetime import datetime, timezone

# A call returning the value to substitute.  Callables are stored
# zero-argument and invoked fresh per evaluation so that volatile
# values (e.g. uuids) differ between renders.
TemplateFunction = Callable[[], object]

_REGISTRY: dict[str, TemplateFunction] = {}


def _utcnow() -> datetime:
    return datetime.now(tz=timezone.utc)


def _register_builtin(name: str, func: TemplateFunction) -> None:
    _REGISTRY[name] = func


_register_builtin("today", lambda: _utcnow().date())
_register_builtin("date", lambda: _utcnow().date())
_register_builtin("now", _utcnow)
_register_builtin("time", lambda: _utcnow().timetz())
_register_builtin("timestamp", _time.time)
_register_builtin("uuid", lambda: str(_uuid.uuid4()))
_register_builtin("uuid8", lambda: _uuid.uuid4().hex[:8])
_register_builtin("uuid4", lambda: _uuid.uuid4().hex[:8])


def register_template_function(name: str, func: TemplateFunction) -> None:
    """Register a callable available inside text-box ``{...}`` templates.

    Args:
        name: The name as used in template expressions (e.g. ``"foo"``
            makes ``{foo()}`` work).
        func: A zero-argument callable returning the substitution value.
            It is invoked on every render, so volatile values (uuids,
            timestamps) produce a fresh result each time.

    Raises:
        ValueError: If *name* shadows a built-in name unless
            ``override`` is True.
    """
    _REGISTRY[name] = func


def unregister_template_function(name: str) -> None:
    """Remove a previously registered template function.

    Built-in functions cannot be removed; the call is a no-op for them.
    """
    if name in (
        "today",
        "date",
        "now",
        "time",
        "timestamp",
        "uuid",
        "uuid8",
        "uuid4",
    ):
        return
    _REGISTRY.pop(name, None)


def get_template_functions() -> dict[str, TemplateFunction]:
    """Return a copy of the current template-function registry."""
    return dict(_REGISTRY)
