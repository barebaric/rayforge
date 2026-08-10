"""Translation helpers for pipeline status messages.

raygeo reports batch progress with a machine-readable payload.  An
active node emits ``{key}`` (e.g. ``workpiece:{wp_uid}:{step_uid}``)
and, while an assembler or transformer is running,
``{key}\\t{detail}`` where ``detail`` is an assembler phase such as
``"contour: assemble"`` or a transformer spec name such as
``"overscan"``.  A node that just finished emits ``\\t{key}``.  This
module turns the payload into a human-readable, translatable status
string for the progress bar.  ``gettext`` ``_()`` is applied here, so
the templates are picked up by ``xgettext``.
"""

from collections.abc import Mapping
from gettext import gettext as _
from typing import TYPE_CHECKING

from ..core.step_registry import step_registry
from .intent_builder import parse_workpiece_key
from .transformer.registry import transformer_registry

if TYPE_CHECKING:
    from ..core.step import Step
    from ..core.workpiece import WorkPiece

#: Labels for stages owned by the pipeline itself (not provided by an
#: addon step or transformer).
_PIPELINE_LABELS = {
    "aggregate": lambda: _("Aggregate"),
}


def _activity_label(detail: str) -> str | None:
    """Translate a ``\\t`` detail suffix to a user-facing activity label.

    :param detail: Text after the ``\\t`` in a batch progress payload,
        e.g. ``"contour: assemble"`` or ``"overscan"``.
    :returns: A ``_()``-marked activity label, or ``None`` when the
        detail is not a recognised assembler or transformer (e.g.
        ``"compute: done"`` or an internal raster/optimize message).
    """
    pipeline = _PIPELINE_LABELS.get(detail.partition(":")[0])
    if pipeline is not None:
        return pipeline()
    transformer = transformer_registry.progress_label(detail)
    if transformer is not None:
        return transformer
    name, sep, _rest = detail.partition(":")
    if sep:
        return step_registry.progress_label(name)
    return None


def status_message_for_key(
    key: str,
    workpieces_by_uid: Mapping[str, "WorkPiece"],
    steps_by_uid: Mapping[str, "Step"],
) -> str:
    """Translate a pipeline batch progress payload into a status string.

    :param key: A raygeo batch progress payload: a node key with an
        optional ``\\t``-separated activity detail (or an empty string
        when idle).
    :param workpieces_by_uid: Map of workpiece uid to :class:`WorkPiece`.
    :param steps_by_uid: Map of step uid to :class:`Step`.
    :returns: A ``_()``-marked status message, or ``""`` when idle or
        when the payload only carries a completion marker.
    """
    if not key:
        return ""
    node_key, sep, detail = key.partition("\t")
    if not node_key:
        return ""
    base = _node_status_message(node_key, workpieces_by_uid, steps_by_uid)
    if sep:
        activity = _activity_label(detail)
        if activity:
            return _("{status} — {activity}").format(
                status=base, activity=activity
            )
    return base


def _node_status_message(
    node_key: str,
    workpieces_by_uid: Mapping[str, "WorkPiece"],
    steps_by_uid: Mapping[str, "Step"],
) -> str:
    """Translate a bare node key into its base status string."""
    if node_key == "job":
        return _("Aggregating job")
    if node_key == "job:encode":
        return _("Generating machine code")
    if node_key == "job:machinexform":
        return _("Applying machine transform")
    parsed = parse_workpiece_key(node_key)
    if parsed is not None:
        wp_uid, step_uid = parsed
        workpiece = workpieces_by_uid.get(wp_uid)
        step = steps_by_uid.get(step_uid)
        if workpiece is None or step is None:
            return _("Processing")
        return _("Processing '{workpiece}' — {step}").format(
            workpiece=workpiece.name, step=step.typelabel
        )
    if node_key.startswith("step:"):
        step = steps_by_uid.get(node_key.split(":", 1)[1])
        if step is None:
            return _("Assembling")
        return _("Assembling '{step}'").format(step=step.typelabel)
    return _("Processing")
