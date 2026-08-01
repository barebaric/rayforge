"""Translation helpers for assembler warnings emitted by raygeo.

raygeo produces typed ``AssemblyWarning`` objects (kind + structured
fields); this module turns them into human-readable, translatable
strings for the UI.  ``gettext`` ``_()`` is applied here, so the
templates are picked up by ``xgettext``.
"""

from gettext import gettext as _

from raygeo.ops.assembly import AssemblyWarningKind


def translate_assembly_warning(w) -> str:
    """Translate an ``AssemblyWarning`` into a user-facing string.

    :param w: A raygeo ``AssemblyWarning`` (or any object exposing
        ``kind``, ``face_id``, ``region`` and ``detail``).
    :returns: A ``_()``-marked, formatted message.
    """
    label = w.face_id if w.face_id else _("default face")
    if w.kind == AssemblyWarningKind.FACE_FAILED:
        return _("Face '{face}' could not be machined: {detail}").format(
            face=label, detail=w.detail
        )
    if w.kind == AssemblyWarningKind.REGION_FAILED:
        idx = w.region if w.region is not None else "?"
        return _(
            "Region {region} of face '{face}' could not be machined: {detail}"
        ).format(region=idx, face=label, detail=w.detail)
    return _("Machining warning: {detail}").format(detail=w.detail)
