from gettext import gettext as _
from typing import TYPE_CHECKING

from gi.repository import Adw, Gtk

from ...core.job_origin import JobAnchor, JobOrigin, StartFrom
from ...core.undo import ChangePropertyCommand
from ...machine.job_placement import JobPlacementError, resolve_start_point
from ..shared.gtk import apply_css

if TYPE_CHECKING:
    from ...core.doc import Doc
    from ...machine.models.machine import Machine

_MODES = [
    StartFrom.ABSOLUTE,
    StartFrom.CURRENT_POSITION,
    StartFrom.USER_ORIGIN,
]

# Rows of the selector as seen on screen, top to bottom.
_GRID = [
    [JobAnchor.TOP_LEFT, JobAnchor.TOP, JobAnchor.TOP_RIGHT],
    [JobAnchor.LEFT, JobAnchor.CENTER, JobAnchor.RIGHT],
    [JobAnchor.BOTTOM_LEFT, JobAnchor.BOTTOM, JobAnchor.BOTTOM_RIGHT],
]

apply_css(
    """
button.job-origin-cell {
    min-width: 14px;
    min-height: 14px;
    padding: 0;
    border-radius: 3px;
}
button.job-origin-cell:checked {
    background-color: @accent_bg_color;
}
"""
)


class JobOriginRows:
    """
    The "Start From" mode and the 9-point job origin selector.

    Provides two preference rows: a combo row for the mode and an
    action row holding a 3x3 grid of toggle buttons for the anchor.
    Changes go through the document's undo history.
    """

    def __init__(self):
        self.doc: Doc | None = None
        self.machine: Machine | None = None
        self._updating = False

        self.start_from_row = Adw.ComboRow(
            title=_("Start From"),
            model=Gtk.StringList.new([mode.label for mode in _MODES]),
        )
        self.start_from_row.set_tooltip_text(
            _(
                "Where the job runs: as placed on the canvas, at the "
                "laser head, or on the zero point of the active WCS"
            )
        )
        self.start_from_row.connect("notify::selected", self._on_mode_changed)

        self.anchor_row = Adw.ActionRow(title=_("Job Origin"))
        self.anchor_row.set_tooltip_text(
            _("The point of the job that is placed on the start position")
        )
        grid = Gtk.Grid(row_spacing=2, column_spacing=2)
        grid.set_valign(Gtk.Align.CENTER)
        self.anchor_buttons: dict[JobAnchor, Gtk.ToggleButton] = {}
        group: Gtk.ToggleButton | None = None
        for row, anchors in enumerate(_GRID):
            for col, anchor in enumerate(anchors):
                button = Gtk.ToggleButton()
                button.add_css_class("job-origin-cell")
                button.set_tooltip_text(anchor.label)
                if group is None:
                    group = button
                else:
                    button.set_group(group)
                button.connect("toggled", self._on_anchor_toggled, anchor)
                grid.attach(button, col, row, 1, 1)
                self.anchor_buttons[anchor] = button
        self.anchor_row.add_suffix(grid)

        self.update()

    def set_doc(self, doc: "Doc | None"):
        if self.doc is not None:
            self.doc.job_origin_changed.disconnect(self._on_job_origin_changed)
        self.doc = doc
        if doc is not None:
            doc.job_origin_changed.connect(self._on_job_origin_changed)
        self.update()

    def set_machine(self, machine: "Machine | None"):
        self.machine = machine
        self.update()

    def _on_job_origin_changed(self, sender):
        self.update()

    def update(self):
        """Mirrors the document's job origin into the rows."""
        job_origin = self.doc.job_origin if self.doc else JobOrigin()
        self._updating = True
        try:
            self.start_from_row.set_selected(
                _MODES.index(job_origin.start_from)
            )
            self.anchor_buttons[job_origin.anchor].set_active(True)
        finally:
            self._updating = False
        self.start_from_row.set_sensitive(self.doc is not None)
        self.start_from_row.set_subtitle(self._describe_start(job_origin))
        self.anchor_row.set_subtitle(job_origin.anchor.label)
        self.anchor_row.set_sensitive(
            self.doc is not None and not job_origin.is_absolute
        )

    def _describe_start(self, job_origin: JobOrigin) -> str:
        if job_origin.is_absolute:
            return _("As placed on the canvas")
        if self.machine is None:
            return ""
        if job_origin.start_from == StartFrom.USER_ORIGIN:
            if self.machine.wcs_origin_is_workarea_origin:
                return _("On the work area origin")
            return _("On the zero point of {wcs}").format(
                wcs=self.machine.active_wcs
            )
        try:
            point = resolve_start_point(self.machine, job_origin.start_from)
        except JobPlacementError:
            return _("Needs a connected, idle machine")
        if point is None:
            return ""
        x, y = self._world_to_wcs(point)
        if self.machine.pointer_alignment_enabled:
            text = _("At the pointer dot: X {x:.2f}  Y {y:.2f}")
        else:
            text = _("At the laser head: X {x:.2f}  Y {y:.2f}")
        return text.format(x=x, y=y)

    def _world_to_wcs(self, point: tuple[float, float]) -> tuple[float, float]:
        """A world point in the coordinates of the active WCS."""
        assert self.machine is not None
        panel = self.machine.panel
        m_x, m_y = panel.world_point_to_machine(*point)
        off_x, off_y, _z = panel.get_command_offset(
            wcs_offset=self.machine.get_active_wcs_offset(),
            wcs_is_workarea_origin=self.machine.wcs_origin_is_workarea_origin,
        )
        return (m_x - off_x, m_y - off_y)

    def _set_job_origin(self, job_origin: JobOrigin):
        doc = self.doc
        if doc is None or job_origin == doc.job_origin:
            return
        command = ChangePropertyCommand(
            target=doc,
            property_name="job_origin",
            new_value=job_origin,
            setter_method_name="set_job_origin",
            name=_("Change Start From"),
        )
        doc.history_manager.execute(command)

    def _on_mode_changed(self, row, _pspec):
        if self._updating or self.doc is None:
            return
        index = row.get_selected()
        if not 0 <= index < len(_MODES):
            return
        current = self.doc.job_origin
        self._set_job_origin(JobOrigin(_MODES[index], current.anchor))

    def _on_anchor_toggled(self, button, anchor: JobAnchor):
        if self._updating or self.doc is None or not button.get_active():
            return
        current = self.doc.job_origin
        self._set_job_origin(JobOrigin(current.start_from, anchor))
