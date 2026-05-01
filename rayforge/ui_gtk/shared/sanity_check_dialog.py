from gettext import gettext as _
from typing import Callable, Optional

from gi.repository import Adw, Gtk, Pango

from ...machine.sanity.result import (
    IssueSeverity,
    SanityReport,
    ISSUE_CATEGORY_LABELS,
)


class SanityCheckDialog(Adw.MessageDialog):
    def __init__(
        self,
        parent: Gtk.Window,
        report: SanityReport,
        on_proceed: Optional[Callable[[], None]] = None,
        **kwargs,
    ):
        super().__init__(transient_for=parent, **kwargs)
        self._report = report
        self._on_proceed = on_proceed

        self.set_size_request(550, -1)
        self.set_heading(_("Job Sanity Check"))
        self.set_body(self._build_summary())

        content = self._build_issue_list()
        self.set_extra_child(content)

        self.add_response("cancel", _("_Cancel"))
        self.add_response("proceed", _("_Proceed"))
        self.set_default_response("cancel")
        self.set_close_response("cancel")
        if report.has_errors:
            self.set_response_appearance(
                "proceed", Adw.ResponseAppearance.DESTRUCTIVE
            )
        else:
            self.set_response_appearance(
                "proceed", Adw.ResponseAppearance.SUGGESTED
            )

        self.connect("response", self._on_response)

    def _build_summary(self) -> str:
        n_errors = sum(
            1 for i in self._report.issues if i.severity == IssueSeverity.ERROR
        )
        n_warnings = sum(
            1
            for i in self._report.issues
            if i.severity == IssueSeverity.WARNING
        )
        parts = []
        if n_errors:
            parts.append(_("{} error(s)").format(n_errors))
        if n_warnings:
            parts.append(_("{} warning(s)").format(n_warnings))
        if not parts:
            return _("No issues found.")
        return _(
            "Found {summary}. Proceeding may cause "
            "damage to your machine or workpiece."
        ).format(summary=", ".join(parts))

    def _build_issue_list(self) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        box.set_margin_top(12)

        errors = [
            i for i in self._report.issues if i.severity == IssueSeverity.ERROR
        ]
        warnings = [
            i
            for i in self._report.issues
            if i.severity == IssueSeverity.WARNING
        ]

        if errors:
            box.append(self._make_section_label(_("Errors")))
            for issue in errors:
                box.append(self._make_issue_row(issue))

        if warnings:
            box.append(self._make_section_label(_("Warnings")))
            for issue in warnings:
                box.append(self._make_issue_row(issue))

        scrolled = Gtk.ScrolledWindow()
        scrolled.set_propagate_natural_height(True)
        scrolled.set_max_content_height(300)
        scrolled.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        scrolled.set_child(box)
        return scrolled

    def _make_section_label(self, text: str) -> Gtk.Label:
        label = Gtk.Label(
            label=text,
            xalign=0.0,
            margin_top=(8 if text == _("Warnings") else 0),
        )
        label.add_css_class("caption-heading")
        return label

    def _make_issue_row(self, issue) -> Gtk.Box:
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        row.set_margin_top(2)
        row.set_margin_bottom(2)

        icon_name = self._get_icon_name(issue.severity)
        icon = Gtk.Image.new_from_icon_name(icon_name)
        icon.set_pixel_size(16)
        icon.set_valign(Gtk.Align.START)
        row.append(icon)

        desc = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)

        title_text = self._format_title(issue)
        title = Gtk.Label(label=title_text, xalign=0.0, wrap=True)
        title.set_attributes(Pango.AttrList.new())
        desc.append(title)

        subtitle_text = self._format_subtitle(issue)
        if subtitle_text:
            sub = Gtk.Label(
                label=subtitle_text,
                xalign=0.0,
                wrap=True,
            )
            sub.add_css_class("caption")
            sub.add_css_class("dim-label")
            desc.append(sub)

        row.append(desc)
        return row

    @staticmethod
    def _get_icon_name(severity: IssueSeverity) -> str:
        if severity == IssueSeverity.ERROR:
            return "dialog-error-symbolic"
        return "dialog-warning-symbolic"

    @staticmethod
    def _format_title(issue) -> str:
        label = ISSUE_CATEGORY_LABELS[issue.category]
        if issue.zone_name:
            return f'{label}: "{issue.zone_name}"'
        return label

    @staticmethod
    def _format_subtitle(issue) -> str:
        parts = []
        if issue.segment_start and issue.segment_end:
            parts.append(
                "({:.1f}, {:.1f}) → ({:.1f}, {:.1f})".format(
                    *issue.segment_start, *issue.segment_end
                )
            )
        elif issue.segment_end:
            parts.append("({:.1f}, {:.1f})".format(*issue.segment_end))
        if issue.message:
            parts.append(issue.message)
        return "\n".join(parts)

    def _on_response(self, dialog, response_id):
        self.destroy()
        if response_id == "proceed" and self._on_proceed:
            self._on_proceed()
