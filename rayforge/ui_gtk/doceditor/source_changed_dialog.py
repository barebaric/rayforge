"""Dialog asking whether imported files that changed should be reloaded."""

from collections.abc import Callable
from gettext import gettext as _
from gettext import ngettext

from gi.repository import Adw, Gtk

from ...core.source_asset import SourceAsset

ReloadCallback = Callable[[list[SourceAsset], bool, list[SourceAsset]], None]
IgnoreCallback = Callable[[list[SourceAsset]], None]


class SourceChangedDialog(Adw.AlertDialog):
    """
    Lists imported files that changed on disk and lets the user choose
    which of them to reload, and whether to reload them automatically
    from now on.
    """

    def __init__(
        self,
        assets: list[SourceAsset],
        on_reload: ReloadCallback | None = None,
        on_ignore: IgnoreCallback | None = None,
    ):
        super().__init__()
        self._assets = list(assets)
        self._on_reload = on_reload
        self._on_ignore = on_ignore
        self._checks: dict[str, Gtk.CheckButton] = {}

        count = len(self._assets)
        self.set_heading(
            ngettext("Imported File Changed", "Imported Files Changed", count)
        )
        if count == 1:
            body = _(
                '"{name}" was changed on disk. Reload it? Position, size '
                "and settings are kept."
            ).format(name=self._assets[0].name)
        else:
            body = _(
                "These imported files were changed on disk. Reload them? "
                "Position, size and settings are kept."
            )
        self.set_body(body)
        self.set_extra_child(self._build_options(count))

        self.add_response("ignore", _("_Ignore"))
        self.add_response("reload", _("_Reload"))
        self.set_response_appearance(
            "reload", Adw.ResponseAppearance.SUGGESTED
        )
        self.set_default_response("reload")
        self.set_close_response("ignore")
        self.connect("response", self._on_response)

    def _build_options(self, count: int) -> Gtk.Widget:
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        if count > 1:
            for asset in self._assets:
                check = Gtk.CheckButton(label=asset.name, active=True)
                check.set_tooltip_text(str(asset.source_file))
                self._checks[asset.uid] = check
                box.append(check)
        self._always = Gtk.CheckButton(
            label=ngettext(
                "Always reload this file without asking",
                "Always reload the selected files without asking",
                count,
            )
        )
        box.append(self._always)
        return box

    def set_selected(self, asset: SourceAsset, selected: bool) -> None:
        check = self._checks.get(asset.uid)
        if check is not None:
            check.set_active(selected)

    def set_always(self, always: bool) -> None:
        self._always.set_active(always)

    def selected_assets(self) -> list[SourceAsset]:
        return [
            asset
            for asset in self._assets
            if asset.uid not in self._checks
            or self._checks[asset.uid].get_active()
        ]

    def _on_response(self, dialog, response_id: str):
        if response_id == "reload":
            selected = self.selected_assets()
            skipped = [a for a in self._assets if a not in selected]
            if self._on_reload:
                self._on_reload(selected, self._always.get_active(), skipped)
        elif self._on_ignore:
            self._on_ignore(list(self._assets))
        if self.get_root() is not None:
            self.close()
