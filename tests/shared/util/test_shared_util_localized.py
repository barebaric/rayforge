"""
Tests for the localized utilities, including the addon domain patch.
"""

import gettext
from pathlib import Path

from rayforge.shared.util.localized import register_addon_domain


def _build_addon_locale(
    tmp_path: Path, entries: list[tuple[str, str]]
) -> Path:
    """Write a fake addon locale tree containing a single domain.

    Args:
        tmp_path: pytest fixture tmp dir.
        entries: List of (msgid, msgstr) tuples to compile into the .mo.

    Returns:
        Path to the locale directory to pass to register_addon_domain.
    """
    from rayforge.shared.util.po_compiler import write_mo_file

    locale_dir = tmp_path / "locale"
    mo_path = locale_dir / "en" / "LC_MESSAGES" / "fakeaddon.mo"
    write_mo_file(mo_path, entries)
    return locale_dir


_HEADER = "Content-Type: text/plain; charset=UTF-8\n"


class TestRegisterAddonDomain:
    """Tests for register_addon_domain."""

    def test_translates_string_from_addon(self, tmp_path: Path, monkeypatch):
        """Addon translations are returned when present."""
        import importlib

        import rayforge.shared.util.localized as mod

        monkeypatch.setattr(gettext, "gettext", gettext.gettext)
        try:
            locale_dir = _build_addon_locale(
                tmp_path,
                [("", _HEADER), ("Hello", "Hola")],
            )
            register_addon_domain("fakeaddon", locale_dir)

            assert gettext.gettext("Hello") == "Hola"
        finally:
            importlib.reload(mod)

    def test_does_not_return_empty_translation(self, tmp_path: Path):
        """Regression test for issue #315.

        If an addon .mo file accidentally contains empty msgstr entries,
        register_addon_domain must fall back to the original message
        instead of returning an empty string (which would blank UI text).
        """
        import importlib

        import rayforge.shared.util.localized as mod

        importlib.reload(mod)
        try:
            locale_dir = _build_addon_locale(
                tmp_path,
                # An entry mapping a string to an empty translation, as a
                # buggy .mo compiler would produce.
                [("", _HEADER), ("Cancel", "")],
            )
            mod.register_addon_domain("fakeaddon", locale_dir)

            assert gettext.gettext("Cancel") == "Cancel"
        finally:
            importlib.reload(mod)
