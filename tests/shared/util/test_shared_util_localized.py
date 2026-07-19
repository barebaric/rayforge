"""
Tests for the localized utilities, including the addon domain patch.
"""

import pytest
from pathlib import Path
import gettext as gettext_module

from rayforge.shared.util.localized import register_addon_domain
from rayforge.shared.util.po_compiler import write_mo_file

_HEADER = "Content-Type: text/plain; charset=UTF-8\n"


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
    locale_dir = tmp_path / "locale"
    mo_path = locale_dir / "en" / "LC_MESSAGES" / "fakeaddon.mo"
    write_mo_file(mo_path, entries)
    return locale_dir


@pytest.fixture
def isolated_gettext():
    """Snapshot and restore gettext.gettext and the addon domain chain.

    register_addon_domain patches the global ``gettext.gettext`` and
    appends to a process-wide chain of addon translators. Without
    clean teardown, that state leaks across tests.
    """
    import rayforge.shared.util.localized as mod

    saved_gettext = gettext_module.gettext
    saved_translators = list(mod._chain._translators)
    saved_installed = mod._chain._installed
    saved_original = mod._chain._original
    try:
        yield
    finally:
        gettext_module.gettext = saved_gettext
        mod._chain._translators = saved_translators
        mod._chain._installed = saved_installed
        mod._chain._original = saved_original


class TestRegisterAddonDomain:
    """Tests for register_addon_domain."""

    def test_translates_string_from_addon(
        self, tmp_path: Path, isolated_gettext
    ):
        """Addon translations are returned when present."""
        locale_dir = _build_addon_locale(
            tmp_path,
            [("", _HEADER), ("Hello", "Hola")],
        )
        register_addon_domain("fakeaddon", locale_dir)

        assert gettext_module.gettext("Hello") == "Hola"

    def test_does_not_return_empty_translation(
        self, tmp_path: Path, isolated_gettext
    ):
        """Regression test for issue #315.

        If an addon .mo file accidentally contains empty msgstr entries,
        register_addon_domain must fall back to the original message
        instead of returning an empty string (which would blank UI text).
        """
        locale_dir = _build_addon_locale(
            tmp_path,
            # An entry mapping a string to an empty translation, as a
            # buggy .mo compiler would produce.
            [("", _HEADER), ("Cancel", "")],
        )
        register_addon_domain("fakeaddon", locale_dir)

        assert gettext_module.gettext("Cancel") == "Cancel"

    def test_falls_back_to_msgid_when_no_translation(
        self, tmp_path: Path, isolated_gettext
    ):
        """Unknown strings fall back to the msgid unchanged."""
        locale_dir = _build_addon_locale(
            tmp_path,
            [("", _HEADER), ("Hello", "Hola")],
        )
        register_addon_domain("fakeaddon", locale_dir)

        assert gettext_module.gettext("Unknown") == "Unknown"
