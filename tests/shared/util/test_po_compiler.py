"""
Tests for the pure Python .po to .mo compiler.
"""

from pathlib import Path

import pytest

from rayforge.shared.util.po_compiler import (
    parse_po_file,
    write_mo_file,
    compile_po_to_mo,
)


class TestParsePoFile:
    """Tests for parse_po_file function."""

    def test_parse_simple_entry(self, tmp_path: Path):
        """Test parsing a simple single-entry .po file."""
        po_content = """msgid ""
msgstr ""
"Content-Type: text/plain; charset=UTF-8\\n"

msgid "Hello"
msgstr "Hallo"
"""
        po_file = tmp_path / "test.po"
        po_file.write_text(po_content)

        entries = parse_po_file(po_file)
        assert len(entries) == 1
        assert entries[0] == ("Hello", "Hallo")

    def test_parse_multiple_entries(self, tmp_path: Path):
        """Test parsing multiple entries."""
        po_content = """msgid ""
msgstr ""

msgid "Hello"
msgstr "Hallo"

msgid "Goodbye"
msgstr "Auf Wiedersehen"
"""
        po_file = tmp_path / "test.po"
        po_file.write_text(po_content)

        entries = parse_po_file(po_file)
        assert len(entries) == 2
        assert entries[0] == ("Hello", "Hallo")
        assert entries[1] == ("Goodbye", "Auf Wiedersehen")

    def test_parse_multiline_string(self, tmp_path: Path):
        """Test parsing multi-line strings."""
        po_content = """msgid ""
msgstr ""

msgid "This is a "
"multi-line string"
msgstr "Dies ist eine "
"mehrzeilige Zeichenkette"
"""
        po_file = tmp_path / "test.po"
        po_file.write_text(po_content)

        entries = parse_po_file(po_file)
        assert len(entries) == 1
        assert entries[0][0] == "This is a multi-line string"
        assert entries[0][1] == "Dies ist eine mehrzeilige Zeichenkette"

    def test_parse_newlines_in_string(self, tmp_path: Path):
        """Test parsing strings with embedded newlines."""
        po_content = """msgid ""
msgstr ""

msgid "Line 1\\nLine 2"
msgstr "Zeile 1\\nZeile 2"
"""
        po_file = tmp_path / "test.po"
        po_file.write_text(po_content)

        entries = parse_po_file(po_file)
        assert len(entries) == 1
        assert entries[0][0] == "Line 1\nLine 2"
        assert entries[0][1] == "Zeile 1\nZeile 2"

    def test_parse_with_comments(self, tmp_path: Path):
        """Test that comments are ignored."""
        po_content = """# This is a comment
msgid ""
msgstr ""

# Another comment
msgid "Hello"
msgstr "Hallo"
"""
        po_file = tmp_path / "test.po"
        po_file.write_text(po_content)

        entries = parse_po_file(po_file)
        assert len(entries) == 1
        assert entries[0] == ("Hello", "Hallo")

    def test_parse_empty_msgid_skipped(self, tmp_path: Path):
        """Test that empty msgid (header) is skipped."""
        po_content = """msgid ""
msgstr ""
"Content-Type: text/plain; charset=UTF-8\\n"

msgid "Hello"
msgstr "Hallo"
"""
        po_file = tmp_path / "test.po"
        po_file.write_text(po_content)

        entries = parse_po_file(po_file)
        assert len(entries) == 1
        # The header with empty msgid should not be in entries
        assert entries[0] == ("Hello", "Hallo")

    def test_parse_empty_msgstr(self, tmp_path: Path):
        """Test parsing entries with empty translations."""
        po_content = """msgid ""
msgstr ""

msgid "Untranslated"
msgstr ""
"""
        po_file = tmp_path / "test.po"
        po_file.write_text(po_content)

        entries = parse_po_file(po_file)
        assert len(entries) == 1
        assert entries[0] == ("Untranslated", "")


class TestWriteMoFile:
    """Tests for write_mo_file function."""

    def test_write_mo_header(self, tmp_path: Path):
        """Test that .mo file has correct header."""
        entries = [("Hello", "Hallo")]
        mo_file = tmp_path / "test.mo"

        write_mo_file(mo_file, entries)

        content = mo_file.read_bytes()
        # Check magic number (little endian 0x950412DE)
        assert content[0:4] == b"\xde\x12\x04\x95"
        # Check format revision (0)
        assert content[4:8] == b"\x00\x00\x00\x00"
        # Check number of entries (1)
        assert content[8:12] == b"\x01\x00\x00\x00"

    def test_write_mo_sorted_entries(self, tmp_path: Path):
        """Test that entries are sorted by msgid."""
        entries = [("Zebra", "Zebra"), ("Apple", "Apfel"), ("Mango", "Mango")]
        mo_file = tmp_path / "test.mo"

        write_mo_file(mo_file, entries)

        content = mo_file.read_bytes()
        # The strings should appear in sorted order: Apple, Mango, Zebra
        apple_pos = content.find(b"Apple")
        mango_pos = content.find(b"Mango")
        zebra_pos = content.find(b"Zebra")

        assert apple_pos < mango_pos < zebra_pos

    def test_write_mo_raises_on_empty(self, tmp_path: Path):
        """Test that writing empty entries raises error."""
        mo_file = tmp_path / "test.mo"

        with pytest.raises(ValueError, match="No entries"):
            write_mo_file(mo_file, [])


class TestCompilePoToMo:
    """Tests for compile_po_to_mo function."""

    def test_compile_basic(self, tmp_path: Path):
        """Test basic compilation from .po to .mo."""
        po_content = """msgid ""
msgstr ""

msgid "Hello"
msgstr "Hallo"

msgid "Goodbye"
msgstr "Auf Wiedersehen"
"""
        po_file = tmp_path / "test.po"
        mo_file = tmp_path / "test.mo"
        po_file.write_text(po_content)

        result = compile_po_to_mo(po_file, mo_file)

        assert result is True
        assert mo_file.exists()
        assert mo_file.stat().st_size > 0

    def test_compile_creates_missing_directories(self, tmp_path: Path):
        """Test that compilation creates parent directories."""
        po_content = """msgid ""
msgstr ""

msgid "Hello"
msgstr "Hallo"
"""
        po_file = tmp_path / "test.po"
        mo_file = tmp_path / "subdir" / "nested" / "test.mo"
        po_file.write_text(po_content)

        result = compile_po_to_mo(po_file, mo_file)

        assert result is True
        assert mo_file.exists()

    def test_compile_returns_false_on_empty_po(self, tmp_path: Path):
        """Test that empty .po file returns False."""
        po_content = """msgid ""
msgstr ""
"""
        po_file = tmp_path / "test.po"
        mo_file = tmp_path / "test.mo"
        po_file.write_text(po_content)

        result = compile_po_to_mo(po_file, mo_file)

        assert result is False

    def test_compile_returns_false_on_invalid_po(self, tmp_path: Path):
        """Test that invalid .po file returns False."""
        po_file = tmp_path / "test.po"
        mo_file = tmp_path / "test.mo"
        po_file.write_text("This is not a valid .po file")

        result = compile_po_to_mo(po_file, mo_file)

        assert result is False

    def test_compile_preserves_unicode(self, tmp_path: Path):
        """Test that Unicode characters are preserved."""
        po_content = """msgid ""
msgstr ""

msgid "Hello"
msgstr "你好世界"
"""
        po_file = tmp_path / "test.po"
        mo_file = tmp_path / "test.mo"
        po_file.write_text(po_content, encoding="utf-8")

        result = compile_po_to_mo(po_file, mo_file)

        assert result is True
        content = mo_file.read_bytes()
        assert "你好世界".encode("utf-8") in content


class TestMoFileWithGettext:
    """Integration tests using Python's gettext module."""

    def test_mo_file_loadable_by_gettext(self, tmp_path: Path):
        """Test that compiled .mo file can be loaded by gettext."""
        import gettext

        po_content = """msgid ""
msgstr ""
"Content-Type: text/plain; charset=UTF-8\\n"

msgid "Hello"
msgstr "Hallo"

msgid "Goodbye"
msgstr "Auf Wiedersehen"
"""
        po_file = tmp_path / "test.po"
        # gettext expects: localedir/lang/LC_MESSAGES/domain.mo
        mo_file = tmp_path / "en" / "LC_MESSAGES" / "test.mo"
        po_file.write_text(po_content)

        result = compile_po_to_mo(po_file, mo_file)
        assert result is True

        # Try to load with gettext
        translation = gettext.translation(
            "test", localedir=tmp_path, languages=["en"], fallback=True
        )
        _ = translation.gettext

        assert _("Hello") == "Hallo"
        assert _("Goodbye") == "Auf Wiedersehen"

    def test_mo_file_with_format_strings(self, tmp_path: Path):
        """Test .mo file with format strings."""
        import gettext

        po_content = """msgid ""
msgstr ""

msgid "Hello, %s!"
msgstr "Hallo, %s!"
"""
        po_file = tmp_path / "test.po"
        mo_file = tmp_path / "en" / "LC_MESSAGES" / "test.mo"
        po_file.write_text(po_content)

        result = compile_po_to_mo(po_file, mo_file)
        assert result is True

        translation = gettext.translation(
            "test", localedir=tmp_path, languages=["en"], fallback=True
        )
        _ = translation.gettext

        assert _("Hello, %s!") % "World" == "Hallo, World!"
