"""
Pure Python .po to .mo compiler.

This module provides a cross-platform way to compile gettext .po files
to .mo files without requiring the external msgfmt utility.
"""

import struct
from pathlib import Path
from typing import List, Tuple


def parse_po_file(po_path: Path) -> List[Tuple[str, str]]:
    """
    Parse a .po file and return list of (msgid, msgstr) tuples.

    Args:
        po_path: Path to the .po file.

    Returns:
        List of (msgid, msgstr) tuples, excluding the header entry.
    """
    entries = []
    msgid_lines = []
    msgstr_lines = []
    in_msgid = False
    in_msgstr = False

    with open(po_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if line.startswith("msgid "):
                # Save previous entry if we have one
                if msgid_lines and msgstr_lines:
                    msgid = _join_po_lines(msgid_lines)
                    msgstr = _join_po_lines(msgstr_lines)
                    if msgid:
                        entries.append((msgid, msgstr))

                msgid_lines = [line[6:]]  # Remove "msgid "
                msgstr_lines = []
                in_msgid = True
                in_msgstr = False

            elif line.startswith("msgstr "):
                in_msgid = False
                in_msgstr = True
                msgstr_lines = [line[7:]]  # Remove "msgstr "

            elif line.startswith('"') and in_msgid:
                msgid_lines.append(line)

            elif line.startswith('"') and in_msgstr:
                msgstr_lines.append(line)

            elif line.startswith("#") or not line:
                # Comment or blank line - finalize current entry
                if in_msgstr and msgid_lines and msgstr_lines:
                    msgid = _join_po_lines(msgid_lines)
                    msgstr = _join_po_lines(msgstr_lines)
                    if msgid:
                        entries.append((msgid, msgstr))
                    msgid_lines = []
                    msgstr_lines = []
                    in_msgid = False
                    in_msgstr = False

    # Handle last entry if file doesn't end with blank line
    if msgid_lines and msgstr_lines:
        msgid = _join_po_lines(msgid_lines)
        msgstr = _join_po_lines(msgstr_lines)
        if msgid:
            entries.append((msgid, msgstr))

    return entries


def _join_po_lines(lines: List[str]) -> str:
    """
    Join quoted .po file lines into a single string.

    Handles multi-line strings like:
        "This is a "
        "multi-line string"
    """
    return "".join(s.strip('"') for s in lines).replace("\\n", "\n")


def write_mo_file(mo_path: Path, entries: List[Tuple[str, str]]) -> None:
    """
    Write entries to a .mo file.

    Args:
        mo_path: Path to write the .mo file.
        entries: List of (msgid, msgstr) tuples.
    """
    if not entries:
        raise ValueError("No entries to write")

    # Ensure parent directories exist
    mo_path.parent.mkdir(parents=True, exist_ok=True)

    # Sort entries by msgid for binary search
    entries = sorted(entries, key=lambda x: x[0])

    # Calculate sizes and offsets
    num_entries = len(entries)

    # Build string data
    orig_strings = []
    trans_strings = []
    orig_data = bytearray()
    trans_data = bytearray()

    for msgid, msgstr in entries:
        orig_bytes = msgid.encode("utf-8")
        trans_bytes = msgstr.encode("utf-8")

        orig_strings.append((len(orig_bytes), len(orig_data)))
        trans_strings.append((len(trans_bytes), len(trans_data)))

        orig_data.extend(orig_bytes)
        orig_data.append(0)  # null terminator
        trans_data.extend(trans_bytes)
        trans_data.append(0)  # null terminator

    # Calculate offsets
    # Header: 7 * 4 bytes (magic, revision, num, orig_off,
    # trans_off, hash_size, hash_off)
    header_size = 7 * 4
    orig_table_size = num_entries * 8  # Each entry: length (4) + offset (4)
    trans_table_size = num_entries * 8

    orig_table_offset = header_size
    trans_table_offset = orig_table_offset + orig_table_size
    hash_table_offset = trans_table_offset + trans_table_size
    orig_strings_offset = hash_table_offset  # No hash table
    trans_strings_offset = orig_strings_offset + len(orig_data)

    with open(mo_path, "wb") as f:
        # Magic number (little endian format)
        f.write(struct.pack("<I", 0x950412DE))

        # Format revision (0)
        f.write(struct.pack("<I", 0))

        # Number of entries
        f.write(struct.pack("<I", num_entries))

        # Offset of original strings table
        f.write(struct.pack("<I", orig_table_offset))

        # Offset of translated strings table
        f.write(struct.pack("<I", trans_table_offset))

        # Hash table size (0 = no hash table)
        f.write(struct.pack("<I", 0))

        # Hash table offset
        f.write(struct.pack("<I", 0))

        # Original strings table
        for length, offset in orig_strings:
            f.write(struct.pack("<II", length, orig_strings_offset + offset))

        # Translated strings table
        for length, offset in trans_strings:
            f.write(struct.pack("<II", length, trans_strings_offset + offset))

        # Original strings data
        f.write(orig_data)

        # Translated strings data
        f.write(trans_data)


def compile_po_to_mo(po_path: Path, mo_path: Path) -> bool:
    """
    Compile a .po file to a .mo file.

    Args:
        po_path: Path to the source .po file.
        mo_path: Path to write the destination .mo file.

    Returns:
        True if compilation succeeded, False otherwise.
    """
    try:
        entries = parse_po_file(po_path)
        if not entries:
            return False
        write_mo_file(mo_path, entries)
        return True
    except Exception:
        return False
