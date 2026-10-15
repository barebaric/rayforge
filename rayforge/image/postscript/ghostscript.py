"""
Conversion of PostScript, EPS and legacy Illustrator files to PDF with
Ghostscript, an optional runtime dependency.
"""

import hashlib
import logging
import shutil
import subprocess
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path

logger = logging.getLogger(__name__)

GHOSTSCRIPT_NAMES = ("gs", "gswin64c", "gswin32c")
CONVERSION_TIMEOUT_S = 120
_CACHE_SIZE = 8
_cache: "OrderedDict[str, bytes]" = OrderedDict()
_cache_lock = threading.Lock()


class GhostscriptNotFound(Exception):
    """Raised when no Ghostscript executable is available."""


class GhostscriptError(Exception):
    """Raised when Ghostscript fails to convert a file."""


def find_ghostscript() -> str | None:
    """Returns the path of the Ghostscript console executable, if any."""
    for name in GHOSTSCRIPT_NAMES:
        path = shutil.which(name)
        if path:
            return path
    return None


def clear_cache():
    """Forgets all cached conversions."""
    with _cache_lock:
        _cache.clear()


def _command(gs: str, source: Path, target: Path) -> list[str]:
    return [
        gs,
        "-q",
        "-dSAFER",
        "-dBATCH",
        "-dNOPAUSE",
        "-sDEVICE=pdfwrite",
        "-dEPSCrop",
        "-dNoOutputFonts",
        f"-sOutputFile={target}",
        str(source),
    ]


def _run(gs: str, data: bytes) -> bytes:
    with tempfile.TemporaryDirectory(prefix="rayforge-gs-") as tmp:
        source = Path(tmp) / "input.ps"
        target = Path(tmp) / "output.pdf"
        source.write_bytes(data)
        try:
            proc = subprocess.run(
                _command(gs, source, target),
                capture_output=True,
                timeout=CONVERSION_TIMEOUT_S,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            raise GhostscriptError(str(e)) from e
        pdf = target.read_bytes() if target.exists() else b""
        if proc.returncode != 0 or not pdf.startswith(b"%PDF"):
            message = proc.stderr.decode(errors="replace").strip()
            raise GhostscriptError(message or f"exit status {proc.returncode}")
        return pdf


def convert_to_pdf(data: bytes) -> bytes:
    """
    Converts PostScript or EPS data to PDF. Text is converted to outlines
    and an EPS bounding box becomes the page size. Results are cached by
    content, so repeated previews of the same file convert only once.

    Raises:
        GhostscriptNotFound: If Ghostscript is not installed.
        GhostscriptError: If the conversion fails.
    """
    key = hashlib.sha256(data).hexdigest()
    with _cache_lock:
        cached = _cache.get(key)
        if cached is not None:
            _cache.move_to_end(key)
            return cached

    gs = find_ghostscript()
    if gs is None:
        raise GhostscriptNotFound()

    logger.debug(f"Converting {len(data)} bytes of PostScript with {gs}")
    pdf = _run(gs, data)
    with _cache_lock:
        _cache[key] = pdf
        while len(_cache) > _CACHE_SIZE:
            _cache.popitem(last=False)
    return pdf
