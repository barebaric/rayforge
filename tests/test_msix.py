import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_RENDER = _ROOT / "scripts/win/msix/render_manifest.py"
_TEMPLATE = _ROOT / "scripts/win/msix/AppxManifest.xml.in"


def _render(tmp_path: Path, version: str, publisher: str):
    output = tmp_path / "AppxManifest.xml"
    result = subprocess.run(
        [
            sys.executable,
            str(_RENDER),
            "--template",
            str(_TEMPLATE),
            "--output",
            str(output),
            "--version",
            version,
            "--executable",
            "rayforge-v1.15.0.exe",
            "--identity-name",
            "Rayforge.Dev",
            "--publisher",
            publisher,
            "--publisher-display-name",
            "Rayforge",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return result, output


def test_renders_stable_version(tmp_path):
    result, output = _render(tmp_path, "1.15.0", "CN=Rayforge Dev")

    assert result.returncode == 0, result.stderr
    text = output.read_text(encoding="utf-8")
    assert 'Name="Rayforge.Dev"' in text
    assert 'Version="1.15.0.0"' in text
    assert 'Executable="rayforge-v1.15.0.exe"' in text


def test_accepts_four_part_version(tmp_path):
    result, output = _render(tmp_path, "1.0.0.123", "CN=Rayforge Dev")

    assert result.returncode == 0, result.stderr
    text = output.read_text(encoding="utf-8")
    assert 'Version="1.0.0.123"' in text


def test_rejects_prerelease_version(tmp_path):
    result, output = _render(tmp_path, "1.15.0-beta1", "CN=Rayforge Dev")

    assert result.returncode != 0
    assert "stable numeric version" in result.stderr
    assert not output.exists()


def test_escapes_publisher(tmp_path):
    result, output = _render(tmp_path, "1.15.0", "CN=Rayforge & Co")

    assert result.returncode == 0, result.stderr
    text = output.read_text(encoding="utf-8")
    assert 'Publisher="CN=Rayforge &amp; Co"' in text
