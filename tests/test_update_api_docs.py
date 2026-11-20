import importlib.util
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "update_api_docs.py"


def load_module() -> Any:
    spec = importlib.util.spec_from_file_location("update_api_docs", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def module() -> Any:
    return load_module()


@pytest.mark.parametrize(
    "line, expected",
    [
        ("raygeo==1.86.0", ("v1.86.0", "tag")),
        (
            "raygeo @ git+https://github.com/barebaric/raygeo@main",
            ("main", "branch"),
        ),
        (
            "raygeo @ git+https://github.com/barebaric/raygeo@v1.86.0",
            ("v1.86.0", "tag"),
        ),
        (
            (
                "raygeo @ git+https://github.com/barebaric/raygeo"
                "@acebab7017d1e7ec86f815beb114b371da3ee39a"
            ),
            ("acebab7017d1e7ec86f815beb114b371da3ee39a", "commit"),
        ),
    ],
)
def test_get_pinned_ref(module, monkeypatch, tmp_path, line, expected):
    requirements = tmp_path / "requirements.txt"
    requirements.write_text(line + "\n", encoding="utf-8")
    monkeypatch.setattr(module, "REQUIREMENTS", requirements)

    pinned = module._get_pinned_ref("raygeo")

    assert (pinned.ref, pinned.kind) == expected


@pytest.mark.parametrize(
    "pinned, expected",
    [
        (
            ("v1.86.0", "tag"),
            (
                "https://github.com/barebaric/raygeo"
                "/archive/refs/tags/v1.86.0.tar.gz"
            ),
        ),
        (
            ("main", "branch"),
            (
                "https://github.com/barebaric/raygeo"
                "/archive/refs/heads/main.tar.gz"
            ),
        ),
        (
            ("acebab7017d1e7ec86f815beb114b371da3ee39a", "commit"),
            (
                "https://github.com/barebaric/raygeo"
                "/archive/acebab7017d1e7ec86f815beb114b371da3ee39a.tar.gz"
            ),
        ),
    ],
)
def test_archive_url(module, pinned, expected):
    package = module.ApiDocsPackage(
        name="raygeo",
        repo_url="https://github.com/barebaric/raygeo",
        output_dir=Path("docs"),
    )

    url = module._archive_url(package, module.PinnedRef(*pinned))

    assert url == expected
