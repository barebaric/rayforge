"""Tests for the FastAPI application published by the webui addon."""

import subprocess
import sys

import pytest

from rayforge.config import BUILTIN_ADDONS_DIR

ADDON_ROOT = str(BUILTIN_ADDONS_DIR / "rayforge-addon-webui")


class TestServerApp:
    def test_app_modules_import_without_gtk(self):
        code = (
            f"import sys; sys.path.insert(0, {ADDON_ROOT!r}); "
            "import webui.server; import webui.service; "
            "assert 'gi' not in sys.modules"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert result.returncode == 0, result.stderr

    def test_system_info(self, client):
        response = client.get("/api/v1/system/info")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "rayforge"
        assert data["machine_count"] >= 1

    def test_unknown_route_returns_404(self, client):
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404


class TestTokenAuth:
    @pytest.fixture
    def secured_client(self, editor, webui_registry):
        from fastapi.testclient import TestClient

        factory = webui_registry.get("webui")
        app = factory(editor=editor, context=editor.context, token="s3cret")
        return TestClient(app)

    def test_rejects_missing_token(self, secured_client):
        response = secured_client.get("/api/v1/system/info")
        assert response.status_code == 401

    def test_rejects_wrong_token(self, secured_client):
        response = secured_client.get(
            "/api/v1/system/info",
            headers={"Authorization": "Bearer wrong"},
        )
        assert response.status_code == 401

    def test_accepts_valid_token(self, secured_client):
        response = secured_client.get(
            "/api/v1/system/info",
            headers={"Authorization": "Bearer s3cret"},
        )
        assert response.status_code == 200
