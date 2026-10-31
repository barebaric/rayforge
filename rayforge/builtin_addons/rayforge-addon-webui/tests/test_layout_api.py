"""Tests for the layout API endpoints."""

from rayforge.doceditor.layout.base import LayoutStrategy
from rayforge.doceditor.layout.registry import layout_registry


class _FakeStrategy(LayoutStrategy):
    slug = "fake"

    def calculate_deltas(self, context=None):
        return {}


def _import_two_workpieces(client):
    svg = (
        b'<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40">'
        b'<rect x="5" y="5" width="30" height="30"/></svg>'
    )
    for index in range(2):
        response = client.post(
            "/api/v1/import",
            files={"file": (f"part{index}.svg", svg, "image/svg+xml")},
        )
        assert response.status_code == 200, response.text


class TestLayoutsEndpoint:
    def test_lists_registered_strategies(self, client):
        response = client.get("/api/v1/layouts")
        assert response.status_code == 200
        layouts = response.json()
        ids = [entry["id"] for entry in layouts]
        assert "pixel-perfect" in ids
        for entry in layouts:
            assert entry["label"]

    def test_reflects_addon_registrations(self, client):
        layout_registry.register(_FakeStrategy, name="fake-fitter")
        try:
            response = client.get("/api/v1/layouts")
            assert response.status_code == 200
            assert {"id": "fake-fitter", "label": "fake-fitter layout"} in (
                response.json()
            )
        finally:
            layout_registry.unregister("fake-fitter")


class TestLayoutEndpoint:
    def test_unknown_strategy_returns_404(self, client):
        _import_two_workpieces(client)
        response = client.post(
            "/api/v1/layout", json={"strategy": "no-such-thing"}
        )
        assert response.status_code == 404
        assert response.json()["detail"]

    def test_empty_document_returns_400(self, client):
        response = client.post(
            "/api/v1/layout", json={"strategy": "pixel-perfect"}
        )
        assert response.status_code == 400
        assert response.json()["detail"]

    def test_invalid_params_return_400(self, client):
        _import_two_workpieces(client)
        response = client.post(
            "/api/v1/layout",
            json={"strategy": "pixel-perfect", "params": {"bogus": 1}},
        )
        assert response.status_code == 400
        assert "bogus" in response.json()["detail"]

    def test_runs_pixel_perfect_and_is_undoable(self, client):
        _import_two_workpieces(client)
        before = client.get("/api/v1/doc/tree").json()
        positions_before = {
            wp["uid"]: tuple(wp["pos_mm"]) for wp in before["workpieces"]
        }
        response = client.post(
            "/api/v1/layout", json={"strategy": "pixel-perfect"}
        )
        assert response.status_code == 200, response.text
        assert response.json()["moved"] == 2
        after = client.get("/api/v1/doc/tree").json()
        positions_after = {
            wp["uid"]: tuple(wp["pos_mm"]) for wp in after["workpieces"]
        }
        assert positions_after != positions_before
        history = client.get("/api/v1/doc/history").json()
        assert history["can_undo"] is True
        assert history["undo_labels"]
        undo = client.post("/api/v1/doc/undo")
        assert undo.status_code == 200
        restored = client.get("/api/v1/doc/tree").json()
        positions_restored = {
            wp["uid"]: tuple(wp["pos_mm"]) for wp in restored["workpieces"]
        }
        assert positions_restored == positions_before
