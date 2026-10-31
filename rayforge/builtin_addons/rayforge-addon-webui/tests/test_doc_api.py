"""Tests for the document-level API endpoints."""

import time

SVG_CONTENT = (
    b'<svg xmlns="http://www.w3.org/2000/svg" width="100" height="50" '
    b'viewBox="0 0 100 50">'
    b'<rect x="10" y="10" width="80" height="30"/></svg>'
)


def _upload_svg(client, name="test.svg"):
    response = client.post(
        "/api/v1/import",
        files={"file": (name, SVG_CONTENT, "image/svg+xml")},
    )
    assert response.status_code == 200, response.text
    return response.json()


class TestDocTree:
    def test_fresh_doc_has_three_layers(self, client):
        response = client.get("/api/v1/doc/tree")
        assert response.status_code == 200
        tree = response.json()
        assert len(tree["layers"]) == 3
        assert tree["active_layer_uid"] == tree["layers"][0]["uid"]
        assert tree["workpieces"] == []
        assert tree["groups"] == []
        first = tree["layers"][0]
        assert first["order"] == 0
        assert first["name"]
        assert first["color"]
        assert first["workflow_uid"]
        assert first["step_uids"] == []

    def test_tree_round_trip_after_import(self, client):
        result = _upload_svg(client)
        assert len(result["created"]) == 1
        uid = result["created"][0]
        tree = client.get("/api/v1/doc/tree").json()
        workpieces = tree["workpieces"]
        assert len(workpieces) == 1
        wp = workpieces[0]
        assert wp["uid"] == uid
        assert wp["layer_uid"] == tree["layers"][0]["uid"]
        pos_x, pos_y = wp["pos_mm"]
        width, height = wp["size_mm"]
        assert 0 <= pos_x <= 200
        assert 0 <= pos_y <= 200
        assert 15 < width < 30
        assert 5 < height < 12
        assert wp["angle_deg"] == 0
        assert wp["geometry_revision"] >= 0
        assert wp["transform_revision"] >= 0


class TestImport:
    def test_analyze_svg(self, client):
        response = client.post(
            "/api/v1/import/analyze",
            files={"file": ("test.svg", SVG_CONTENT, "image/svg+xml")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["supported"] is True
        assert data["action"] in ("direct_load", "interactive_config")
        assert data["importer"] == "SvgImporter"
        assert "DIRECT_VECTOR" in data["features"]

    def test_analyze_unsupported(self, client):
        response = client.post(
            "/api/v1/import/analyze",
            files={"file": ("x.xyz", b"data", "application/octet-stream")},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["supported"] is False
        assert data["action"] == "unsupported"

    def test_import_adds_workpiece_with_undo(self, client):
        result = _upload_svg(client)
        assert len(result["created"]) == 1
        history = client.get("/api/v1/doc/history").json()
        assert history["can_undo"] is True

    def test_import_unsupported_type_fails(self, client):
        response = client.post(
            "/api/v1/import",
            files={"file": ("x.xyz", b"not-a-vector", "application/octets")},
        )
        assert response.status_code == 400


class TestHistory:
    def test_fresh_doc_not_dirty(self, client):
        history = client.get("/api/v1/doc/history").json()
        assert history["can_undo"] is False
        assert history["can_redo"] is False
        assert history["dirty"] is False

    def test_undo_redo_round_trip(self, client):
        result = _upload_svg(client)
        uid = result["created"][0]
        tree = client.get("/api/v1/doc/tree").json()
        original_pos = tree["workpieces"][0]["pos_mm"]

        response = client.put(
            f"/api/v1/items/{uid}/transform",
            json={"pos_mm": [30.0, 40.0]},
        )
        assert response.status_code == 200, response.text

        tree = client.get("/api/v1/doc/tree").json()
        moved = tree["workpieces"][0]["pos_mm"]
        assert moved[0] == 30.0
        assert moved[1] == 40.0

        history = client.get("/api/v1/doc/history").json()
        assert history["can_undo"] is True
        assert history["can_redo"] is False
        assert history["dirty"] is True
        assert history["undo_labels"]

        response = client.post("/api/v1/doc/undo")
        assert response.status_code == 200
        assert response.json()["can_redo"] is True
        tree = client.get("/api/v1/doc/tree").json()
        undone = tree["workpieces"][0]["pos_mm"]
        assert undone == original_pos

        response = client.post("/api/v1/doc/redo")
        assert response.status_code == 200
        tree = client.get("/api/v1/doc/tree").json()
        redone = tree["workpieces"][0]["pos_mm"]
        assert redone == [30.0, 40.0]

    def test_doc_new_resets(self, client):
        _upload_svg(client)
        response = client.post("/api/v1/doc/new")
        assert response.status_code == 200
        tree = response.json()
        assert len(tree["layers"]) == 3
        assert tree["workpieces"] == []
        history = client.get("/api/v1/doc/history").json()
        assert history["can_undo"] is False
        assert history["dirty"] is False


class TestItemTransform:
    def test_size_and_angle(self, client):
        uid = _upload_svg(client)["created"][0]
        tree = client.get("/api/v1/doc/tree").json()
        original_size = tree["workpieces"][0]["size_mm"]

        response = client.put(
            f"/api/v1/items/{uid}/transform",
            json={"size_mm": [40.0, 20.0], "angle_deg": 45.0},
        )
        assert response.status_code == 200, response.text
        tree = client.get("/api/v1/doc/tree").json()
        wp = tree["workpieces"][0]
        assert wp["size_mm"][0] == 40.0
        assert wp["size_mm"][1] == 20.0
        assert wp["angle_deg"] == 45.0
        assert wp["transform_revision"] >= 1

        client.post("/api/v1/doc/undo")
        tree = client.get("/api/v1/doc/tree").json()
        wp = tree["workpieces"][0]
        assert wp["size_mm"] == original_size
        assert wp["angle_deg"] == 0

    def test_unknown_item_404(self, client):
        response = client.put(
            "/api/v1/items/nope/transform",
            json={"pos_mm": [1.0, 2.0]},
        )
        assert response.status_code == 404

    def test_geometry_endpoint(self, client):
        uid = _upload_svg(client)["created"][0]
        response = client.get(f"/api/v1/items/{uid}/geometry")
        assert response.status_code == 200
        data = response.json()
        assert data["polylines"]
        assert data["outline"] is not None
        outline = data["outline"]
        assert outline[2] > 0
        assert outline[3] > 0

    def test_geometry_unknown_item_404(self, client):
        response = client.get("/api/v1/items/nope/geometry")
        assert response.status_code == 404

    def test_geometry_curves_are_linearized(self, client):
        svg = (
            b'<svg xmlns="http://www.w3.org/2000/svg" width="40" height="40" '
            b'viewBox="0 0 40 40">'
            b'<circle cx="20" cy="20" r="15" fill="none" stroke="black"/>'
            b"</svg>"
        )
        response = client.post(
            "/api/v1/import",
            files={"file": ("circle.svg", svg, "image/svg+xml")},
        )
        assert response.status_code == 200, response.text
        uid = response.json()["created"][0]
        data = client.get(f"/api/v1/items/{uid}/geometry").json()
        assert data["polylines"]
        longest = max(len(poly) for poly in data["polylines"])
        assert longest > 16

    def test_texture_png(self, client):
        uid = _upload_svg(client)["created"][0]
        response = client.get(f"/api/v1/items/{uid}/texture.png")
        if response.status_code == 200:
            assert response.headers["content-type"] == "image/png"
            assert response.headers["etag"]
            assert response.content[:8] == b"\x89PNG\r\n\x1a\n"
        else:
            assert response.status_code == 404


class TestItemEditCommands:
    def test_delete_and_undelete(self, client):
        uid = _upload_svg(client)["created"][0]
        response = client.post("/api/v1/items/delete", json={"uids": [uid]})
        assert response.status_code == 200
        assert response.json()["removed"] == [uid]
        tree = client.get("/api/v1/doc/tree").json()
        assert tree["workpieces"] == []
        client.post("/api/v1/doc/undo")
        tree = client.get("/api/v1/doc/tree").json()
        assert len(tree["workpieces"]) == 1

    def test_duplicate(self, client):
        uid = _upload_svg(client)["created"][0]
        response = client.post("/api/v1/items/duplicate", json={"uids": [uid]})
        assert response.status_code == 200
        created = response.json()["created"]
        assert len(created) == 1
        assert created[0] != uid
        tree = client.get("/api/v1/doc/tree").json()
        assert len(tree["workpieces"]) == 2

    def test_flip(self, client):
        uid = _upload_svg(client)["created"][0]
        before = client.get(f"/api/v1/items/{uid}/geometry").json()
        tree = client.get("/api/v1/doc/tree").json()
        size_before = tree["workpieces"][0]["size_mm"]
        response = client.post(
            "/api/v1/items/flip",
            json={"uids": [uid], "axis": "horizontal"},
        )
        assert response.status_code == 200
        after = client.get(f"/api/v1/items/{uid}/geometry").json()
        assert after["outline"] == before["outline"]
        tree = client.get("/api/v1/doc/tree").json()
        assert tree["workpieces"][0]["size_mm"] == size_before
        history = client.get("/api/v1/doc/history").json()
        assert history["can_undo"] is True

    def test_flip_invalid_axis(self, client):
        uid = _upload_svg(client)["created"][0]
        response = client.post(
            "/api/v1/items/flip", json={"uids": [uid], "axis": "diagonal"}
        )
        assert response.status_code == 400

    def test_delete_missing_uid_404(self, client):
        response = client.post(
            "/api/v1/items/delete", json={"uids": ["missing"]}
        )
        assert response.status_code == 404


class TestItemAlign:
    def _wait_for_layout(self, client, label, timeout=2.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            history = client.get("/api/v1/doc/history").json()
            if label in history["undo_labels"]:
                return history
            time.sleep(0.02)
        raise AssertionError(f"'{label}' never appeared in undo history")

    def test_align_top_records_undo(self, client):
        first = _upload_svg(client, "a.svg")["created"][0]
        second = _upload_svg(client, "b.svg")["created"][0]
        response = client.put(
            f"/api/v1/items/{second}/transform",
            json={"pos_mm": [50.0, 60.0]},
        )
        assert response.status_code == 200, response.text

        response = client.post(
            "/api/v1/items/align",
            json={"uids": [first, second], "mode": "top"},
        )
        assert response.status_code == 200, response.text
        self._wait_for_layout(client, "Align Top")

        tree = client.get("/api/v1/doc/tree").json()
        by_uid = {wp["uid"]: wp for wp in tree["workpieces"]}
        tops = [
            by_uid[first]["pos_mm"][1] + by_uid[first]["size_mm"][1],
            by_uid[second]["pos_mm"][1] + by_uid[second]["size_mm"][1],
        ]
        assert abs(tops[0] - tops[1]) < 1e-6

        history = client.get("/api/v1/doc/history").json()
        assert history["can_undo"] is True
        response = client.post("/api/v1/doc/undo")
        assert response.status_code == 200
        tree = client.get("/api/v1/doc/tree").json()
        by_uid = {wp["uid"]: wp for wp in tree["workpieces"]}
        assert by_uid[second]["pos_mm"] == [50.0, 60.0]

    def test_align_unknown_mode_400(self, client):
        uid = _upload_svg(client)["created"][0]
        response = client.post(
            "/api/v1/items/align",
            json={"uids": [uid], "mode": "middle"},
        )
        assert response.status_code == 400

    def test_align_single_item_400(self, client):
        uid = _upload_svg(client)["created"][0]
        response = client.post(
            "/api/v1/items/align",
            json={"uids": [uid], "mode": "left"},
        )
        assert response.status_code == 400


def _run_on_core(app, fn):
    """Run ``fn`` (returning a coroutine fn result) on the core loop."""
    import asyncio

    from rayforge.shared import tasker

    async def _run():
        return fn()

    future = asyncio.run_coroutine_threadsafe(_run(), tasker.task_mgr.loop)
    return future.result(timeout=10)


class TestStockItems:
    def _add_stock(self, app) -> str:
        from raygeo.geo import Geometry

        geometry = Geometry()
        geometry.move_to(0, 0)
        geometry.line_to(100, 0)
        geometry.line_to(100, 50)
        geometry.line_to(0, 50)
        geometry.close_path()

        def _create(editor):
            from rayforge.core.stock import StockItem
            from rayforge.core.stock_asset import StockAsset

            asset = StockAsset(name="Test Sheet", geometry=geometry)
            editor.doc.add_asset(asset, silent=True)
            item = StockItem(stock_asset_uid=asset.uid, name="Test Sheet")
            item.set_size(100, 50)
            item.pos = (10, 20)
            editor.doc.add_child(item)
            return item.uid

        return _run_on_core(app, lambda: _create(app.state.editor))

    def test_tree_contains_stock_items(self, client, app):
        uid = self._add_stock(app)
        tree = client.get("/api/v1/doc/tree").json()
        stock = tree["stock_items"]
        assert len(stock) == 1
        entry = stock[0]
        assert entry["uid"] == uid
        assert entry["type"] == "stock"
        assert entry["name"] == "Test Sheet"
        assert entry["pos_mm"] == [10.0, 20.0]
        assert entry["size_mm"] == [100.0, 50.0]
        assert entry["angle_deg"] == 0

    def test_stock_geometry_endpoint(self, client, app):
        uid = self._add_stock(app)
        response = client.get(f"/api/v1/items/{uid}/geometry")
        assert response.status_code == 200
        data = response.json()
        assert data["outline"] == [10.0, 20.0, 100.0, 50.0]
        assert data["polylines"]
