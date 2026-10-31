"""Tests for the WebSocket event hub."""

from fastapi.testclient import TestClient


def _drain_until_doc_event(websocket, max_events=20):
    for _ in range(max_events):
        event = websocket.receive_json()
        if event.get("type") == "doc" and event.get("kind") != "snapshot":
            return event
    raise AssertionError("No doc event received")


class TestWebSocket:
    def test_connect_receives_initial_snapshot(self, app, editor):
        with (
            TestClient(app) as client,
            client.websocket_connect("/api/v1/ws") as websocket,
        ):
            events = [websocket.receive_json() for _ in range(3)]
        kinds = [event["type"] for event in events]
        assert kinds[0] == "doc"
        assert events[0]["kind"] == "snapshot"
        assert "workpiece_uids" in events[0]["data"]
        assert "machine" in kinds
        assert "viewport" in kinds

    def test_doc_mutation_pushes_event(self, app):
        with (
            TestClient(app) as client,
            client.websocket_connect("/api/v1/ws") as websocket,
        ):
            response = client.post("/api/v1/layers")
            assert response.status_code == 200
            event = _drain_until_doc_event(websocket)
            assert event["type"] == "doc"
            assert "revisions" in event
            assert isinstance(event["revisions"].get("tree"), int)

    def test_tree_revision_advances_and_matches_tree(self, app):
        with TestClient(app) as client:
            first = client.get("/api/v1/doc/tree").json()
            client.post("/api/v1/layers")
            second = client.get("/api/v1/doc/tree").json()
        assert isinstance(first["rev"], int)
        assert second["rev"] > first["rev"]

    def test_import_pushes_structure_event(self, app, editor):
        svg = (
            b'<svg xmlns="http://www.w3.org/2000/svg" width="100" '
            b'height="50" viewBox="0 0 100 50">'
            b'<rect x="10" y="10" width="80" height="30"/></svg>'
        )
        with (
            TestClient(app) as client,
            client.websocket_connect("/api/v1/ws") as websocket,
        ):
            response = client.post(
                "/api/v1/import",
                files={"file": ("t.svg", svg, "image/svg+xml")},
            )
            assert response.status_code == 200
            event = _drain_until_doc_event(websocket)
            assert event["type"] == "doc"

    def test_token_query_param_accepted(self, editor, webui_registry):
        factory = webui_registry.get("webui")
        app = factory(editor=editor, context=editor.context, token="s3cret")
        with (
            TestClient(app) as client,
            client.websocket_connect("/api/v1/ws?token=s3cret") as websocket,
        ):
            event = websocket.receive_json()
            assert event["type"] == "doc"

    def test_wrong_token_rejected(self, editor, webui_registry):
        from starlette.testclient import TestClient as TC
        from starlette.websockets import WebSocketDisconnect

        factory = webui_registry.get("webui")
        app = factory(editor=editor, context=editor.context, token="s3cret")
        with TC(app) as client:
            connected = True
            try:
                with client.websocket_connect("/api/v1/ws?token=wrong"):
                    pass
            except WebSocketDisconnect:
                connected = False
            assert connected is False
