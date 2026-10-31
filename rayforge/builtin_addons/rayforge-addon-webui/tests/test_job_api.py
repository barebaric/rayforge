"""Tests for the job and ops preview API endpoints."""

from test_doc_api import _upload_svg


class TestJobApi:
    def test_job_status_idle(self, client):
        response = client.get("/api/v1/job/status")
        assert response.status_code == 200
        data = response.json()
        assert data["active"] is False
        assert data["job"] is None

    def test_job_hold_and_cancel_accepted(self, client):
        response = client.post("/api/v1/job/hold", json={"hold": True})
        assert response.status_code == 200
        response = client.post("/api/v1/job/cancel")
        assert response.status_code == 200

    def test_job_start_empty_doc_clean_error(self, client):
        response = client.post("/api/v1/job/start")
        assert response.status_code in (400, 409)
        assert response.json()["detail"]

    def test_ops_404_without_artifact(self, client):
        _upload_svg(client)
        response = client.get("/api/v1/ops")
        assert response.status_code == 404

    def test_gcode_404_without_artifact(self, client):
        response = client.get("/api/v1/project/gcode")
        assert response.status_code == 404


class TestProjectApi:
    def test_project_save_round_trip(self, client):
        _upload_svg(client)
        response = client.get("/api/v1/project/save")
        assert response.status_code == 200
        content = response.content
        assert content[:2] == b"PK"
        assert len(content) > 100

    def test_project_open_restores_tree(self, client):
        _upload_svg(client)
        saved = client.get("/api/v1/project/save").content
        client.post("/api/v1/doc/new")
        response = client.post(
            "/api/v1/project/open",
            files={
                "file": (
                    "project.ryp",
                    saved,
                    "application/octet-stream",
                )
            },
        )
        assert response.status_code == 200, response.text
        tree = response.json()
        assert len(tree["workpieces"]) == 1
        history = client.get("/api/v1/doc/history").json()
        assert history["dirty"] is False


class TestStepsApi:
    def test_step_types_registered(self, client):
        response = client.get("/api/v1/steps")
        assert response.status_code == 200
        steps = response.json()["steps"]
        assert steps
        names = {step["step_type"] for step in steps}
        assert "ContourStep" in names
        first = steps[0]
        assert first["schema"]
        assert first["schema"][0]["fields"]

    def test_add_step_to_layer(self, client):
        tree = client.get("/api/v1/doc/tree").json()
        layer_uid = tree["layers"][0]["uid"]
        response = client.post(
            f"/api/v1/layers/{layer_uid}/steps",
            json={"step_type": "ContourStep"},
        )
        assert response.status_code == 200, response.text
        step = response.json()
        assert step["step_type"] == "ContourStep"
        assert step["layer_uid"] == layer_uid

        tree = client.get("/api/v1/doc/tree").json()
        assert tree["layers"][0]["step_uids"] == [step["uid"]]

    def test_add_unknown_step_type_404(self, client):
        tree = client.get("/api/v1/doc/tree").json()
        layer_uid = tree["layers"][0]["uid"]
        response = client.post(
            f"/api/v1/layers/{layer_uid}/steps",
            json={"step_type": "Nope"},
        )
        assert response.status_code == 404

    def test_update_step_params(self, client):
        tree = client.get("/api/v1/doc/tree").json()
        layer_uid = tree["layers"][0]["uid"]
        step = client.post(
            f"/api/v1/layers/{layer_uid}/steps",
            json={"step_type": "ContourStep"},
        ).json()
        params = dict(step["params"])
        params["cut_speed"] = 1234
        response = client.put(
            f"/api/v1/steps/{step['uid']}", json={"params": params}
        )
        assert response.status_code == 200, response.text
        assert response.json()["params"]["cut_speed"] == 1234
        history = client.get("/api/v1/doc/history").json()
        assert history["can_undo"] is True

    def test_step_schema_endpoint(self, client):
        tree = client.get("/api/v1/doc/tree").json()
        layer_uid = tree["layers"][0]["uid"]
        step = client.post(
            f"/api/v1/layers/{layer_uid}/steps",
            json={"step_type": "ContourStep"},
        ).json()
        response = client.get(f"/api/v1/steps/{step['uid']}/schema")
        assert response.status_code == 200
        groups = response.json()["groups"]
        assert groups
        fields = groups[0]["fields"]
        keys = {field["key"] for field in fields}
        assert "cut_speed" in keys

    def test_delete_step(self, client):
        tree = client.get("/api/v1/doc/tree").json()
        layer_uid = tree["layers"][0]["uid"]
        step = client.post(
            f"/api/v1/layers/{layer_uid}/steps",
            json={"step_type": "ContourStep"},
        ).json()
        response = client.delete(f"/api/v1/steps/{step['uid']}")
        assert response.status_code == 200
        tree = client.get("/api/v1/doc/tree").json()
        assert tree["layers"][0]["step_uids"] == []


class TestLayersApi:
    def test_add_layer(self, client):
        response = client.post("/api/v1/layers")
        assert response.status_code == 200
        layer = response.json()
        tree = client.get("/api/v1/doc/tree").json()
        assert tree["active_layer_uid"] == layer["uid"]
        assert len(tree["layers"]) == 4

    def test_update_layer(self, client):
        tree = client.get("/api/v1/doc/tree").json()
        uid = tree["layers"][0]["uid"]
        response = client.put(
            f"/api/v1/layers/{uid}",
            json={
                "name": "Renamed",
                "visible": False,
                "color": "#123456",
                "wcs": "G55",
                "rotary_enabled": True,
            },
        )
        assert response.status_code == 200, response.text
        layer = response.json()
        assert layer["name"] == "Renamed"
        assert layer["visible"] is False
        assert layer["color"] == "#123456"
        assert layer["wcs"] == "G55"
        assert layer["rotary_enabled"] is True
        history = client.get("/api/v1/doc/history").json()
        assert history["can_undo"] is True

    def test_activate_layer(self, client):
        tree = client.get("/api/v1/doc/tree").json()
        uid = tree["layers"][1]["uid"]
        response = client.post(f"/api/v1/layers/{uid}/activate")
        assert response.status_code == 200
        assert response.json()["active_layer_uid"] == uid

    def test_reorder_layers(self, client):
        tree = client.get("/api/v1/doc/tree").json()
        uids = [layer["uid"] for layer in tree["layers"]]
        response = client.post(
            "/api/v1/layers/reorder", json={"uids": list(reversed(uids))}
        )
        assert response.status_code == 200
        assert response.json()["layer_uids"] == list(reversed(uids))
        tree = client.get("/api/v1/doc/tree").json()
        assert tree["layers"][0]["uid"] == uids[-1]
        assert tree["layers"][0]["order"] == 0

    def test_delete_layer(self, client):
        tree = client.get("/api/v1/doc/tree").json()
        uid = tree["layers"][1]["uid"]
        response = client.delete(f"/api/v1/layers/{uid}")
        assert response.status_code == 200
        tree = client.get("/api/v1/doc/tree").json()
        assert len(tree["layers"]) == 2

    def test_cannot_delete_last_layer(self, client):
        tree = client.get("/api/v1/doc/tree").json()
        keep = tree["layers"][0]["uid"]
        for layer in tree["layers"][1:]:
            client.delete(f"/api/v1/layers/{layer['uid']}")
        response = client.delete(f"/api/v1/layers/{keep}")
        assert response.status_code == 400

    def test_get_unknown_layer_404(self, client):
        assert client.get("/api/v1/layers/nope").status_code == 404
        assert (
            client.put("/api/v1/layers/nope", json={"name": "x"}).status_code
            == 404
        )
        assert client.delete("/api/v1/layers/nope").status_code == 404
