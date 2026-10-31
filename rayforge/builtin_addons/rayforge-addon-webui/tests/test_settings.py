"""Tests for the settings and machine settings endpoints."""


class TestAppSettings:
    def test_get_returns_whitelisted_fields(self, client):
        response = client.get("/api/v1/settings")
        assert response.status_code == 200
        data = response.json()
        assert data["theme"] == "system"
        assert data["import_dpi"] == 96.0
        assert data["cache_budget_bytes"] > 0
        assert data["unit_preferences"]["length"] == "mm"
        assert "language" in data
        assert "ops_color_mode" in data

    def test_put_round_trip(self, client):
        response = client.put(
            "/api/v1/settings",
            json={
                "values": {
                    "theme": "dark",
                    "language": "de",
                    "import_dpi": 300,
                    "watch_source_files": False,
                    "ops_color_mode": "layer",
                    "unit_preferences": {"length": "cm"},
                }
            },
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["theme"] == "dark"
        assert data["language"] == "de"
        assert data["import_dpi"] == 300.0
        assert data["watch_source_files"] is False
        assert data["ops_color_mode"] == "layer"
        assert data["unit_preferences"]["length"] == "cm"

        refetched = client.get("/api/v1/settings").json()
        assert refetched["theme"] == "dark"
        assert refetched["import_dpi"] == 300.0

    def test_put_updates_config_object(self, client, app):
        client.put("/api/v1/settings", json={"values": {"theme": "light"}})
        config = app.state.editor.context.config
        assert config.theme == "light"

    def test_put_unknown_setting_rejected(self, client):
        response = client.put(
            "/api/v1/settings", json={"values": {"bogus": 1}}
        )
        assert response.status_code == 400
        assert "bogus" in response.json()["detail"]

    def test_put_invalid_theme_rejected(self, client):
        response = client.put(
            "/api/v1/settings", json={"values": {"theme": "hotdog"}}
        )
        assert response.status_code == 400

    def test_put_invalid_dpi_range_rejected(self, client):
        response = client.put(
            "/api/v1/settings", json={"values": {"import_dpi": 99999}}
        )
        assert response.status_code == 400


class TestMachineSettings:
    def _machine_id(self, client) -> str:
        machines = client.get("/api/v1/machines").json()["machines"]
        return machines[0]["id"]

    def test_get_returns_geometry_and_motion_fields(self, client):
        response = client.get(
            f"/api/v1/machines/{self._machine_id(client)}/settings"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["origin"] == "bottom_left"
        assert data["panel_orientation"] == "native"
        assert len(data["axis_extents"]) == 2
        assert len(data["work_margins"]) == 4
        assert data["reverse_x_axis"] is False
        assert data["soft_limits_enabled"] is True
        assert data["jog_speed"] > 0

    def test_put_round_trip_and_persistence(self, client):
        machine_id = self._machine_id(client)
        response = client.put(
            f"/api/v1/machines/{machine_id}/settings",
            json={
                "values": {
                    "name": "Renamed Machine",
                    "axis_extents": [300, 400],
                    "origin": "top_right",
                    "reverse_x_axis": True,
                    "soft_limits_enabled": False,
                    "jog_speed": 2500,
                }
            },
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["name"] == "Renamed Machine"
        assert data["axis_extents"] == [300.0, 400.0]
        assert data["origin"] == "top_right"
        assert data["reverse_x_axis"] is True
        assert data["soft_limits_enabled"] is False
        assert data["jog_speed"] == 2500

        refetched = client.get(
            f"/api/v1/machines/{machine_id}/settings"
        ).json()
        assert refetched == data

    def test_put_clamps_margins_to_extents(self, client):
        machine_id = self._machine_id(client)
        response = client.put(
            f"/api/v1/machines/{machine_id}/settings",
            json={"values": {"work_margins": [10, 20, 5000, 30]}},
        )
        assert response.status_code == 200
        _left, _top, right, _bottom = response.json()["work_margins"]
        assert right < 5000

    def test_put_invalid_origin_rejected(self, client):
        machine_id = self._machine_id(client)
        response = client.put(
            f"/api/v1/machines/{machine_id}/settings",
            json={"values": {"origin": "middle_center"}},
        )
        assert response.status_code == 404

    def test_put_unknown_setting_rejected(self, client):
        machine_id = self._machine_id(client)
        response = client.put(
            f"/api/v1/machines/{machine_id}/settings",
            json={"values": {"bogus": 1}},
        )
        assert response.status_code == 404
        assert "bogus" in response.json()["detail"]

    def test_unknown_machine_returns_404(self, client):
        response = client.get("/api/v1/machines/nope/settings")
        assert response.status_code == 404

    def test_put_writes_machine_yaml(self, client, app):
        import yaml

        machine_id = self._machine_id(client)
        client.put(
            f"/api/v1/machines/{machine_id}/settings",
            json={
                "values": {
                    "origin": "bottom_right",
                    "axis_extents": [250, 350],
                }
            },
        )
        machine_mgr = app.state.editor.context.machine_mgr
        path = machine_mgr.filename_from_id(machine_id)
        assert path.exists()
        data = yaml.safe_load(path.read_text())
        assert data["machine"]["origin"] == "bottom_right"
        assert data["machine"]["axis_extents"] == [250, 350]
