"""Tests for the machine listing, state, and control API endpoints."""

from test_coordinates import build_machine

from rayforge.machine.models.machine import Origin
from rayforge.machine.models.machine_panel import PanelOrientation


class TestMachineListing:
    def test_list_machines(self, client, headless_context):
        response = client.get("/api/v1/machines")
        assert response.status_code == 200
        machines = response.json()["machines"]
        assert len(machines) >= 1
        active = context_active(headless_context)
        assert any(m["id"] == active.id and m["active"] for m in machines)
        entry = machines[0]
        assert entry["name"]
        assert "driver" in entry
        assert "connected" in entry

    def test_machine_state(self, client):
        response = client.get("/api/v1/machine/state")
        assert response.status_code == 200
        state = response.json()
        assert state["status"] == "UNKNOWN"
        assert state["machine_pos"] == [None, None, None]
        assert state["active_wcs"]
        assert state["machine_id"]

    def test_connect_disconnect_unknown_404(self, client):
        response = client.post("/api/v1/machines/nope/connect")
        assert response.status_code == 404
        response = client.post("/api/v1/machines/nope/disconnect")
        assert response.status_code == 404


def context_active(context):
    return context.machine


class TestMachineControl:
    def test_home(self, client):
        response = client.post("/api/v1/machine/home", json={})
        assert response.status_code == 200
        assert response.json()["status"] == "accepted"

    def test_home_single_axis(self, client):
        response = client.post("/api/v1/machine/home", json={"axes": ["x"]})
        assert response.status_code == 200

    def test_home_invalid_axis(self, client):
        response = client.post("/api/v1/machine/home", json={"axes": ["w"]})
        assert response.status_code == 400

    def test_jog(self, client):
        response = client.post(
            "/api/v1/machine/jog",
            json={"direction": "+x", "distance_mm": 10.0},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["deltas"] == {"x": 10.0}

    def test_jog_invalid_direction(self, client):
        response = client.post(
            "/api/v1/machine/jog",
            json={"direction": "north-east", "distance_mm": 10.0},
        )
        assert response.status_code == 400

    def test_jog_zero_distance_rejected(self, client):
        response = client.post(
            "/api/v1/machine/jog",
            json={"direction": "+x", "distance_mm": 0},
        )
        assert response.status_code == 422

    def test_move_to(self, client):
        response = client.post(
            "/api/v1/machine/move-to", json={"x": 10.0, "y": 20.0}
        )
        assert response.status_code == 200

    def test_set_power(self, client):
        response = client.post(
            "/api/v1/machine/set-power", json={"percent": 0.5}
        )
        assert response.status_code == 200

    def test_set_power_unknown_head(self, client):
        response = client.post(
            "/api/v1/machine/set-power",
            json={"percent": 0.5, "head": "no-such-head"},
        )
        assert response.status_code == 404

    def test_set_power_out_of_range(self, client):
        response = client.post(
            "/api/v1/machine/set-power", json={"percent": 1.5}
        )
        assert response.status_code == 422

    def test_set_focus_power(self, client):
        response = client.post(
            "/api/v1/machine/set-focus-power", json={"percent": 0.1}
        )
        assert response.status_code == 200

    def test_run_raw(self, client):
        response = client.post(
            "/api/v1/machine/run-raw", json={"gcode": "G0 X0 Y0"}
        )
        assert response.status_code == 200

    def test_clear_alarm(self, client):
        response = client.post("/api/v1/machine/clear-alarm")
        assert response.status_code == 200

    def test_select_wcs(self, client):
        response = client.post(
            "/api/v1/machine/select-wcs", json={"slot": "G55"}
        )
        assert response.status_code == 200
        assert response.json()["active_wcs"] == "G55"
        state = client.get("/api/v1/machine/state").json()
        assert state["active_wcs"] == "G55"

    def test_select_unknown_wcs(self, client):
        response = client.post(
            "/api/v1/machine/select-wcs", json={"slot": "G99"}
        )
        assert response.status_code == 404

    def test_sync_wcs(self, client):
        response = client.post("/api/v1/machine/sync-wcs")
        assert response.status_code == 200

    def test_run_missing_macro(self, client):
        response = client.post("/api/v1/machine/macros/nope/run")
        assert response.status_code == 404


class TestViewportEndpoint:
    def test_default_machine_flags(self, client, headless_context):
        response = client.get("/api/v1/viewport")
        assert response.status_code == 200
        data = response.json()
        machine = headless_context.machine
        assert data["orientation"] == machine.panel.orientation.value
        assert data["x_axis_right"] == machine.panel.x_axis_right
        assert data["y_axis_down"] == machine.panel.y_axis_down
        assert data["extents"] == list(machine.panel.extents)
        assert data["rotary_active"] is False

    def test_viewport_tracks_machine_changes(self, client, headless_context):
        machine = build_machine(
            headless_context,
            origin=Origin.BOTTOM_RIGHT,
            reverse_x=True,
            reverse_y=False,
            orientation=PanelOrientation.ROTATED_RIGHT,
        )
        headless_context.config.set_machine(machine)
        response = client.get("/api/v1/viewport")
        assert response.status_code == 200
        data = response.json()
        panel = machine.panel
        assert data["orientation"] == panel.orientation.value
        assert data["x_axis_right"] == panel.x_axis_right
        assert data["y_axis_down"] == panel.y_axis_down
        assert data["x_axis_negative"] == panel.x_axis_negative
        assert data["y_axis_negative"] == panel.y_axis_negative
        assert data["extents"] == list(panel.extents)
        assert data["extents"][0] == machine.axis_extents[1]
        assert data["extents"][1] == machine.axis_extents[0]
