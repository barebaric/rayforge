"""Tests for the extended machine settings endpoints."""

WCS_SLOT = "G55"


def _machine_id(client) -> str:
    machines = client.get("/api/v1/machines").json()["machines"]
    return machines[0]["id"]


def _settings(client, machine_id) -> dict:
    response = client.get(f"/api/v1/machines/{machine_id}/settings")
    assert response.status_code == 200
    return response.json()


class TestDriverList:
    def test_lists_registered_drivers(self, client):
        response = client.get("/api/v1/drivers")
        assert response.status_code == 200
        names = [entry["name"] for entry in response.json()]
        assert "NoDeviceDriver" in names
        assert "GrblSerialDriver" in names
        for entry in response.json():
            assert entry["maturity"]


class TestDriverSetup:
    def test_get_returns_setup_schema_and_values(self, client):
        machine_id = _machine_id(client)
        response = client.get(f"/api/v1/machines/{machine_id}/driver-setup")
        assert response.status_code == 200
        data = response.json()
        assert data["driver"] == "NoDeviceDriver"
        assert isinstance(data["setup"]["fields"], list)
        assert isinstance(data["values"], dict)
        assert data["device_settings"] is None

    def test_get_grbl_driver_exposes_device_settings(self, client):
        machine_id = _machine_id(client)
        client.put(
            f"/api/v1/machines/{machine_id}/driver",
            json={"driver": "GrblSerialDriver"},
        )
        data = client.get(f"/api/v1/machines/{machine_id}/driver-setup").json()
        assert data["driver"] == "GrblSerialDriver"
        setup_keys = [field["key"] for field in data["setup"]["fields"]]
        assert "port" in setup_keys
        assert data["values"]["baudrate"] == 115200
        assert data["device_settings"]
        assert data["device_settings"][0]["fields"]

    def test_put_writes_values_and_rebuilds_driver(self, client, app):
        machine_id = _machine_id(client)
        client.put(
            f"/api/v1/machines/{machine_id}/driver",
            json={"driver": "GrblSerialDriver"},
        )
        response = client.put(
            f"/api/v1/machines/{machine_id}/driver-setup",
            json={"values": {"port": "/dev/ttyUSB0", "baudrate": 115200}},
        )
        assert response.status_code == 200, response.text
        assert response.json()["values"]["port"] == "/dev/ttyUSB0"

        machine = app.state.editor.context.machine_mgr.get_machine_by_id(
            machine_id
        )
        assert machine.driver_args["port"] == "/dev/ttyUSB0"
        controller = app.state.editor.context.machine_mgr.get_controller(
            machine_id
        )
        assert controller.driver.did_setup is True

    def test_put_unknown_key_rejected(self, client):
        machine_id = _machine_id(client)
        response = client.put(
            f"/api/v1/machines/{machine_id}/driver-setup",
            json={"values": {"bogus_key": 1}},
        )
        assert response.status_code == 404
        assert "bogus_key" in response.json()["detail"]

    def test_switch_driver_resets_args_and_rebuilds(self, client, app):
        machine_id = _machine_id(client)
        client.put(
            f"/api/v1/machines/{machine_id}/driver",
            json={"driver": "GrblSerialDriver"},
        )
        client.put(
            f"/api/v1/machines/{machine_id}/driver-setup",
            json={"values": {"port": "/dev/ttyUSB9"}},
        )
        response = client.put(
            f"/api/v1/machines/{machine_id}/driver",
            json={"driver": "NoDeviceDriver"},
        )
        assert response.status_code == 200
        assert response.json()["driver"] == "NoDeviceDriver"
        machine = app.state.editor.context.machine_mgr.get_machine_by_id(
            machine_id
        )
        assert machine.driver_name == "NoDeviceDriver"
        assert machine.driver_args == {}

    def test_switch_unknown_driver_rejected(self, client):
        machine_id = _machine_id(client)
        response = client.put(
            f"/api/v1/machines/{machine_id}/driver",
            json={"driver": "NotADriver"},
        )
        assert response.status_code == 404


class TestHeads:
    def test_get_lists_laser_heads(self, client):
        machine_id = _machine_id(client)
        heads = _settings(client, machine_id)["heads"]
        assert len(heads) >= 1
        head = heads[0]
        assert head["type"] == "LaserHead"
        assert "max_power" in head
        assert "frame_power" in head

    def test_put_round_trip(self, client):
        machine_id = _machine_id(client)
        head = _settings(client, machine_id)["heads"][0]
        response = client.put(
            f"/api/v1/machines/{machine_id}/heads",
            json={
                "heads": [
                    {
                        "uid": head["uid"],
                        "max_power": 900,
                        "frame_power": 0.25,
                        "focus_power": 0.1,
                        "pointer_offset_enabled": True,
                        "pointer_offset_x": 1.5,
                        "spot_size_mm": [0.08, 0.08],
                    }
                ]
            },
        )
        assert response.status_code == 200, response.text
        updated = response.json()[0]
        assert updated["max_power"] == 900.0
        assert updated["frame_power"] == 0.25
        assert updated["focus_power"] == 0.1
        assert updated["pointer_offset_enabled"] is True
        assert updated["pointer_offset_x"] == 1.5
        assert updated["spot_size_mm"] == [0.08, 0.08]

    def test_put_unknown_uid_rejected(self, client):
        machine_id = _machine_id(client)
        response = client.put(
            f"/api/v1/machines/{machine_id}/heads",
            json={"heads": [{"uid": "nope", "max_power": 10}]},
        )
        assert response.status_code == 404


class TestMacros:
    def test_crud_round_trip(self, client):
        machine_id = _machine_id(client)
        created = client.post(
            f"/api/v1/machines/{machine_id}/macros",
            json={
                "name": "Home then park",
                "code": "G28\nG0 X0",
                "enabled": True,
            },
        )
        assert created.status_code == 200, created.text
        macro = created.json()
        assert macro["name"] == "Home then park"
        assert macro["code"] == ["G28", "G0 X0"]
        assert macro["enabled"] is True

        listed = client.get(f"/api/v1/machines/{machine_id}/macros").json()
        assert any(entry["uid"] == macro["uid"] for entry in listed["macros"])

        edited = client.put(
            f"/api/v1/machines/{machine_id}/macros/{macro['uid']}",
            json={"name": "Renamed", "enabled": False},
        )
        assert edited.status_code == 200
        assert edited.json()["name"] == "Renamed"
        assert edited.json()["enabled"] is False
        assert edited.json()["code"] == ["G28", "G0 X0"]

        deleted = client.delete(
            f"/api/v1/machines/{machine_id}/macros/{macro['uid']}"
        )
        assert deleted.status_code == 200
        listed = client.get(f"/api/v1/machines/{machine_id}/macros").json()
        assert listed["macros"] == []

    def test_update_unknown_macro_rejected(self, client):
        machine_id = _machine_id(client)
        response = client.put(
            f"/api/v1/machines/{machine_id}/macros/nope",
            json={"name": "X"},
        )
        assert response.status_code == 404


class TestWcsOffsets:
    def test_put_updates_offset_and_label(self, client):
        machine_id = _machine_id(client)
        response = client.put(
            f"/api/v1/machines/{machine_id}/wcs/{WCS_SLOT}",
            json={"offset": [1.5, -2.0, 0.0], "label": "Fixture A"},
        )
        assert response.status_code == 200, response.text
        assert response.json()["offset"] == [1.5, -2.0, 0.0]
        assert response.json()["label"] == "Fixture A"

        systems = {
            cs["name"]: cs
            for cs in _settings(client, machine_id)["coordinate_systems"]
        }
        assert systems[WCS_SLOT]["offset"] == [1.5, -2.0, 0.0]
        assert systems[WCS_SLOT]["label"] == "Fixture A"

    def test_put_unknown_slot_rejected(self, client):
        machine_id = _machine_id(client)
        response = client.put(
            f"/api/v1/machines/{machine_id}/wcs/G99",
            json={"offset": [0, 0, 0]},
        )
        assert response.status_code == 404


class TestNogoZones:
    def test_create_list_delete(self, client):
        machine_id = _machine_id(client)
        created = client.post(
            f"/api/v1/machines/{machine_id}/nogo-zones",
            json={
                "name": "Clamp",
                "shape": "rect",
                "params": {"x": 10, "y": 10, "w": 20, "h": 20},
            },
        )
        assert created.status_code == 200, created.text
        zone = created.json()
        assert zone["shape"] == "rect"
        assert zone["enabled"] is True

        zones = _settings(client, machine_id)["nogo_zones"]
        assert any(entry["uid"] == zone["uid"] for entry in zones)

        viewport = client.get("/api/v1/viewport").json()["nogo_zones"]
        assert viewport and len(viewport[0]) == 8

        deleted = client.delete(
            f"/api/v1/machines/{machine_id}/nogo-zones/{zone['uid']}"
        )
        assert deleted.status_code == 200
        assert _settings(client, machine_id)["nogo_zones"] == []

    def test_create_unknown_shape_rejected(self, client):
        machine_id = _machine_id(client)
        response = client.post(
            f"/api/v1/machines/{machine_id}/nogo-zones",
            json={"shape": "spiral", "params": {}},
        )
        assert response.status_code == 404

    def test_delete_unknown_zone_rejected(self, client):
        machine_id = _machine_id(client)
        response = client.delete(
            f"/api/v1/machines/{machine_id}/nogo-zones/nope"
        )
        assert response.status_code == 404


class TestRotaryModules:
    def test_settings_lists_rotary_modules(self, client):
        machine_id = _machine_id(client)
        assert _settings(client, machine_id)["rotary_modules"] == []

    def test_put_edits_module(self, client, app):
        machine_id = _machine_id(client)

        def _create(editor):
            from rayforge.machine.models.rotary_module import RotaryModule

            machine = editor.context.machine_mgr.get_machine_by_id(machine_id)
            module = RotaryModule()
            machine.add_rotary_module(module)
            return module.uid

        import asyncio

        from rayforge.shared import tasker

        async def _run():
            return _create(app.state.editor)

        uid = asyncio.run_coroutine_threadsafe(
            _run(), tasker.task_mgr.loop
        ).result(timeout=10)

        response = client.put(
            f"/api/v1/machines/{machine_id}/rotary-modules/{uid}",
            json={"name": "Roller attachment", "default_diameter": 40.0},
        )
        assert response.status_code == 200, response.text
        module = response.json()[0]
        assert module["name"] == "Roller attachment"
        assert module["default_diameter"] == 40.0

    def test_put_unknown_module_rejected(self, client):
        machine_id = _machine_id(client)
        response = client.put(
            f"/api/v1/machines/{machine_id}/rotary-modules/nope",
            json={"default_diameter": 30},
        )
        assert response.status_code == 404
