# Drivers (Rust/PyO3 device driver library)

The device drivers consumed by Rayforge live in Raygeo. We also own
Raygeo; fix root causes there instead of building workarounds.

Source repository: https://github.com/barebaric/raygeo

- Rust transports and protocol sessions: `src/driver/`
- Python bindings: `src/python/driver/`, exposed as `raygeo.driver.*`
- Pure-Python emulators and CLI: `python/raygeo/driver/`
- Machine-code encoders: `raygeo.ops.convert` (converters, not drivers)

Every protocol driver shares the `raygeo.driver.Driver` base class, and
the Python bindings are real subclasses of it:
`GrblSerialDriver`, `GrblTelnetDriver`, `GrblBluetoothDriver`,
`GrblNetworkDriver`, `GrblSerialSimpleDriver`, `MarlinSerialDriver`,
`SmoothieDriver`, `XtoolM2Driver`, `XtoolNetworkDriver`,
`OctoPrintDriver`, `EpilogDriver`, `NoDeviceDriver` and `RuidaDriver`.
Rayforge's bindings subclass `NativeDriver`
(`rayforge/machine/driver/native.py`) and supply only connection
config, presentation, probing/profile mapping and settings policy. The
remote driver uses the same typed driver metadata.

The following Rayforge driver is still on its Python stack:

- Ruida (`rayforge/machine/driver/ruidarpa`) uses `ruida-pa`. The
  native `raygeo.driver.ruida` stack (codec, USB/UDP/TCP session,
  GlueScript compiler and Ops encoder) is available but not wired up
  yet.

The `python/raygeo/driver/*/emulator.py` modules contain firmware and
server emulators (not mocks) used by the tests.

The former `raydriver` package is archived. Wire-level identifiers
(`_raydriver._tcp`, `_raydriver_protocol`, `raydriver-daemon` and the
`raydriver` config directory) are intentionally unchanged so existing
daemons and machines keep working.
