# Raydriver (Rust/PyO3 device driver library)

Raydriver hosts the Rust-native device protocol stacks consumed by
Rayforge. We also own it.

Source repository: https://github.com/barebaric/raydriver

Every protocol driver in raydriver shares the `raydriver.driver.Driver`
base class, and the Python bindings are real subclasses of it:
`GrblSerialDriver`, `GrblTelnetDriver`, `GrblBluetoothDriver`,
`GrblNetworkDriver`, `GrblSerialSimpleDriver`, `MarlinSerialDriver`,
`SmoothieDriver`, `XtoolM2Driver`, `XtoolNetworkDriver`,
`OctoPrintDriver`, `EpilogDriver`, `NoDeviceDriver` and `RuidaDriver`.
Rayforge's bindings subclass `NativeDriver`
(`rayforge/machine/driver/native.py`) and supply only connection
config, presentation, probing/profile mapping and settings policy. The
remote driver uses the same typed driver metadata.

Ruida (`rayforge/machine/driver/ruidarpa`) is native too: its binding
uses the `raydriver.ruida` codec, USB/UDP/TCP session, GlueScript
compiler and Ops encoder. The old ruida-pa TUI RPC host mode was removed;
raydriver's `raydriver analyzer tui` is a protocol monitor, not an RPC
host.

The `python/raydriver/*/emulator.py` modules contain firmware and
server emulators (not mocks) used by the tests.
