# Raydriver (Rust/PyO3 device driver library)

Raydriver hosts the Rust-native device protocol stacks consumed by
Rayforge's driver shells. We also own it.

Source repository: https://github.com/barebaric/raydriver

It provides GRBL serial/telnet and ESP3D/FluidNC network sessions,
Smoothieware and Marlin sessions, the xTool WiFi sessions, remote
daemon/client support, and device discovery. Driver shells live under
`rayforge/machine/driver/`.

`python/raydriver/grbl/emulator.py`, `smoothie/emulator.py`,
`marlin/emulator.py` and `esp3d/emulator.py` contain firmware
emulators (not mocks) used by the tests.
