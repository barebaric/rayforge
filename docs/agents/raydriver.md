# Raydriver (Rust/PyO3 GRBL driver)

Raydriver hosts the Rust-native GRBL serial driver, consumed by the
shell driver `GrblSerialDriver`
(`rayforge/machine/driver/grbl/grbl_serial.py`). We also own it.

Source repository: https://github.com/barebaric/raydriver

`python/raydriver/emulator.py` contains a Grbl 1.1 firmware emulator
(not a mock) that can be used by tests.
