# Raydriver (Rust/PyO3 GRBL driver)

Raydriver hosts the Rust-native GRBL serial driver, consumed by the
shell driver `GrblSerialNextDriver`
(`rayforge/machine/driver/grbl/serial_next.py`). We also own it.

Source repository: https://github.com/barebaric/raydriver

`python/raydriver/emulator.py` contains a Grbl 1.1 firmware emulator
(not a mock) that the crate's own tests and
`tests/machine/driver/grbl/test_serial_next.py` exercise through the
`MockTransport` test transport. Dialects remain Rayforge data:
command templates are resolved via
`GrblSerialNextDriver._dialect_templates()`.
