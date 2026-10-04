# Raydriver (Rust/PyO3 GRBL driver)

Raydriver hosts the Rust-native GRBL serial driver, consumed by the
shell driver `GrblSerialDriver`
(`rayforge/machine/driver/grbl/grbl_serial.py`). We also own it.

Source repository: https://github.com/barebaric/raydriver

`python/raydriver/emulator.py` contains a Grbl 1.1 firmware emulator
(not a mock) that the crate's own tests and
`tests/machine/driver/grbl/test_grbl_serial_driver.py` exercise
through the `MockTransport` test transport. Dialects remain Rayforge
data: command templates are resolved via
`GrblSerialDriver._dialect_templates()`.

Machines saved while the driver carried its experimental
`GrblSerialNextDriver` name resolve through a transitional alias in
`rayforge/machine/driver/__init__.py` (remove with the next
release).
